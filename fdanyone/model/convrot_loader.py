import torch
import torch.nn as nn
from safetensors import safe_open
from comfy_kitchen.backends.eager.quantization import mm_int8
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

class QuantizedLinearConvRot(nn.Module):
    """
    Custom INT8 linear module for ConvRot INT8 quantized weights.
    Applies online group-wise Hadamard rotation to activations, dynamically 
    quantizes to INT8, and uses tensor core INT8 GEMM.
    """
    def __init__(self, in_features, out_features, group_size, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Buffers: They move to device automatically when .to() is called
        # Weight is kept transposed [in_features, out_features] because mm_int8 does (M,K) @ (K,N)
        self.register_buffer("weight", torch.empty((in_features, out_features), dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32, device=device))
        self.register_buffer("bias", torch.zeros((out_features,), dtype=dtype, device=device))
        
        # Build Hadamard matrix for this group_size
        H = _build_hadamard(group_size, device=device, dtype=torch.float32)
        self.register_buffer("H", H)
        
    def forward(self, x):
        # 1. Store original shape [..., in_features]
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # 2. Online Activation Rotation: x_rot = x @ H
        x_rot = _rotate_activation(x_flat.float(), self.H.float(), self.group_size)
        
        # 3. Dynamic INT8 Row-wise Quantization of Activations
        row_max = x_rot.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        x_scale = 127.0 / row_max
        x_int8 = (x_rot * x_scale).round().clamp(-127, 127).to(torch.int8)
        
        # 4. INT8 GEMM -> INT32 output
        # x_int8 is [M, K], self.weight is [K, N] => out_int32 is [M, N]
        out_int32 = mm_int8(x_int8, self.weight)
        
        # 5. Fused Dequantization Epilogue + Bias
        # dequant = out_int32 * (1.0 / x_scale) * weight_scale^T
        dequant_x_scale = 1.0 / x_scale
        out_fp32 = out_int32.float() * dequant_x_scale * self.weight_scale.t()
        
        out = out_fp32.to(x.dtype) + self.bias
        return out.view(*orig_shape[:-1], self.out_features)

def load_convrot_model(model, checkpoint_path, device, dtype):
    """
    Swaps standard `nn.Linear` layers in the provided model
    for `QuantizedLinearConvRot` based on the safetensors metadata,
    and loads the weights into the new layers via state_dict assignment.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loading INT8 ConvRot weights from: {checkpoint_path}")
    
    quantized_prefixes = set()
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".comfy_quant"):
                quantized_prefixes.add(k.replace(".comfy_quant", ""))
            
    group_size = 256
    
    # 1. Swap standard nn.Linear layers for QuantizedLinearConvRot on 'meta' device
    for name, module in list(model.named_modules()):
        if name in quantized_prefixes and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            quantized_layer = QuantizedLinearConvRot(
                in_features=module.in_features, 
                out_features=module.out_features, 
                group_size=group_size, 
                device="meta", 
                dtype=dtype
            )
            setattr(parent, child_name, quantized_layer)
            
    import json
    import struct
    import numpy as np
    
    # 2. Iterate and stream weights directly from disk to GPU bypassing Windows mmap limits
    logger.info(f"Streaming INT8 layers from disk to device (bypassing mmap)...")
    
    with open(checkpoint_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size).decode("utf-8")
        metadata = json.loads(header_json)
        data_start = 8 + header_size
        
        for k, info in metadata.items():
            if k == "__metadata__" or k.endswith(".comfy_quant"):
                continue
                
            # Read chunk manually
            dtype_str = info["dtype"]
            shape = info["shape"]
            offsets = info["data_offsets"]
            length = offsets[1] - offsets[0]
            
            f.seek(data_start + offsets[0])
            buffer = f.read(length)
            
            if dtype_str == "I8":
                arr = np.frombuffer(buffer, dtype=np.int8)
                tensor = torch.from_numpy(arr).view(shape)
            elif dtype_str == "F32":
                arr = np.frombuffer(buffer, dtype=np.float32)
                tensor = torch.from_numpy(arr).view(shape)
            elif dtype_str == "BF16":
                arr = np.frombuffer(buffer, dtype=np.uint16)
                tensor = torch.from_numpy(arr.copy()).view(torch.bfloat16).view(shape)
            elif dtype_str == "F16":
                arr = np.frombuffer(buffer, dtype=np.float16)
                tensor = torch.from_numpy(arr).view(shape)
            else:
                logger.warning(f"Unsupported dtype {dtype_str} for {k}")
                continue
                
            is_quantized = any(k.startswith(p + ".") for p in quantized_prefixes)
            if is_quantized and k.endswith(".weight"):
                tensor = tensor.t().contiguous().to(device)
            elif is_quantized:
                tensor = tensor.to(device)
            elif not is_quantized and (tensor.dtype in (torch.float32, torch.float16)):
                tensor = tensor.to(device=device, dtype=dtype)
            else:
                tensor = tensor.to(device)
                
            # Manually assign to the module
            module_name = ".".join(k.split(".")[:-1])
            param_name = k.split(".")[-1]
            try:
                mod = model.get_submodule(module_name)
                if isinstance(getattr(mod, param_name), nn.Parameter):
                    setattr(mod, param_name, nn.Parameter(tensor, requires_grad=False))
                else:
                    setattr(mod, param_name, tensor)
            except AttributeError:
                logger.warning(f"Warning: skipped {k} (not in model)")
                
    logger.info("Successfully loaded ConvRot INT8 model.")
    return model
