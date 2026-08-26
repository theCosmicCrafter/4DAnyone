import torch
import torch.nn as nn
from safetensors import safe_open
from comfy_kitchen.backends.triton.quantization import int8_linear

class QuantizedLinearConvRot(nn.Module):
    """
    Custom INT8 linear module matching ComfyUI's native Triton kernel dispatch.
    Fuses Hadamard rotation, row-wise quantization, tensor-core GEMM, scale epilogue, 
    and bias addition directly into a single pass.
    """
    def __init__(self, in_features, out_features, group_size=256, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Native [out_features, in_features] layout matching safetensors and Triton kernel
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32, device=device))
        self.register_buffer("bias", torch.zeros((out_features,), dtype=dtype, device=device))
        
    def forward(self, x):
        return int8_linear(
            x,
            self.weight,
            self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=self.group_size,
        )

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
    
    # 1. Swap Linear -> QuantizedLinearConvRot for all quantized layers
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name
            if full_child_name in quantized_prefixes and isinstance(child, nn.Linear):
                quant_layer = QuantizedLinearConvRot(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    group_size=group_size,
                    device=device,
                    dtype=dtype
                )
                setattr(module, child_name, quant_layer)
                
    # 2. Iterate and stream weights directly from disk to GPU bypassing Windows mmap limits
    logger.info(f"Streaming INT8 layers from disk to device (bypassing mmap)...")
    
    with open(checkpoint_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = int.from_bytes(header_size_bytes, "little")
        import json
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        
        data_offset_base = 8 + header_size
        
        for k, info in header.items():
            if k == "__metadata__":
                continue
                
            offsets = info["data_offsets"]
            shape = info["shape"]
            dtype_str = info["dtype"]
            
            f.seek(data_offset_base + offsets[0])
            buffer = f.read(offsets[1] - offsets[0])
            
            import struct
            import numpy as np
            
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
            if is_quantized:
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
                
    logger.info("Successfully loaded and initialized INT8 ConvRot model using ComfyUI Triton backend.")
    return model
