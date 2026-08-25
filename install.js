module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Initialize the GVHMR submodule
    {
      when: "{{!exists('third_party/GVHMR/README.md') && !exists('app/third_party/GVHMR/README.md')}}",
      method: "shell.run",
      params: {
        path: "{{exists('app') ? 'app' : '.'}}",
        message: [
          "git submodule update --init third_party/GVHMR",
        ]
      }
    },
    // 2. Install Python dependencies with uv
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "{{exists('app') ? 'app' : '.'}}",
        message: [
          "uv pip install -r requirements.txt",
        ]
      }
    },
    // 3. Install correct PyTorch for platform/GPU
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "{{exists('app') ? 'app' : '.'}}",
        }
      }
    },
    // 4. Download 4DAnyone public model checkpoints from Hugging Face
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "{{exists('app') ? 'app' : '.'}}",
        message: [
          "python scripts/download_model.py",
        ]
      }
    },
    // 5. Download bundled example video clip
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "{{exists('app') ? 'app' : '.'}}",
        message: [
          "python scripts/download_example.py",
        ]
      }
    },
    // 6. If SMPL-X is missing, prompt user with Option D Setup Wizard Modal
    {
      when: "{{!exists('models/body_models/smplx/SMPLX_NEUTRAL.npz') && !exists('app/models/body_models/smplx/SMPLX_NEUTRAL.npz')}}",
      method: "input",
      params: {
        title: "SMPL-X Body Model Setup (Max Planck Institute License)",
        description: "SMPL-X is separately licensed by the Max Planck Institute.\n\nQuick 3-Step Setup:\n1. Open https://smpl-x.is.tue.mpg.de/ in your browser and register (free).\n2. Download 'models_smplx_v1_1.zip' to your normal Downloads folder.\n3. Click Submit below to auto-import it (or paste a custom file path).",
        form: [
          {
            key: "archive_path",
            title: "Custom Path (Optional: leave blank to auto-detect from ~/Downloads)",
            placeholder: "Leave empty to auto-import models_smplx_v1_1.zip from Downloads",
            type: "string"
          }
        ]
      }
    },
    // 7. Import SMPL-X
    {
      when: "{{!exists('models/body_models/smplx/SMPLX_NEUTRAL.npz') && !exists('app/models/body_models/smplx/SMPLX_NEUTRAL.npz')}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "{{exists('app') ? 'app' : '.'}}",
        message: [
          "{{input && input.archive_path ? `python scripts/download_smplx.py --archive_path \"${input.archive_path}\"` : `python scripts/download_smplx.py`}}"
        ]
      }
    },
    // 8. Completion Notification
    {
      method: "notify",
      params: {
        html: "4DAnyone installation & setup complete! Click 'Start Web UI' to launch."
      }
    }
  ]
}
