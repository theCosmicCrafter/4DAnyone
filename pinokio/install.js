module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Clone the repository
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/theCosmicCrafter/4DAnyone.git app",
        ]
      }
    },
    // 2. Initialize the GVHMR submodule
    {
      when: "{{!exists('app/third_party/GVHMR/README.md')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: [
          "git submodule update --init third_party/GVHMR",
        ]
      }
    },
    // 3. Install Python dependencies with uv
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
        ]
      }
    },
    // 4. Install correct PyTorch for platform/GPU
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
    // 5. Download 4DAnyone public model checkpoints from Hugging Face
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python scripts/download_model.py",
        ]
      }
    },
    // 6. Download bundled example video clip
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python scripts/download_example.py",
        ]
      }
    },
    // 7. If SMPL-X is missing, prompt user with Option D Setup Wizard Modal
    {
      when: "{{!exists('app/models/body_models/smplx/SMPLX_NEUTRAL.npz')}}",
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
    // 8. Import SMPL-X
    {
      when: "{{!exists('app/models/body_models/smplx/SMPLX_NEUTRAL.npz')}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "{{input && input.archive_path ? `python scripts/download_smplx.py --archive_path \"${input.archive_path}\"` : `python scripts/download_smplx.py`}}"
        ]
      }
    },
    // 9. Completion Notification
    {
      method: "notify",
      params: {
        html: "4DAnyone installation & setup complete! Click 'Run Inference' to start generating 4D videos."
      }
    }
  ]
}
