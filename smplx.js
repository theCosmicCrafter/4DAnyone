module.exports = {
  run: [
    {
      method: "input",
      params: {
        title: "SMPL-X Body Model Setup & License",
        description: "SMPL-X is a 3D human mesh model licensed by the Max Planck Institute.\n\nQuick 3-Step Setup:\n1. Open https://smpl-x.is.tue.mpg.de/ in your browser and register (free).\n2. Download 'models_smplx_v1_1.zip' to your standard Downloads folder.\n3. Click Submit below to auto-import it (or paste a custom file path).",
        form: [
          {
            key: "archive_path",
            title: "Custom Path (Optional: leave blank to auto-detect from ~/Downloads)",
            placeholder: "Leave empty to auto-import models_smplx_v1_1.zip from Downloads",
            type: "string"
          },
          {
            key: "recycle_archive",
            title: "Recycle / Delete ZIP after import to save disk space?",
            type: "boolean",
            default: true
          }
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "{{exists('app') ? 'app' : '.'}}",
        message: [
          "{{input && input.archive_path ? `python scripts/download_smplx.py --archive_path \"${input.archive_path}\" --recycle_archive ${input.recycle_archive}` : `python scripts/download_smplx.py --recycle_archive ${input.recycle_archive}`}}"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "SMPL-X body model imported successfully! You are ready to Run Inference."
      }
    }
  ]
}
