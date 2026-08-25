module.exports = {
  run: [
    // Accept user input for inference parameters
    {
      method: "input",
      params: {
        title: "4DAnyone 4D Inference (CLI Form)",
        description: "Configure multi-view generation. For non-tech users, default values work great out of the box!",
        form: [
          {
            key: "video_path",
            title: "Input Video Path",
            description: "Path to your input video file (e.g. data/source/pexels/2785536-uhd_2160_3840_25fps.mp4)",
            default: "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4"
          },
          {
            key: "views_per_layer",
            title: "Views Per Layer (Camera Count)",
            description: "Number of camera angles placed around the person in a 360° circle (6 = fast preview, 12 or 24 = studio quality)",
            default: "6"
          },
          {
            key: "views_per_group",
            title: "Views Per Group (VRAM / Memory Saver)",
            description: "How many cameras the AI computes simultaneously. Use '2' for normal gaming GPUs (<16GB VRAM), '3' for 24GB, or 'auto'",
            default: "auto"
          },
          {
            key: "layer_pitches",
            title: "Layer Pitches (Camera Tilt in degrees)",
            description: "Vertical camera elevation. [15] is standard eye-level; [0] is waist level; [-10, 15, 35] creates 3 camera tiers",
            default: "[15]"
          },
          {
            key: "yaw_span",
            title: "Yaw Span (Orbit Angle in degrees)",
            description: "How far around the person to film. 360 = full 360° turnaround circle; 180 = front half-circle",
            default: "360"
          }
        ]
      }
    },
    // Run inference
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python inference.py --video_path \"{{input.video_path}}\" --views_per_layer {{input.views_per_layer}} --views_per_group {{input.views_per_group}} --layer_pitches \"{{input.layer_pitches}}\" --yaw_span {{input.yaw_span}}"
        ]
      }
    },
    // Open the output folder
    {
      method: "fs.open",
      params: {
        path: "app/data/fdanyone"
      }
    },
    {
      method: "notify",
      params: {
        html: "4DAnyone inference complete! Output folder has been opened."
      }
    }
  ]
}
