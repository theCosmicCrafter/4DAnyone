const path = require('path')
module.exports = {
  version: "7.0",
  title: "4DAnyone",
  description: "Create Anyone in 4D from a Casual Monocular Video. Multi-view video synthesis & open-source 4DGS reconstruction.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env") || info.exists("env")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      run: info.running("run.js"),
      smplx: info.running("smplx.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Starting Web UI...",
            href: "start.js",
          }]
        }
      } else if (running.run) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Running CLI Inference",
          href: "run.js",
        }]
      } else if (running.smplx) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "SMPL-X Setup",
          href: "smplx.js",
        }]
      } else if (running.update) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Resetting",
          href: "reset.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-play",
          text: "Start Web UI",
          href: "start.js",
        }, {
          icon: "fa-solid fa-person",
          text: "SMPL-X Setup",
          href: "smplx.js",
        }, {
          icon: "fa-solid fa-terminal",
          text: "CLI Run Form",
          href: "run.js",
        }, {
          icon: "fa-solid fa-arrows-rotate",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Reinstall",
          href: "install.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "Reset",
          href: "reset.js",
          confirm: "Are you sure you wish to reset 4DAnyone? This will delete the app and virtual environment.",
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
