module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    when: "{{exists('app')}}",
    method: "shell.run",
    params: {
      path: "app",
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "{{exists('app') ? 'app' : '.'}}",
      message: "git submodule update --init third_party/GVHMR"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "{{exists('app') ? 'app' : '.'}}",
      message: [
        "uv pip install -r requirements.txt"
      ]
    }
  }]
}
