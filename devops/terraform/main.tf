terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ─── GPU Compute Instance for Inference ───
resource "google_compute_instance" "fdanyone_inference" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "projects/ml-images/global/images/family/common-cu128-debian-11-py311"
      size  = var.boot_disk_size_gb
      type  = "pd-ssd"
    }
  }

  # Attach GPU
  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  # GPU instances require on_host_maintenance = TERMINATE
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible         = var.preemptible
  }

  network_interface {
    network    = "default"
    subnetwork = "default"

    # Ephemeral external IP for setup; remove for production
    access_config {}
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -euo pipefail

    # Install Docker + NVIDIA Container Toolkit
    curl -fsSL https://get.docker.com | sh
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker

    # Clone and build
    cd /opt
    git clone https://github.com/theCosmicCrafter/4DAnyone.git
    cd 4DAnyone
    docker compose -f devops/docker/docker-compose.yml build
  EOF

  tags = ["fdanyone", "gpu-inference"]

  labels = {
    app         = "fdanyone"
    environment = var.environment
  }
}

# ─── Firewall rule (SSH only by default) ───
resource "google_compute_firewall" "fdanyone_ssh" {
  name    = "${var.instance_name}-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.ssh_source_ranges
  target_tags   = ["fdanyone"]
}

# ─── Outputs ───
output "instance_ip" {
  value       = google_compute_instance.fdanyone_inference.network_interface[0].access_config[0].nat_ip
  description = "External IP of the inference instance"
}

output "instance_name" {
  value       = google_compute_instance.fdanyone_inference.name
  description = "Name of the compute instance"
}

output "ssh_command" {
  value       = "gcloud compute ssh ${google_compute_instance.fdanyone_inference.name} --zone=${var.zone}"
  description = "SSH command to connect to the instance"
}
