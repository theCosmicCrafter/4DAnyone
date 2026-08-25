variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone (must have GPU availability)"
  type        = string
  default     = "us-central1-a"
}

variable "instance_name" {
  description = "Name of the compute instance"
  type        = string
  default     = "fdanyone-inference"
}

variable "machine_type" {
  description = "GCE machine type (must support GPU attachment)"
  type        = string
  default     = "n1-standard-16"  # 16 vCPU, 60 GB RAM
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-l4"  # 24 GB VRAM, cost-effective
  # Alternatives:
  #   "nvidia-tesla-a100"  — 40/80 GB, highest throughput
  #   "nvidia-l4"          — 24 GB, good price/performance
  #   "nvidia-tesla-t4"    — 16 GB, budget option (may be tight on VRAM)
}

variable "gpu_count" {
  description = "Number of GPUs to attach"
  type        = number
  default     = 1
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB (needs space for models ~25 GB + data)"
  type        = number
  default     = 200
}

variable "preemptible" {
  description = "Use preemptible/spot instance for cost savings (may be interrupted)"
  type        = bool
  default     = true
}

variable "environment" {
  description = "Environment label (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "ssh_source_ranges" {
  description = "CIDR ranges allowed SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}
