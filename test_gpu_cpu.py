import torch
import subprocess


# TODO: run command: F5
def check_gpu_cuda():
    print("=== Simple GPU/CUDA Availability Check ===")

    # PyTorch CUDA check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print("PyTorch version:", torch.__version__)
        print(f"GPU detected: {device_name}")

        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")

        device_capability = torch.cuda.get_device_capability(0)
        print(
            f"GPU: {device_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})"
        )

    # Simple nvidia-smi check
    print("PyTorch CUDA available:", cuda_available)
    try:
        subprocess.run(
            "nvidia-smi", check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("nvidia-smi: GPU is accessible")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("nvidia-smi: Not available")

    return cuda_available


if __name__ == "__main__":
    cuda_available = check_gpu_cuda()

    if not cuda_available:
        print("\nNo GPU was detected by PyTorch.")
        print("Possible reasons:")
        print("1. Your system does not have a CUDA-capable GPU")
        print("2. CUDA/cuDNN is not installed correctly")
        print("3. You're using CPU-only versions of the libraries")
        print("4. GPU drivers are not installed or outdated")
