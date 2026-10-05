"""Report whether PyTorch can use CUDA and inspect the active GPU."""

import sys

import torch


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No se detectó CUDA. Revisá PyTorch y el driver NVIDIA.")
        return 1

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    print(f"Dispositivo CUDA {device_index}: {torch.cuda.get_device_name(device_index)}")
    print(f"Memoria total: {properties.total_memory / (1024**3):.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())