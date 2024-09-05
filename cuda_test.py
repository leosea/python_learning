import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Print CUDA version
    print(f"CUDA Version: {torch.version.cuda}")
    # Print the number of CUDA devices
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    # Print the name of the current CUDA device
    print(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU.")

# Create a sample tensor and move it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 3).to(device)
print(f"Tensor x is on device: {x.device}")
