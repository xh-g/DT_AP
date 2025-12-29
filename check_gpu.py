import torch
import sys

def check_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            
        # Test a small tensor operation on GPU
        try:
            device = torch.device("cuda")
            x = torch.tensor([1.0, 2.0, 3.0]).to(device)
            y = torch.tensor([4.0, 5.0, 6.0]).to(device)
            z = x + y
            print(f"Test tensor operation on GPU successful: {z}")
        except Exception as e:
            print(f"Error running tensor operation on GPU: {e}")
    else:
        print("GPU training is NOT available. Using CPU.")

if __name__ == "__main__":
    check_gpu()
