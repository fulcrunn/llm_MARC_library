# Execute this code to check if GPU is available and to get the name of the GPU device.

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Returns:
# True
# NVIDIA RTX 3090 (or the name of your GPU device)