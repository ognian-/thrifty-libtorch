import torch
import sys

if len(sys.argv) < 2:
   sys.exit("Pass model name (without .pth extension) as argument")

model_name = sys.argv[1]
path = f"{model_name}.pth" 

data = torch.load(path, map_location='cpu')
print(f"Type: {type(data)}")
for key, value in data.items():
    print(f"{key}: {value.shape}")

torch.save(
    dict(data),
    f"{model_name}-libtorch.pth",
    _use_new_zipfile_serialization=True
)

print(f"Saved as: {model_name}-libtorch.pth")
