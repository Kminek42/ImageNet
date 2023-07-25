import torch
import torchvision

A = torch.rand((1, 3, 3, 5))
print("A\n", A)

A = torchvision.transforms.Normalize(A.mean(), A.std())(A)

print("\n\nNorm(A)\n", A)
