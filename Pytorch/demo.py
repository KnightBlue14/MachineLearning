import torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.mean()

# Compute gradients
z.backward()
print(x.grad)