
import torch
from gsplat import rasterization

# 1. Check CUDA
device = "cuda"
print("GPU:", torch.cuda.get_device_name(0))

# 2. Create 10 Gaussian points
num_gaussians = 10

means = torch.zeros((num_gaussians, 3), device=device)
means[:, 2] = 3.0
means[:, 0] = torch.linspace(-0.5, 0.5, num_gaussians, device=device)

# Identity rotations (quaternion format: w, x, y, z)
quats = torch.zeros((num_gaussians, 4), device=device)
quats[:, 0] = 1.0

# Gaussian scales
scales = torch.full((num_gaussians, 3), 0.15, device=device)

# Opacity
opacities = torch.full((num_gaussians,), 0.8, device=device)

# RGB colors
colors = torch.rand((num_gaussians, 3), device=device)

# Enable gradients for training test
means.requires_grad_(True)

# 3. Create a camera
viewmats = torch.eye(4, device=device).unsqueeze(0)

Ks = torch.tensor(
    [[100.0, 0.0, 64.0],
     [0.0, 100.0, 64.0],
     [0.0, 0.0, 1.0]],
    device=device
).unsqueeze(0)

# 4. Rasterize the Gaussians
render_colors, render_alphas, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=128,
    height=128
)

# 5. Synchronize CUDA and inspect output
torch.cuda.synchronize()

print("\nRasterization successful!")
print("Rendered image shape:", render_colors.shape)
print("Alpha shape:", render_alphas.shape)
print("Rendered image min:", render_colors.min().item())
print("Rendered image max:", render_colors.max().item())

# 6. Test backward pass
loss = render_colors.mean()
loss.backward()

torch.cuda.synchronize()

print("\nBackward pass successful!")
print("Gradient shape:", means.grad.shape)
print("Gradient exists:", means.grad is not None)

print("\nGSPLAT TEST PASSED!")
