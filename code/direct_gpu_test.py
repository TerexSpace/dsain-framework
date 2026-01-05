#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision
import time

print("="*70)
print("DIRECT GPU TRAINING TEST")
print("="*70)

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Simple ResNet18
model = torchvision.models.resnet18(num_classes=10)
model = model.to(device)
print(f"\nModel on device: {next(model.parameters()).device}")

# Dummy data on GPU
batch_size = 128  # Larger batch
data = torch.randn(batch_size, 3, 32, 32, device=device)
labels = torch.randint(0, 10, (batch_size,), device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
model.train()
print(f"\nRunning 100 training steps with batch_size={batch_size}...")
print("Watch GPU util in nvidia-smi - should spike to 70-100%\n")

start = time.time()
for i in range(100):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        print(f"Step {i}: loss = {loss.item():.4f}")

elapsed = time.time() - start
print(f"\nCompleted 100 steps in {elapsed:.2f}s")
print(f"Throughput: {100/elapsed:.2f} steps/sec")
print(f"Samples/sec: {batch_size*100/elapsed:.0f}")

print("\nIf GPU util was 70-100% during this test, GPU works fine.")
print("If GPU util stayed <20%, something is wrong with GPU training.")
print("="*70)
