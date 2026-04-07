import torch
import time
from neuroswift.model import NeuroSwiftConfig, NeuroSwiftLM

# Target: 100+ steps/s cluster-wide. 
# With 8 workers, each worker must do 12.5 steps/s (80ms/step).
# But realistically, 100 steps/s per core should be possible for 2M params.

config = NeuroSwiftConfig(
    vocab_size=4441,
    d_model=176,
    n_layers=1,
    expert_hidden=352,
    num_experts=8,
    top_k=2,
    ternary_mode=True
)
model = NeuroSwiftLM(config).cpu()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

x = torch.randint(0, 4441, (2, 128))
y = torch.randint(0, 4441, (2, 128))

# Warmup
for _ in range(5):
    out = model(x, targets=y)
    out["loss"].backward()
    model.zero_grad()

print("Profiling 100 steps...")
start = time.time()
for _ in range(100):
    out = model(x, targets=y)
    out["loss"].backward()
    # optimizer.step() substitute
    for p in model.parameters():
        if p.grad is not None:
            p.data.add_(p.grad, alpha=-0.001)
    model.zero_grad()

elapsed = time.time() - start
print(f"Elapsed: {elapsed:.4f}s")
print(f"Single worker speed: {100/elapsed:.2f} steps/s")
print(f"Estimated 8-worker speed: {800/elapsed:.2f} steps/s")
