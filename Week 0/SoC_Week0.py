import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
X = torch.linspace(-3, 3, 100).reshape(-1, 1)
y = 2 * X + 1 + torch.randn(X.size()) * 0.5   # y = 2x + 1 + noise

# ---------- Model ----------
model = nn.Linear(1, 1)        # one input, one output
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ---------- Training ----------
losses = []
for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# ---------- Visualize ----------
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()

print(f"Learned weight: {model.weight.item():.3f} (true: 2.0)")
print(f"Learned bias:   {model.bias.item():.3f} (true: 1.0)")