import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Self-Attention Layer
class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)  # (B, heads, N, dim_head)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v)  # (B, heads, N, dim_head)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)

        return out

# Classifier Model
class AttentionClassifier(nn.Module):
    def __init__(self, num_classes=10, emb_dim=128):
        super().__init__()
        self.patch_embed = nn.Linear(3 * 32 * 32, emb_dim)  # Entire image as one patch
        self.self_attn = SimpleSelfAttention(dim=emb_dim, num_heads=4)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)  # Flatten (B, 3*32*32)
        x = self.patch_embed(x).unsqueeze(1)  # (B, 1, emb_dim)
        x = self.self_attn(x)  # (B, 1, emb_dim)
        x = x.mean(dim=1)      # Global average over sequence (only 1 token)
        x = self.mlp(x)
        return x

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CIFAR10 DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model, Loss, Optimizer
model = AttentionClassifier(num_classes=10, emb_dim=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Train Loss: {total_loss/len(trainloader):.4f}, Train Acc: {acc:.2f}%")

# Test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100. * correct / total:.2f}%")
