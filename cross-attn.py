# 🚀 ViT Encoder + 1x1 Conv로 Spatial 정보 복원 검증
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

# ✨ ViT Encoder (Pretrained)
from torchvision.models.vision_transformer import vit_b_16

class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        self.hidden_dim = self.vit.hidden_dim

    def resize_positional_embedding(self, pos_embed, old_size, new_size):
        pos_embed = pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(new_size, new_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        return pos_embed

    def forward(self, x):
        x = self.vit.conv_proj(x)  # (B, C, H/16, W/16)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        pos_embed = self.vit.encoder.pos_embedding[:, 1:, :]
        N = x.shape[1]
        if N != pos_embed.shape[1]:
            pos_embed = self.resize_positional_embedding(pos_embed, old_size=14, new_size=int(N**0.5))

        x = x + pos_embed

        for block in self.vit.encoder.layers:
            x = block(x)

        return x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H/16, W/16)

# ✨ Simple 1x1 Conv Head (no complicated Decoder)
class SimpleSegmentationHead(nn.Module):
    def __init__(self, input_dim, num_classes=21):
        super().__init__()
        self.head = nn.Conv2d(input_dim, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)  # (B, 21, H/16, W/16)

# ✨ Full Model
class ViTSegmentationSimple(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.encoder = ViTEncoder(pretrained=True)
        self.head = SimpleSegmentationHead(input_dim=768, num_classes=num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.head(feats)
        return out

# ✨ Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(log_dir='./run/vit_seg_simple')

# ✨ Data Load (Pascal VOC 2012)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor()
])

trainset = torchvision.datasets.VOCSegmentation(
    root='./data', year='2012', image_set='train', download=True,
    transform=transform, target_transform=target_transform
)
valset = torchvision.datasets.VOCSegmentation(
    root='./data', year='2012', image_set='val', download=True,
    transform=transform, target_transform=target_transform
)

trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

# ✨ Model, Loss, Optimizer
model = ViTSegmentationSimple(num_classes=21).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=255)  # Pascal VOC ignore index

# ✨ Training
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for idx, (images, masks) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")):
        images = images.to(device)
        masks = masks.squeeze(1).to(device).long()

        outputs = model(images)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if idx % 50 == 0:
            print(f"Batch {idx} Loss: {loss.item():.4f}")
            writer.add_image("Train/Input", make_grid(images, normalize=True), global_step=epoch*len(trainloader)+idx)
            writer.add_image("Train/GT", make_grid(masks.unsqueeze(1).float()/21.0, normalize=True), global_step=epoch*len(trainloader)+idx)

    avg_loss = total_loss / len(trainloader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "./saved_model/vit_seg_simple.pth")
print("Training Completed!")

# ✨ Inference
model.eval()
os.makedirs("./saved_pred_simple", exist_ok=True)
with torch.no_grad():
    for idx, (images, masks) in enumerate(tqdm(valloader, desc="Testing")):
        images = images.to(device)
        masks = masks.squeeze(1).to(device)

        outputs = model(images)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        preds = outputs.argmax(dim=1)  # (B, H, W)

        writer.add_image("Test/Input", make_grid(images, normalize=True), global_step=idx)
        writer.add_image("Test/Prediction", make_grid(preds.unsqueeze(1).float()/21.0, normalize=True), global_step=idx)

        save_image(preds.unsqueeze(1).float()/21.0, f"./saved_pred_simple/pred_{idx:04d}.png")

        if idx >= 20:
            break

writer.close()
print("Inference Completed!")
