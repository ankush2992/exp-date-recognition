import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .dataset import ProductDataset
from .transformations import get_train_transform, get_test_transform

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(data_dir, num_epochs=10, batch_size=8):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = ProductDataset(
        root_dir=data_dir,
        split='train',
        transform=get_train_transform()
    )
    val_dataset = ProductDataset(
        root_dir=data_dir,
        split='evaluation',
        transform=get_test_transform()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    num_classes = 5
    model = get_model(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training on device: {device}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Batch [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}')
        checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        lr_scheduler.step()
    return model

if __name__ == '__main__':
    data_dir = 'Product-Real'
    model = train_model(data_dir) 