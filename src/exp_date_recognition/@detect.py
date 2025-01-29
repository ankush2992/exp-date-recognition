import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes=5):
    model = get_model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    return model, device

def detect_dates(model, image_path, confidence_threshold=0.5):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)
    device = next(model.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    return boxes, scores, labels, image

def visualize_detections(image_path, model, idx_to_class=None):
    if idx_to_class is None:
        idx_to_class = {
            1: 'date',
            2: 'due',
            3: 'code',
            4: 'prod'
        }
    boxes, scores, labels, image = detect_dates(model, image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rect)
        class_name = idx_to_class.get(label, f'class_{label}')
        ax.text(
            x1, y1-5, 
            f'{class_name}: {score:.2f}',
            color='white', 
            bbox=dict(facecolor='red', alpha=0.5)
        )
    plt.axis('off')
    plt.show()
    return boxes, scores, labels