import torch
import torchvision.transforms as T
from PIL import Image
import pytesseract
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes=5):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def process_image(image_path, model, size=640):
    # Load and resize 
    image = Image.open(image_path)
    width, height = image.size
    
    if width > height:
        new_width = size
        new_height = int(height * (size / width))
    else:
        new_height = size
        new_width = int(width * (size / height))
    
    image = image.resize((new_width, new_height))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        image_tensor = transform(image).unsqueeze(0)
        predictions = model(image_tensor)
    
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5 and label == 1:  #date class
            x1, y1, x2, y2 = map(int, box)
            region = image.crop((x1, y1, x2, y2))
            text = pytesseract.image_to_string(
                region, 
                config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/.-'
            )
            print(f"Found date (confidence: {score:.2f}): {text.strip()}")
    
    return boxes, scores, labels

def main():
    try:
        print("Loading model...")
        model = get_model()
        
        checkpoint = torch.load('checkpoints/final_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict']) 
        model.eval()
        
        print("Model loaded successfully")
        
        # Process image
        print("\nProcessing image...")
        image_path = 'test_images/test_image.jpg'
        boxes, scores, labels = process_image(image_path, model)
        
        if len(boxes) == 0:
            print("No dates detected in the image")
            
    except Exception as e:
        print(f"\nAn error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

if __name__ == "__main__":
    main()