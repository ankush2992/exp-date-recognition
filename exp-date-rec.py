import pytest
import torch
from PIL import Image
from src.exp_date_recognition.detection.detect import detect
from src.exp_date_recognition.detection.model.resnet50 import get_model_instance
@pytest.fixture
def example_image():
    image = Image.open("tests/assets/2boxes.jpg").convert("RGB")
    return image
@pytest.fixture
def example_categories():

    class Categories:
        int2str_dict = {
            0: "prod",
            1: "date",
            2: "due", 
            3: "code"
        }
        @staticmethod
        def int2str(label):
            return Categories.int2str_dict.get(label)
    return Categories()
def test_detect(example_image, example_categories):
    model = get_model_instance(4, load_fine_tunned="v3")
    detected_image, boxes, labels = detect(model, example_image, example_categories)
    assert isinstance(detected_image, torch.Tensor)
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(labels, list)
    assert all(isinstance(label, str) for label in labels)
if __name__ == '__main__':
    assert True
