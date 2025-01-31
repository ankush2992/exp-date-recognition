import pytest
import torch
from PIL import Image
from src.exp_date_recognition.detection.detect import detect
from src.exp_date_recognition.detection.model.resnet50 import get_model_instance

@pytest.fixture
def example_image():
    # Load an example image for testing
    image = Image.open("tests/assets/2boxes.jpg").convert("RGB")
    return image


@pytest.fixture
def example_categories():
    # Create a dummy categories object for testing
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
    
    # Test the detect function
    detected_image, boxes, labels = detect(model, example_image, example_categories)

    # Assert that detected_image is a PIL.Image object
    assert isinstance(detected_image, torch.Tensor)
    # Assert that boxes is a list of torch.Tensor objects
    assert isinstance(boxes, torch.Tensor)
    # Assert that labels is a list of strings
    assert isinstance(labels, list)
    assert all(isinstance(label, str) for label in labels)


if __name__ == '__main__':
    assert True
