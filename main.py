import torch

from models.unet.model import UNet
from models.unet.inference import predict
from evaluation.segmentation_metrics import (
    pixel_accuracy,
    mean_iou
)
from visualization.segmentation import (
    save_segmentation
)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Create model
    model = UNet(num_classes=19)

    # Test image
    image_path = (
        "data/cityscapes/images/"
        "test_image.png"
    )

    # Prediction
    prediction = predict(
        model,
        image_path,
        device
    )

    print(
        "Prediction shape:",
        prediction.shape
    )

    # Save result
    save_segmentation(
        prediction,
        "outputs/segmentation/result.png"
    )

    print("Segmentation completed!")


if __name__ == "__main__":
    main()