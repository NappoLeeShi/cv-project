import numpy as np
from PIL import Image


def save_segmentation(
    prediction,
    output_path
):

    prediction = prediction.numpy()

    # Chuyển class ID thành grayscale
    image = Image.fromarray(
        np.uint8(prediction)
    )

    image.save(output_path)