from pathlib import Path


ROOT_DIR = Path(__file__).parent

CITYSCAPES_IMAGE_DIR = (
    ROOT_DIR
    / "data"
    / "cityscapes"
    / "images"
)

CITYSCAPES_LABEL_DIR = (
    ROOT_DIR
    / "data"
    / "cityscapes"
    / "labels"
)

NUM_CLASSES = 19

IMAGE_SIZE = (512, 1024)