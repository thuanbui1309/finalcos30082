"""Image transforms for training, validation, and inference."""

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EDGEFACE_MEAN = [0.5, 0.5, 0.5]
EDGEFACE_STD = [0.5, 0.5, 0.5]


def get_train_transforms(img_size: int = 112) -> transforms.Compose:
    """Return augmentation pipeline for training.

    Includes random horizontal flip, slight rotation, color jitter,
    resize, tensor conversion, and ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 112) -> transforms.Compose:
    """Return deterministic pipeline for validation / testing."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform(img_size: int = 112) -> transforms.Compose:
    """Return transform for a single PIL Image at inference time.

    Identical to validation transforms; kept as a separate entry point
    for clarity in inference scripts.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_edgeface_transform(img_size: int = 112) -> transforms.Compose:
    """Return inference transform for EdgeFace models.

    EdgeFace expects pixel values normalised to [-1, 1] via mean/std of 0.5,
    rather than ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EDGEFACE_MEAN, std=EDGEFACE_STD),
    ])
