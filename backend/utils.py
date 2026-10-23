"""Image validation and per-class splits. No image files are saved."""
import io
import math
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 512 * 1024

def preprocess_image_bytes(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('Choose an image smaller than 512 KB.')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError('Image exceeds the 16 megapixel limit.')
            image = ImageOps.exif_transpose(image).convert('RGB').resize((64, 64))
            return np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError('The file is not a supported, readable image.') from None

def load_dataset(dataset):
    if len(dataset) < 2:
        raise ValueError('Add at least two classes, with at least three images each.')
    rng = np.random.default_rng(42)
    train_x, val_x, train_y, val_y = [], [], [], []
    labels = dict(enumerate(sorted(dataset)))
    # Every class is represented in training and validation. Validation images
    # are never used for fitting; tiny datasets still give unreliable estimates.
    for index, name in labels.items():
        images = dataset[name]
        if len(images) < 3:
            raise ValueError(f'Class "{name}" needs at least three images.')
        order = rng.permutation(len(images))
        count = max(1, math.ceil(len(images) * .2))
        val_x.extend(images[i] for i in order[:count])
        train_x.extend(images[i] for i in order[count:])
        val_y.extend([index] * count)
        train_y.extend([index] * (len(images) - count))
    return np.asarray(train_x), np.asarray(val_x), np.asarray(train_y), np.asarray(val_y), labels
