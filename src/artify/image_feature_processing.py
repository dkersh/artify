from PIL import Image, ImageCms
import numpy as np
import rasterfairy
from sklearn.manifold import TSNE


def extract_colour_features(img: Image.Image) -> np.ndarray:
    """Resize img, convert from RGB to LAB representation and flatten.

    LAB representation is useful for clustering similar images together
    prior to using a dimension reduction algorithm such as TSNE so that is what
    we opt for here.

    Args:
        img (Image.Image): Input Pillow Image

    Returns:
        np.ndarray: Flatten, LAB representation of image.
    """
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(
        ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB"
    )

    small_img = img.resize((16, 16), Image.Resampling.BICUBIC)
    lab_img = ImageCms.applyTransform(small_img, rgb2lab_transform)

    return np.array(lab_img).astype(np.float32).flatten()


def sort_images_by_colour(images: list[Image.Image]) -> list[Image.Image]:
    """Run TSNE on image features, and project onto a 2D grid.

    Args:
        images (list[Image.Image]): List of images to be projected.

    Returns:
        list[Image.Image]: List of Pillow images, projected onto a 2D grid.
    """
    features = np.array([extract_colour_features(img) for img in images])

    embedding = TSNE(n_components=2).fit_transform(features)

    assignment = rasterfairy.transformPointCloud2D(embedding)
    idx = np.lexsort((assignment[0][:, 0], assignment[0][:, 1]))

    sorted_album_art = [images[i] for i in idx]

    return sorted_album_art
