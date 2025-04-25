import cv2
import numpy as np


def normalize_int8(image: np.array):
    return cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ).astype(np.uint8)


def clahe(img, clip=1.5):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(img)
    return cl


def truncate_normalization(image_mask: tuple):
    """Normalize an image within a given ROI mask

    Args:
        source (list): list of tuples containing cropped images and roi masks

    Returns:
        np.array: normalized image
    """
    img, mask = image_mask
    Pmin = np.percentile(img[mask != 0], 2)
    Pmax = np.percentile(img[mask != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    if Pmax != Pmin:
        normalized = (truncated - Pmin) / (Pmax - Pmin)
    else:
        normalized = np.zeros_like(truncated)
    normalized[mask == 0] = 0
    return normalized


def crop_to_roi(image: np.array):
    """Crop mammogram to breast region.

    Args:
        img_list (list): List of original image as uint8 np.arrays

    Returns:
        tuple (list, list): (cropped_images, rois)
    """
    original_image = image.copy()
    image = clahe(original_image, 1.0)

    _, breast_mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        breast_mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (original_image[y: y + h, x: x + w], breast_mask[y: y + h, x: x + w])
