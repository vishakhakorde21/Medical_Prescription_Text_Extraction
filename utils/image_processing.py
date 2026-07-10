import cv2
import numpy as np


def load_image(image_path):
    """
    Reads an image from disk.
    """
    image = cv2.imread(image_path)
    return image


def resize_image(image, width=800):
    """
    Resize image while maintaining aspect ratio.
    """

    original_height, original_width = image.shape[:2]

    ratio = width / original_width
    new_height = int(original_height * ratio)

    resized = cv2.resize(image, (width, new_height))

    return resized


def convert_to_grayscale(image):
    """
    Converts a color image to grayscale.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def blur_image(image):
    """
    Applies Gaussian Blur.
    """

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    return blurred


def adaptive_threshold(image):
    """
    Applies Adaptive Gaussian Thresholding.
    """

    threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return threshold


def remove_noise(image):
    """
    Removes small noise from image.
    """

    denoised = cv2.fastNlMeansDenoising(
        image,
        None,
        10,
        7,
        21
    )

    return denoised


def morphological_processing(image):
    """
    Performs morphological opening followed by closing.

    Opening:
        Removes tiny noise.

    Closing:
        Connects broken handwriting.
    """

    kernel = np.ones((2, 2), np.uint8)

    opening = cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel
    )

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closing