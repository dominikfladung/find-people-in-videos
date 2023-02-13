import cv2


def scale_image_by_height(image, height):
    """
    It takes an image and a height, and returns a scaled version of the image with the given height
    
    :param image: The cv2 image to be scaled
    :param height: The height of the image to be returned
    :return: The image is being returned.
    """
    ratio = image.shape[1] / image.shape[0]
    width = round(height * ratio)
    return cv2.resize(image, (width, height))
