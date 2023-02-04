import cv2


def scale_image_by_height(image, height):
    ratio = image.shape[1] / image.shape[0]
    width = round(height * ratio)
    return cv2.resize(image, (width, height))
