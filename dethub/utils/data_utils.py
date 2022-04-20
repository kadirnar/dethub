def read_image(image_path: str):
    """
    Loads image as numpy array from given path.
    """
    import cv2
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image
    return image