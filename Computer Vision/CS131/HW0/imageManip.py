import numpy as np
from PIL import Image
from skimage import color

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    # YOUR CODE HERE
    out = Image.open(image_path)
    # END YOUR CODE

    return out

def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    # YOUR CODE HERE
    x = np.array(image)
    out = x * x * 0.5
    #out = Image.fromarray(np.uint8(out))
    # END YOUR CODE

    return out

def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    # YOUR CODE HERE
    x = np.array(image)
    h = np.shape(x)[0]
    w = np.shape(x)[1]
    for i in range(h):
        for j in range(w):
            for k in range(3):
                x[i, j, k] = (x[i, j, 0] * 299 + x[i, j, 1] *
                              587 + x[i, j, 2] * 114 + 500) / 1000

    out = Image.fromarray(np.uint8(x))
    # END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    # YOUR CODE HERE
    dic = {'R': 0, 'G': 1, 'B': 2}
    x = np.array(image)
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            x[i, j, dic[channel]] = 0
    out = Image.fromarray(np.uint8(x))
    # END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)

    # YOUR CODE HERE
    out = np.zeros(np.shape(lab))
    dic = {"L": 0, "A": 1, "B": 2}
    for i in range(np.shape(lab)[0]):
        for j in range(np.shape(lab)[1]):
            out[i, j, dic[channel]] = lab[i, j, dic[channel]]
    out = Image.fromarray(np.uint8(out))
    # END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    dic = {"H": 0, "S": 1, "V": 2}
    # YOUR CODE HERE
    out = np.zeros(np.shape(hsv))
    for i in range(np.shape(hsv)[0]):
        for j in range(np.shape(hsv)[1]):
            out[i, j, dic[channel]] = hsv[i, j, dic[channel]]
    out = Image.fromarray(np.uint8(color.hsv2rgb(out)*256))
    # END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    # YOUR CODE HERE
    dic = {'R': 0, 'G': 1, 'B': 2}
    img1 = np.array(image1)
    img2 = np.array(image2)
    out = np.zeros(np.shape(img1))

    for i in range(np.shape(img1)[0]):
        for j in range(np.shape(img1)[1]):
            out[i, j, dic[channel1]] = img1[i, j, dic[channel1]]
            out[i, j, dic[channel2]] = img2[i, j, dic[channel2]]

    out = Image.fromarray(np.uint8(out))

    # END YOUR CODE

    return out
