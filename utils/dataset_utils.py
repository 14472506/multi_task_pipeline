"""
Detials
"""
# imports
from PIL import Image

# functions
# ----- centre crop
def basic_square_crop(img):
    """
    Detials
    """
    # getting width and height
    width, height = img.size

    # getting centre width and height
    centre_width = width/2
    centre_height = height/2

    # max square size 
    max_size = min(width, height)
    half_max = max_size/2

    # getting cropping window
    left = centre_width - half_max
    right = centre_width + half_max
    top = centre_height - half_max
    bottom = centre_height + half_max

    # cropping
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img

def resize(img, size=1000):
    """
    Detials
    """
    # resizing image
    resized_img = img.resize((size, size))

    # returing resized image
    return(resized_img)