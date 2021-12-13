from PIL import Image, ImageChops, ImageOps
import numpy as np

def diff(image1, image2):
    return np.subtract(np.asarray(image1),np.asarray(image2))
    #return np.square(np.subtract(np.asarray(image1),np.asarray(image2))).mean()

def comp(base, b):
    a = ImageOps.grayscale(base)
    b = ImageOps.grayscale(b)
    arr1 = diff(a, b)
    arr = np.zeros(shape=(arr1.shape[0], arr1.shape[1], 4), dtype=np.uint8)
    arr[:, :, 0] = arr1
    arr[:, :, 3] = 128

    base = base.convert('RGBA')
    img = Image.fromarray(arr)
    base = Image.alpha_composite(base, img)



    return base

for compname in ['person', 'logo', 'mountain', 'storefront']:

    a = Image.open(f'{compname}_base.png')
    b = Image.open(f'{compname}_upscale.png')
    c = comp(a, b)
    c.save(f'{compname}_comp.png')