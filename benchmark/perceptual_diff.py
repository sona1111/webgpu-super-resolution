from PIL import Image
import imagehash
import numpy as np

def mse(image1, image2):

    return np.square(np.subtract(np.asarray(image1),np.asarray(image2))).mean()

for compname in ['baboon', 'sandcats', 'fox']:

    image_one = 'baboon_base.png'

    base = Image.open(f'{compname}_base.png').convert('RGB')
    esrgan = Image.open(f'{compname}_upscale_esrgan.png').convert('RGB')
    realplus = Image.open(f'{compname}_upscale_realplus.png').convert('RGB')
    realmse = Image.open(f'{compname}_upscale_realmse.png').convert('RGB')
    bicubic = Image.open(f'{compname}_bicubic.png').convert('RGB')

    print(f'ESRGAN error ({compname}): {mse(base,esrgan)}')
    print(f'realplus error ({compname}): {mse(base,realplus)}')
    print(f'realmse error ({compname}): {mse(base,realmse)}')
    print(f'Bicubic error ({compname}): {mse(base,bicubic)}')


