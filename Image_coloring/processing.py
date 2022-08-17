import torch
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def predict(model, image):
    img = image.convert("RGB")
    img = img.reshape(256, 256, Image.BICUBIC)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1

    L = L[None, :]

    output = model.net_G(L)

    output = output.detach()
    oo = lab_to_rgb(L, output)
    return oo[0]
