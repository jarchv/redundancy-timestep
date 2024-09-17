import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets

def save_pil_image(data, dir, integers=False):
    data = (data * 0.5 + 0.5) * 255
    data = data.detach().cpu()
    data = torch.unbind(data, dim=0)
    data = make_grid(list(data), padding=2, nrow=4, pad_value=100)
    data = data.to(torch.uint8)
    data = F.to_pil_image(data)
    data.save(dir)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class CustomCOCODataset(datasets.CocoCaptions):
    def __init__(self, root, annFile, transform=None):
        super(CustomCOCODataset, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, index):
        img, caption = super(CustomCOCODataset, self).__getitem__(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, caption[:5]


def get_coco_train_loader(args):
    """
    Get the COCO train loader with the specified image size from the arguments.
    """
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], inplace=True)])
    #transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), inplace=True)])
    dataset = CustomCOCODataset(
        root = args.imgs_path,
        annFile = args.caps_path,
        transform=transform)
    
    loader  = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last=True)

    return loader, len(dataset)

def get_celeb_data(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    celeb_dataset = torchvision.datasets.ImageFolder(root ='../CELEBA', transform = transform)

    indices = list(range(0, len(celeb_dataset)))
    train_idx = indices[:162770]
    train_set = torch.utils.data.Subset(celeb_dataset, train_idx)
    
    train_size = len(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    return train_loader, train_size

