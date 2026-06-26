import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os

from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets

def save_pil_image(data, dir, integers=False):
    data = (data * 0.5 + 0.5) * 255
    data = data.detach().cpu()
    data = torch.unbind(data, dim=0)
    data = make_grid(list(data), padding=2, nrow=5, pad_value=100)
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

def get_tiny_train(args):
    assert args.img_size == 64, "Tiny ImageNet only supports image size of 64x64"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root ='../tiny-imagenet-200/train', transform = transform)

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)

    return loader, len(dataset)

def get_cifar10_data(args):
    assert args.img_size == 32, "CIFAR10 only supports image size of 32x32"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Download dataset
    cifar10_dataset = torchvision.datasets.CIFAR10(root ='../cifar10', train=True, transform = transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        cifar10_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    return train_loader, len(cifar10_dataset)

def create_sample_cifar_valid(args):
    # Download dataset
    transform = transforms.Compose([
        transforms.ToTensor()])
    cifar10_dataset = torchvision.datasets.CIFAR10(root ='../cifar10', train=True, transform=transform, download=True)

    valid_loader = torch.utils.data.DataLoader(
        cifar10_dataset, 
        batch_size=100, 
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)

    dict_labels ={}
    for i, (images, labels) in enumerate(valid_loader):
        for j, label in enumerate(labels):
            label = label.item()          
            if label not in dict_labels:
                dict_labels[label] = []
            if len(dict_labels[label]) >= args.num_samples // args.num_classes:
                continue
            dict_labels[label].append(images[j])
    
    if not os.path.exists("cifar10_valid_4metrics"):
        os.makedirs("cifar10_valid_4metrics")

    sort_pairs_by_labes = sorted(dict_labels.items(), key=lambda x: x[0])
    for i, (label, images) in enumerate(sort_pairs_by_labes):
        for j, image in enumerate(images):
            save_pil_image(image.unsqueeze(0)*2.-1., f"cifar10_valid_4metrics/img_{i*(args.num_samples // args.num_classes)+j+1}_class_{label}.jpg")

def create_sample_celeba_valid(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.ToTensor()])
    celeb_dataset = torchvision.datasets.ImageFolder(root ='../CELEBA', transform = transform)
    indices = list(range(0, len(celeb_dataset)))
    valid_idx = indices[:162770]
    valid_set = torch.utils.data.Subset(celeb_dataset, valid_idx)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)

    if not os.path.exists("celeba_valid_4metrics"):
        os.makedirs("celeba_valid_4metrics")
    
    for i, (images, labels) in enumerate(valid_loader):
        if i > args.num_samples // args.batch_size:
            break
        for j, image in enumerate(images):
            save_pil_image(image.unsqueeze(0)*2.-1., f"celeba_valid_4metrics/img_{i*100+j+1}.jpg")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for data loading")
    parser.add_argument("--num_samples", type=int, default=50000, help="Total number of samples to create for metrics evaluation")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes in the dataset")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for data loading")
    args = parser.parse_args()

    #create_sample_cifar_valid(args)
    create_sample_celeba_valid(args)