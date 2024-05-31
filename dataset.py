
import numpy as np
import os
from glob import glob
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

class CustomDataset(Dataset):
    def __init__(self, img_dir, height, width, transform=None):
        self.img_dir = img_dir
        self.img_paths = glob.glob(os.path.join(img_dir, '*/*_image.jpg'))
        self.transform = transform
        self.HEIGHT = height
        self.WIDTH = width

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')

        # Create the corresponding mask path
        mask_path = img_path.replace('leftImg8bit', 'gtFine').replace(
            '_image.jpg', '_label.png')
        # Load mask
        mask = Image.open(mask_path).convert('L')
        # Convert the mask to a numpy array and process it
        mask = np.array(mask)
        mask[mask == 255] = 7
        mask = torch.tensor(mask, dtype=torch.long)

        # Resize the mask
        # Add batch and channel dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask.float(), size=(self.HEIGHT, self.WIDTH), mode='nearest')
        # Remove batch and channel dimensions and convert back to long
        mask = mask.squeeze(0).squeeze(0).long()

        if self.transform:
            image = self.transform(image)

        return image, mask


def Create_DataLoader(batch_size, height, width, img_train, img_val):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width))
    ])
    train_dataset = CustomDataset(
        img_train, height, width, transform=train_transforms)
    val_dataset = CustomDataset(img_val, height,width,transform=train_transforms)

    # creating Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
