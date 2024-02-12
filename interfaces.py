from pytorch_lightning import LightningDataModule
import torch
import os
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from utils import calculate_mean_std, custom_transform

class LPRCharacter(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 transforms=None,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        
        classes = sorted(os.listdir(data_dir + '/' + split))
        if '.DS_Store' in classes:
            classes.remove('.DS_Store')
        # Labels each class
        self.classes_label = {classes[i]: i for i in range(len(classes))}
        
        self.all_classes_dir = os.listdir(os.path.join(self.data_dir, split))
        self.all_imgs = []
        for class_dir in self.all_classes_dir:
            if class_dir == ".DS_Store":
                continue
            self.all_imgs += os.listdir(os.path.join(self.data_dir, split, class_dir))
            if '.ipynb_checkpoints' in self.all_imgs:
                self.all_imgs.remove('.ipynb_checkpoints')
            
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index):
        # Get image path at current index
        _class = self.all_imgs[index].split("_")[1][0]
        img_path = os.path.join(self.data_dir, self.split, _class, self.all_imgs[index])

        img = Image.open(img_path).convert("RGB")
        
        # Transform image. Should be  tensorization, resizing, normalization
        if self.transforms:
            img = self.transforms(img)
            
        label = torch.tensor(self.classes_label[_class])
        return img, label

class LPRCharacterDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 mean: list,
                 std: list,
                 batch_size: int = 32,
                 numworkers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.numworkers = numworkers
        if not mean or not std:
            mean, std = calculate_mean_std(self.data_dir)
        self._train_transforms = custom_transform(mean, std, 'train')
        self._val_transforms = custom_transform(mean, std, 'val')
        self._test_transforms = custom_transform(mean, std, 'test')

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = LPRCharacter(self.data_dir,
                                                        split="train",
                                                        transforms=self._train_transforms)
            self.val_dataset = LPRCharacter(self.data_dir,
                                                      split="val",
                                                      transforms=self._val_transforms)
        if stage == "test" or stage is None:
            self.test_dataset = LPRCharacter(self.data_dir,
                                                       split="test",
                                                       transforms=self._test_transforms)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.numworkers,
                          persistent_workers=True
                          )
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.numworkers,
                          persistent_workers=True
                          )
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.numworkers,
                          persistent_workers=True
                          )    