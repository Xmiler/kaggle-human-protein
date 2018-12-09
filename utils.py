from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset


def open_image(image_name, colors):
    image_name = str(image_name)
    channels = [Image.open(image_name+'_'+color+'.png') for color in colors]
    img = np.stack([np.asanyarray(c) for c in channels], axis=-1)
    img = Image.fromarray(img)
    return img


class ProteinDataset(Dataset):
    def __init__(self, csv_file, images_dir,
                 colors=['red', 'green', 'blue'], 
                 idxs=None, transforms=None):
        csv_content = pd.read_csv(csv_file)
        self._filenames = np.array(csv_content['Id'].tolist())
        self._labels = MultiLabelBinarizer().fit_transform([tuple(int(i) for i in item.split(' ')) 
                                                            for item in  csv_content['Target'].tolist()])
        assert len(self._filenames) == len(self._labels)
        if idxs is not None:
            self._filenames = self._filenames[idxs]
            self._labels = self._labels[idxs]
        
        self._images_dir = images_dir
        self._colors = colors        
        self._transforms = transforms
        
    def __len__(self):
        return len(self._filenames)
    
    def __getitem__(self, idx):
        image = open_image(self._images_dir / self._filenames[idx], 
                           self._colors)
        if self._transforms is not None:
            image = self._transforms(image)
        
        labels = self._labels[idx]
        
        sample = (image, labels)
        
        return sample