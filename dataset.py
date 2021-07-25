import pandas as pd
import os
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

class FashionDataset(Dataset):
    def __init__(self, args, transform=None, phase='train'):
        super().__init__()
        self.exp_random_seed = args.seed
        self.args = args
        self.phase = phase
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        np.random.seed(self.exp_random_seed)

        assert phase in ['train', 'val', 'test']
        self.dataframe = pd.read_csv(os.path.join(args.data_dir, 'attributes_clean.csv'))

        ## Replacing N/A with additional class.
        self.dataframe['neck'] = self.dataframe['neck'].fillna(7.0)
        self.dataframe['sleeve_length'] = self.dataframe['sleeve_length'].fillna(4.0)
        self.dataframe['pattern'] = self.dataframe['pattern'].fillna(10.0)

        total_samples = len(self.dataframe)
        idx_perm = np.arange(total_samples)
        np.random.shuffle(idx_perm)
        # print(idx_perm)
        self.train_idxs = idx_perm[:int(total_samples*args.train_ratio)]
        self.val_idxs = idx_perm[int(total_samples*args.train_ratio):int(total_samples*(args.train_ratio+args.val_ratio))]
        self.test_idxs = idx_perm[int(total_samples*(args.train_ratio+args.val_ratio)):]

    def __getitem__(self, index):
        if(self.phase == 'train'):
            df_idx = self.train_idxs[index]
        elif(self.phase == 'val'):
            df_idx = self.val_idxs[index]
        elif(self.phase == 'test'):
            df_idx = self.test_idxs[index]

        image_path = os.path.join(self.args.data_dir, 'images', self.dataframe.filename[df_idx])
        pattern_label = int(self.dataframe.pattern[df_idx])
        neck_type_label = int(self.dataframe.neck[df_idx])
        sleeve_length_label = int(self.dataframe.sleeve_length[df_idx])

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        return image, pattern_label, sleeve_length_label, neck_type_label

    def __len__(self):
        if(self.phase == 'train'):
            return len(self.train_idxs)
        elif(self.phase == 'val'):
            return len(self.val_idxs)
        elif(self.phase == 'test'):
            return len(self.test_idxs)
