import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, metadata_file, transform=None, score_range=(0, 1)):
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.transform = transform
        self.score_range = score_range
        self.data = pd.read_csv(metadata_file, sep=',')
        self.image_paths = self.data['dis_img_path'].values
        self.quality_scores = self.normalize_scores(self.data['score'].values)

    def normalize_scores(self, scores):
        min_score, max_score = self.score_range
        return (scores - min_score) / (max_score - min_score)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        quality_score = self.quality_scores[idx]

        return image, quality_score