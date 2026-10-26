import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class CARS(Dataset):
    """
    CARS Dataset
    You should download this dataset from below url.
    url: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    """

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): path to dataset
        """
        self.data_path = data_path
        self.transform = transform
        self.fpaths = sorted(glob.glob(os.path.join(data_path, "*.jpg")))
        gray_lst = [
            266, 1085, 2176, 3048, 3439, 3469, 3539, 4577, 4848,
            5177, 5502, 5713, 6947, 7383, 7693, 7774, 8137, 8144,
        ]
        for num in gray_lst:
            self.fpaths.remove(os.path.join(data_path, "{:05d}.jpg".format(num)))

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.fpaths[idx]))
        return img

    def __len__(self):
        return len(self.fpaths)
