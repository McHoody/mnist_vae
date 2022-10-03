import pytorch_lightning as pl
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, download: bool = False):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.download = download

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None) -> None:
        transformations = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        self.train_dataset = MNIST(self.data_dir, train=True, transform=transformations, download=self.download)
        self.test_dataset = MNIST(self.data_dir, train=False, transform=transformations, download=self.download)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
