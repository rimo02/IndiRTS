import torch
from src.model import CreateModel
from dataset import Create_DataLoader
from torchsummary import summary
import torch.nn as nn
import pytorch_lightning as pl
from logger.callbacks import MetricsPlotter
from trainer.trainer import SegmentationModel


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


BATCH_SIZE = 32
HEIGHT = 224
WIDTH = 224
device = get_default_device()

# Dataset path
dataset_path = './idd20k_lite/'
img_train = dataset_path + 'leftImg8bit/train/'
img_val = dataset_path + 'leftImg8bit/val/'
train_dl,  val_dl = Create_DataLoader(
    BATCH_SIZE, HEIGHT, WIDTH, img_train, img_val)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

dataset = {"train": train_dl, "val": val_dl}

model = CreateModel(8).to(device)
print(summary(model, input_size=(3, 224, 224)))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
criterion = nn.CrossEntropyLoss().to(device)
pl_model = SegmentationModel(model, optimizer, criterion)
cbs = MetricsPlotter()

trainer = pl.Trainer(callbacks=cbs, max_epochs=100)
trainer.fit(pl_model, dataset['train'], dataset['val'])
print("Model training done...")
