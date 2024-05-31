import torchmetrics
import pytorch_lightning as pl


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=8)
        self.iouscore = torchmetrics.JaccardIndex(
            task='multiclass', num_classes=8)
        self.f1score = torchmetrics.F1Score(task='multiclass', num_classes=8)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch
        out = self.forward(image)
        loss = self.criterion(out, mask)
        acc = self.accuracy(out, mask)
        iouscore = self.iouscore(out, mask)
        f1score = self.f1score(out, mask)
        self.log(f"{stage}_iouscore", iouscore, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_accuracy", acc, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_f1score", f1score, prog_bar=True, on_epoch=True)

        return {"loss": loss, "iouscore": iouscore, "accuracy": acc, "f1score": f1score}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def configure_optimizers(self):
        return self.optimizer
