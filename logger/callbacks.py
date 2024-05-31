import pytorch_lightning as pl
import matplotlib.pyplot as plt
from collections import defaultdict


class MetricsPlotter(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(list)

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        self.metrics['train_loss'].append(
            logs['train_loss'].cpu().numpy())
        self.metrics['val_loss'].append(logs['valid_loss'].cpu().numpy())
        self.metrics['train_accuracy'].append(
            logs['train_accuracy'].cpu().numpy())
        self.metrics['val_accuracy'].append(
            logs['valid_accuracy'].cpu().numpy())
        self.metrics['train_f1score'].append(
            logs['train_f1score'].cpu().numpy())
        self.metrics['val_f1score'].append(
            logs['valid_f1score'].cpu().numpy())
        self.metrics['train_iouscore'].append(
            logs['train_iouscore'].cpu().numpy())
        self.metrics['val_iouscore'].append(
            logs['valid_iouscore'].cpu().numpy())

    def on_fit_end(self, trainer, pl_module):
        self.plot_metrics()

    def plot_metrics(self):
        epochs = range(1, len(self.metrics['train_loss']) + 1)

        plt.figure(figsize=(20, 10))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.metrics['train_loss'], label='Train Loss')
        plt.plot(epochs, self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(
            epochs, self.metrics['train_accuracy'], label='Train Accuracy')
        plt.plot(epochs, self.metrics['val_accuracy'],
                 label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.metrics['train_f1score'], label='Train F1 Score')
        plt.plot(epochs, self.metrics['val_f1score'],
                 label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')

        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(
            epochs, self.metrics['train_iouscore'], label='Train F1 Score')
        plt.plot(epochs, self.metrics['val_iouscore'],
                 label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.tight_layout()
        plt.show()
