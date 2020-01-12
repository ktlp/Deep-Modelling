from abc import ABC, abstractmethod
from .base_dataset import BaseDataset
from .base_net import BaseNet


class BaseTrainer(ABC):
    """Trainer base class. Compatible with sklearn modules."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def fit(self, dataset: BaseDataset, net: BaseNet):
        """
        Implements fit method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def predict(self, dataset: BaseDataset, net: BaseNet):
        """
        Implement predict method that evaluates the model output for a given input.
        """
        pass

    @abstractmethod
    def score(self, dataset: BaseDataset, net: BaseNet):
        """
        Implements the score method that evaluates the performance of the model for the test_set.
        """
        pass