from src.base.base_trainer import BaseTrainer
from src.base.base_dataset import BaseDataset
from src.base.base_net import BaseNet

import logging
import time

import torch
import torch.optim as optim
import torch.nn as nn

class Regression_Trainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 20,
                 lr_milestones: tuple = (), batch_size: int = 16, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    @staticmethod
    def weighted_mse(outputs, targets):
        MAX = 10**6
        weights = torch.sqrt(torch.abs(1/targets))
        weights[weights > MAX] = MAX
        loss = torch.mean(weights*(outputs - targets) ** 2)
        return loss

    def fit(self, dataset: BaseDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Early stopping

        # Loss criterion
        self.criterion = self.weighted_mse
        #self.criterion = nn.MSELoss()
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            dist = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, y = data

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.predict(inputs, net)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

                diff = torch.abs((outputs - y)/(y+1e-3))
                dist += torch.mean(diff)

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t Accuracy: {:.4f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, 1-dist.item()/n_batches))
        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def score(self, dataset: BaseDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        loss_total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, y = data

                outputs = self.predict(inputs, net)
                loss = self.criterion(outputs, y)
                loss_total += loss.item()

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = loss_total

    def predict(self, input: BaseDataset, net: BaseNet):
        return net(input)

