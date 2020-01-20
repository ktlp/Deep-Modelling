from src.base.base_trainer import BaseTrainer
from src.base.base_dataset import BaseDataset
from src.base.base_net import BaseNet
from src.models.Regressor import Regressor
from src.models.Classifier import Classifier
from src.utils.train_utils import Mean_accuracy, Success_rate, Weighted_mse, EarlyStopping
import logging
import time

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

class Trainer(BaseTrainer):
	def __init__(self, network, optimizer_name: str = 'adam', early_stopping = 'False', lr: float = 0.001, n_epochs: int = 20,
				 lr_milestones: tuple = (), batch_size: int = 16, weight_decay: float = 1e-6, device: str = 'cuda',
				 n_jobs_dataloader: int = 0):
		super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
						 n_jobs_dataloader)

		# assertion
		assert isinstance(network, (Regressor, Classifier))
		if isinstance(network, Regressor):
			# set up criterion, acuracy metric
			self.criterion = Weighted_mse()
			self.accuracy = Mean_accuracy()
		else:
			# set up criterion, accuracy metric
			self.criterion = nn.CrossEntropyLoss()
			self.accuracy = Success_rate()

		self.early_stopping = early_stopping

		# Results
		self.train_time = None
		self.test_auc = None
		self.test_time = None
		self.test_scores = None

	def train_epoch(self, net :BaseNet, train_loader ):
		self.accuracy.zero()
		net.train()
		self.scheduler.step()

		loss_epoch = 0.0
		n_batches = 0
		dist = 0
		epoch_start_time = time.time()
		for data in train_loader:
			inputs, y = data

			# Zero the network parameter gradients
			self.optimizer.zero_grad()

			# Update network parameters via backpropagation: forward + backward + optimize
			outputs = self.predict(inputs, net)
			loss = self.criterion(outputs, y)
			if torch.isnan(loss):
				raise ValueError('loss is nan while training')
			self.accuracy(inputs, outputs, y)
			loss.backward()
			self.optimizer.step()

			loss_epoch += loss.item()
			n_batches += 1

		# log epoch statistics
		epoch_train_time = time.time() - epoch_start_time
		return loss_epoch, self.accuracy.value, epoch_train_time, n_batches


	def fit(self, dataset: BaseDataset, net: BaseNet, validate = 0):

		# initialize logger
		self.logger = logging.getLogger()

		# Get train data loader
		train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

		# Set optimizer (Adam optimizer for now)
		self.optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
							   amsgrad=self.optimizer_name == 'amsgrad')
		# Set learning rate scheduler
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=0.1)

		# Early Stopping
		if self.early_stopping:
			early_stopping = EarlyStopping(patience=15)

		self.logger.info('Starting training...')
		start_time = time.time()


		for epoch in range(self.n_epochs):

			# lr scheduler update
			if epoch in self.lr_milestones:
				self.logger.info('  LR scheduler: new learning rate is %g' % float(self.scheduler.get_lr()[0]))

			# train epoch
			loss_epoch, accuracy, epoch_train_time, n_batches = self.train_epoch(net, train_loader)

			# Validate on test set and log
			if validate > 0 :
				validation_score, validation_time, validation_accuracy = self.score(dataset, net)
				self.logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t Accuracy: {:.4f}\t Validation Time: {:.3f}\t Validation Loss: {:.4f}\t Validation Accuracy: {:.4f}'
								 .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch,
										 self.accuracy.value, validation_time,validation_score, validation_accuracy  ))
			else:
				self.logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t Accuracy: {:.4f}'
						.format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch ,
								self.accuracy.value))

			# check early stopping
			if self.early_stopping:
				early_stopping(self.accuracy.value, net)
				if early_stopping.early_stop:
					# load best model so far
					early_stopping.load_best_model(net)
					print('Early Stopping..')
					break

		self.train_time = time.time() - start_time
		self.logger.info('Training time: %.3f' % self.train_time)
		self.logger.info('Finished training.')
		return net

	def score(self, dataset: BaseDataset, net: BaseNet):

		# Get test data loader
		_, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

		# Testing
		self.accuracy.zero()

		start_time = time.time()

		loss_total = 0
		with torch.no_grad():
			for data in test_loader:
				inputs, y = data

				outputs = self.predict(inputs, net)
				loss = self.criterion(outputs, y)
				loss_total += loss.item()
				self.accuracy(inputs,outputs, y)
		test_time = time.time() - start_time

		return loss_total, test_time, self.accuracy.value

	def predict(self, input: BaseDataset, net: BaseNet):
		return net(input)

