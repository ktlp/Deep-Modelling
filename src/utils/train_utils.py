import torch
import torch.optim as optim
import numpy as np
import os

class Weighted_mse():
	def __init__(self):
		self.MAX = 10 ** 6
	def __call__(self, outputs, targets):
		weights = torch.sqrt(torch.abs(1 / targets))
		weights[weights > self.MAX] = self.MAX
		loss = torch.mean(weights * (outputs - targets) ** 2)
		return loss

class Success_rate():
	def __init__(self):
		self.counter = 0
		self.epoch_counter = 0
		self.input_counter = 0

	def __call__(self, inputs, outputs, targets):
		self.input_counter += inputs.size(0)
		self.counter += inputs.size(0) - torch.nonzero(torch.max(outputs, dim=1)[1] - targets).size(0)
		return

	def zero(self):
		self.counter = 0
		self.epoch_counter = 0
		self.input_counter = 0
		return

	@property
	def value(self):
		return self.counter/self.input_counter

class Mean_accuracy():
	def __init__(self):
		self.counter = 0
		self.epoch_counter = 0
		self.input_counter = 0
		self.DELTA = 1e-6

	def __call__(self, inputs, outputs, targets):
		self.input_counter += inputs.size(0)
		diff = torch.abs((outputs - targets) )/(targets + self.DELTA)
		self.counter += torch.mean(diff)
		return

	def zero(self):
		self.counter = 0
		self.epoch_counter = 0
		self.input_counter = 0
		return

	@property
	def value(self):
		return 1 - self.counter.item()/self.input_counter

class EarlyStopping():

	def __init__(self, patience=7, metric = 'accuracy'):
		assert metric in ['accuracy', 'loss'], 'Metric monitores should be one of accuracy, loss'
		self.metric = metric
		self.patience = patience
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf


	def __call__(self, metric, model):

		if self.metric == 'accuracy':
			score = metric
		else:
			score = metric*(-1)

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(score, model)
		elif score < self.best_score:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(score, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		torch.save(model.state_dict(), 'checkpoint.pt')
		self.val_loss_min = val_loss

	def load_best_model(self, model):
		model_dict = torch.load('checkpoint.pt')
		model.load_state_dict(model_dict)
		os.remove('checkpoint.pt')
