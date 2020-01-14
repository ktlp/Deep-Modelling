import torch
import torch.optim as optim

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

	def __call__(self, epoch, inputs, outputs, targets):
		if self.epoch_counter == epoch:
			self.input_counter += inputs.size(0)
			self.counter += inputs.size(0) - torch.nonzero(torch.max(outputs, dim=1)[1] - targets).size(0)
		else:
			self.input_counter = inputs.size(0)
			self.counter = inputs.size(0) - torch.nonzero(torch.max(outputs, dim=1)[1] - targets).size(0)
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

	def __call__(self, epoch, inputs, outputs, targets):
		if self.epoch_counter == epoch:
			self.input_counter += inputs.size(0)
			diff = torch.abs((outputs - targets) )/(targets + self.DELTA)
			self.counter += torch.mean(diff)
		else:
			self.input_counter = inputs.size(0)
			diff = torch.abs((outputs - targets) )/(targets + self.DELTA)
			self.counter = torch.mean(diff)
		return

	@property
	def value(self):
		return 1 - self.counter.item()/self.input_counter
