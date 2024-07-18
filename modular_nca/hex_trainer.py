# TODO
# Migrate hyperparameters
# Clean up goal logic

import torch.optim as optim
import torch
import torch.nn as nn
import os

from.config import checkpoint_dir

learning_rate = 0.001
checkpoint_count = 5000
iter_count = 1
step_range = (80, 90)

epoch_readout_interval = 1
batch_coverage = 1

gradient_norm = 3

class HexNNTrainer:

	def __init__(self, model, dataset):

		self.model = model
		self.dataset = dataset
	
		self.loss_fn = nn.MSELoss() #nn.BCELoss()
		self.optimizer = optim.Adam(model.nn.parameters(), lr=learning_rate) #optim.SGD(model.nn.parameters(), lr=learning_rate, momentum=0.8)
		
		self.last_loss = 1

	def learn_batch(self, num_epochs):

		self.run(num_epochs, goal="distribution")

	def learn_time_series(self, num_epochs, stability_pre_iters=0, stability_post_iters=0):

		self.run(num_epochs, goal="sequential", add_start_iters=stability_pre_iters, add_end_iters=stability_post_iters)

	def learn_reconstruction_goal(self, num_epochs, stability_pre_iters=0, stability_post_iters=0):

		self.run(num_epochs, goal="pairwise", add_start_iters=stability_pre_iters, add_end_iters=stability_post_iters)

	def run(self, num_epochs, goal="distribution", add_start_iters=0, add_end_iters=0):

		sample_count = len(self.dataset.samples)
		batch_indices = None

		if goal == "sequential":
			iter_count = sample_count + add_start_iters + add_end_iters - 1
			batch_indices = torch.arange(-add_start_iters, sample_count + add_end_iters)

		for file in os.listdir(checkpoint_dir):
			os.remove(checkpoint_dir + file)

		# Setup

		batch_size = round(batch_coverage * sample_count)
		completed_checkpoints = 0

		self.model.nn.train()

		loss = None

		for epoch in range(num_epochs):

			param = epoch/(num_epochs-1)
			if goal == "distribution":
				batch_indices = torch.randint(0, sample_count, (batch_size, ))
				self.model.reset_grid_rand()

			elif goal == "sequential":

				self.model.reset_grid_seed(self.dataset.samples[0])

			elif goal == "pairwise":
				batch_indices = torch.randint(0, int(sample_count/2), (iter_count, )) * 2

			total_loss = 0
			self.optimizer.zero_grad()

			for i in range(iter_count):

				loss_indices = batch_indices
				if goal == "sequential":
					
					mask_index = torch.clip(batch_indices[i], 0, sample_count-1)

					self.model.set_mask(self.dataset.masks[mask_index])
					loss_indices = torch.clip(batch_indices[i+1], 0, sample_count-1)[None]

				elif goal == "pairwise":
					self.model.reset_grid_seed(self.dataset.samples[batch_indices[i]])
					self.model.set_mask(self.dataset.masks[int(batch_indices[i]/2)])
					loss_indices = batch_indices[None, i]+1

				
				for j in range(torch.randint(step_range[0], step_range[1], (1,))):
					self.model.step()

				loss = self.get_loss(loss_indices)
				total_loss += loss

				loss.backward(retain_graph=True)

			if gradient_norm is not None:
				torch.nn.utils.clip_grad_norm_(self.model.nn.parameters(), max_norm=gradient_norm)
			
			self.optimizer.step()
			self.model.state.detach()

			self.last_loss = float(total_loss.item())

			if (epoch+1) % epoch_readout_interval == 0:
				print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.8f}')

			if ((epoch+1) == round(num_epochs * ((completed_checkpoints + 1)/checkpoint_count))):

				torch.save(self.model.nn.state_dict(), checkpoint_dir + "model-%d.tar" %(epoch+1))
				completed_checkpoints += 1

	def get_loss(self, batch_indices):

		observed = self.model.state[:, :4]
		batch = self.dataset.samples[batch_indices]
		
		loss = self.loss_fn(observed.unsqueeze(0).expand_as(batch), batch)
		
		return loss
		


