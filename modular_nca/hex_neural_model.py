import numpy as np
import torch

from .hex_model import HexModel
from .hex_nn import HexNN
from .config import checkpoint_dir

grad_factor = 0.75
living_threshold = 0.2
cell_fire_rate = 0.8

nn_input_entries = 8

class HexNeuralModel(HexModel):

	def __init__(self, num_hidden_layers, nn_hidden_layer_sizes, divisions):

		labels = ["rgb", "a"]
		labels.extend(["h%d-%d" %(x*3, (x+1)*3) for x in range(int(num_hidden_layers/3))])

		super().__init__(4 + num_hidden_layers, divisions, labels)

		self.nn = HexNN(input_size=(self.num_layers*nn_input_entries), output_size=self.num_layers, hidden_layer_sizes=nn_hidden_layer_sizes)

		self.neighbours = self.layout.neighbour_indices.copy()

		self.custom_mask = torch.zeros((self.cell_count, 3), dtype=self.state.dtype)
		
		self.border_mask = torch.zeros((self.cell_count), dtype=torch.bool)
		self.border_mask[self.layout.border_indices] = True

		self.update_mask = torch.ones((self.cell_count), dtype=torch.bool)

	def scramble_neighbours(self):

		self.neighbours = np.take(self.layout.neighbour_indices, np.random.permutation(self.layout.neighbour_indices.shape[0]), axis=0)

	def set_mask(self, hex_mask):

		self.custom_mask = hex_mask

		self.update_mask = (torch.mean(self.custom_mask, axis=1) < living_threshold) & ~self.border_mask

	def blit_mask(self):

		self.state[:, :3] = self.custom_mask

	def preview_mask(self, hex_mask):

		self.state[:, 0].copy_(hex_mask)

	def reset_grid_rand(self):
		
		self.state = torch.rand(*self.state.shape) * 0.1	

		self.state[:, :4] = torch.tensor([0.0, 0.0, 0.0])
		self.state[self.layout.center_index, :4] = torch.tensor([1.0, 1.0, 1.0])

		self.last_state.copy_(self.state)

		self.age = 0

	def reset_grid_seed(self, hex_image):

		self.state = torch.zeros(*self.state.shape)
		self.last_state = torch.zeros(*self.state.shape)
		
		self.blit_mask()
		self.state[:, :4] = hex_image

		self.last_state.copy_(self.state)
		
		self.age = 0

	def load_nn(self, model_epoch_num):

		self.nn.load_state_dict(torch.load(checkpoint_dir + "model-%d.tar"%model_epoch_num))
		self.nn.eval()

	def step(self):

		cell_count = len(self.state)
		inputs = torch.zeros((cell_count, nn_input_entries, self.num_layers), dtype=torch.float32)

		# Set inputs
		grad_t = self.state - self.last_state * grad_factor

		neighbour_state = self.last_state[self.neighbours]
		neighbour_grad_t = grad_t[self.neighbours]
		
		for i in range(3):
			inputs[:, i] = 2 * neighbour_state[:, i] -2 * neighbour_state[:, i+3] + neighbour_state[:, i+1] - neighbour_state[:, i+2] - neighbour_state[:, (i+4)%6] + neighbour_state[:, (i+5)%6]

		for i in range(3):
			inputs[:, i+3] = 2 * neighbour_grad_t[:, i] -2 * neighbour_grad_t[:, i+3] + neighbour_grad_t[:, i+1] - neighbour_grad_t[:, i+2] - neighbour_grad_t[:, (i+4)%6] + neighbour_grad_t[:, (i+5)%6]
		
		inputs[:, 6] = self.last_state
		inputs[:, 7] = grad_t

		# Get outputs from nn forward pass - only allow some cells to update

		fire_mask = (torch.rand(cell_count) < cell_fire_rate) & self.update_mask
		updates = self.nn(inputs.view((cell_count, -1))[fire_mask])

		self.state[fire_mask] = self.state[fire_mask] + updates

		# Prevent communication between 'dead' cells

		alpha = self.state[:, 3]
		neighbour_max_alpha, _ = torch.max(alpha[self.neighbours], dim=1)

		dead_mask = torch.where((alpha < living_threshold) & (neighbour_max_alpha < living_threshold))
		self.state[dead_mask] = 0

		# Step
		self.last_state.copy_(self.state)

		self.state.requires_grad_(True)

		self.age += 1
