import numpy as np
import torch

class HexModel:

	def __init__(self, num_layers, divisions, labels):

		self.num_layers = num_layers

		self.layout = HexLayout(divisions)
		self.cell_count = len(self.layout.locs)

		self.labels = labels

		self.state = torch.zeros((self.cell_count, self.num_layers), dtype=torch.float32)
		self.last_state = torch.zeros_like(self.state)

		self.age = 0
	
	def blit(self, hex_image):

		self.state[:, :3].copy_(hex_image)

	def set_selection(self, selection_param):

		selected_index = None

		dist = np.sum((selection_param - self.layout.locs) ** 2, axis=1)

		min_index = np.argmin(dist)

		if dist[min_index] < self.layout.cell_rad**2:
			
			self.on_select(min_index)
	
	def on_select(self, cell_index):

		print("Selection:", cell_index)


	def step(self):

		self.age += 1

	"""
	def highlight(self, cell_index):

		self.state = torch.rand(*self.state.shape)
		self.state[cell_index, :3] = [1, 1, 1]

		colours = torch.zeros((6, 3), dtype=torch.float32)
		colours[:, 0] = torch.linspace(0, 1, 6)

		ns = self.layout.neighbour_indices[cell_index]

		state_mask = ns != -1
		ns_mask = torch.where(state_mask)

		self.state[ns[state_mask], :3] = colours[ns_mask]
	"""

class HexLayout:

	def __init__(self, target_divisions):

		# Ensure odd
		divisions = int(target_divisions/2)*2 + 1

		hex_height = 2.0 / divisions
		hex_radius = (((hex_height/2.0)**2)/0.75)**0.5

		self.cell_rad = hex_radius

		x_spacing = hex_radius * 1.5
		y_spacing = hex_height

		self.num_rows = divisions
		self.num_cols = 2 * int(divisions/2) + 1

		center_col = int(self.num_cols/2)

		# Compute locs

		self.locs = []
		
		border_indices = []
		self.grid_indices = np.full((self.num_cols, self.num_rows), -1, dtype=int)

		start_x = -(x_spacing * self.num_cols)/2.0 + x_spacing/2.0
		start_y = -hex_height*divisions/2.0 + hex_height/2.0

		cell_index = 0
		for i in range(self.num_cols):

			dist = i - center_col

			cur_row_count = self.num_cols - abs(dist)

			offset_y = abs(dist)/2*y_spacing
			for j in range(cur_row_count):
				
				cx = start_x + i * x_spacing
				cy = start_y + offset_y + j * y_spacing

				self.locs.append((cx, cy))

				self.grid_indices[i, j] = cell_index

				if i == 0 or i == (self.num_cols - 1) or j == 0 or j == (cur_row_count - 1):
					border_indices.append(cell_index)

				cell_index += 1
				

		self.locs = np.array(self.locs)
		self.border_indices = np.array(border_indices)
		border_mapping = np.array([3, 4, 5, 0, 1, 2]) # n, ne, se, s, sw, nw -> s, sw, nw, n, ne, se

		self.neighbour_indices = np.full((len(self.locs), 6), -1, dtype=np.int32)

		self.center_index = self.grid_indices[center_col, self.num_rows//2]

		cell_index = 0
		for i in range(self.num_cols):

			dist = i - center_col
			cur_row_count = self.num_cols - abs(dist)

			for j in range(cur_row_count):
				
				n, ne, se, s, sw, nw = self.neighbour_indices[cell_index]

				shift_e = 0 if dist < 0 else -1
				shift_w = 0 if dist < 1 else 1

				if i > 0:
					if (j + shift_w) < self.num_rows:
						sw = self.grid_indices[i-1, j + shift_w]
					if (j + shift_w) > 0:
						nw = self.grid_indices[i-1, j-1 + shift_w]

				if  i < (self.num_cols - 1):
					if (j + shift_e) > -1:
						ne = self.grid_indices[i+1, j + shift_e]
					if (j + shift_e) < (self.num_rows - 1):
						se = self.grid_indices[i+1, j+1 + shift_e]
				
				if j > 0:
					n = self.grid_indices[i, j-1]

				if j < (self.num_rows - 1):
					s = self.grid_indices[i, j+1]

				
				# Determine neighbours for border cells, following rotational symmetry

				self.neighbour_indices[cell_index] = (n, ne, se, s, sw, nw)

				if cell_index in self.border_indices:

					partner_i = self.num_cols - 1 - i
					partner_j = cur_row_count - 1 - j

					partner_index = self.grid_indices[partner_i, partner_j]

					missing_indices = np.where(self.neighbour_indices[cell_index] == -1)

					self.neighbour_indices[cell_index, missing_indices] = partner_index

				cell_index += 1


	def get_col(self, col):

		ids = self.grid_indices[col, :]

		return ids[ids != -1]
