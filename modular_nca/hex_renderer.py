# TODO
# Implement multi-view layout logic based on num hidden layers

import numpy as np
import cv2
import torch

CELL_BORDER_AMOUNT = 0.05
GRID_BORDER_AMOUNT = 0.01
ZOOM_STRENGTH = 25

class HexRenderer:
    
	def __init__(self, preview_size, model):

		self.model = model
		self.size = preview_size

		w, h = preview_size
		limiting_dim = min(h, w)

		self.rad = (limiting_dim * (1 - GRID_BORDER_AMOUNT * 2)) / 2

		self.centre = (round(w/2), round(h/2))
		self.origin = (round(self.centre[0] - self.rad), round(self.centre[1] - self.rad))

		self.preview = np.zeros((h, w, 3), dtype=np.uint8)

		angle_step = 2*np.pi/6

		self.hex_angles = np.linspace(angle_step/2, 2*np.pi-angle_step/2, 6)

		self.hex_unit_vectors = np.zeros((6, 2), dtype=np.float32)
		self.hex_unit_vectors[:, 0] = np.sin(self.hex_angles)[:]
		self.hex_unit_vectors[:, 1] = np.cos(self.hex_angles)[:]

		self.cull_mask = None
		self.arr_mapped_vectors = []
		self.view_centre = None
		self.view_radius = None

		self.recalculate_view(0, np.array((0, 0)))

	def recalculate_view(self, view_scale, view_offset):

		locs = self.model.layout.locs

		zoom_scale = 1 + view_scale * ZOOM_STRENGTH
		rad_scaling = self.rad * zoom_scale
		cell_scaling = self.model.layout.cell_rad * rad_scaling

		self.view_centre = (self.centre + view_offset * rad_scaling).astype(np.int32)
		self.view_radius = round(rad_scaling)

		hex_scaled_vectors = self.hex_unit_vectors * cell_scaling

		translated_locs = locs + view_offset
		scaled_locs = translated_locs * rad_scaling

		cull_thresh = (1/zoom_scale) ** 2
		self.cull_mask = np.where((translated_locs[:, 0]**2 + translated_locs[:, 1]**2) < cull_thresh)[0]

		arr_mapped_vectors = scaled_locs[:, None, :] + (self.centre + hex_scaled_vectors)
		self.arr_mapped_vectors = np.round(arr_mapped_vectors).astype(np.int32)

	def draw_state(self, mode, readout=False):

		colour_source = self.model.state[:, :3]
		label = self.model.labels[mode-1]

		# Animating, showing visible layers
		if mode == 1:
			pass

		# Animating, showing alpha layer
		elif mode == 2:
			colour_source = self.model.state[:, 3]

		# Animating, showing hidden layers
		elif mode > 2:
			ls = (mode-2)*3+1
			le = (mode-1)*3+1

			colour_source = self.model.state[:, ls:le]


		self.preview[:] = 0

		#cv2.circle(self.preview, self.centre, round(self.rad), [255, 255, 255], 2)
		#cv2.polylines(self.preview, self.arr_mapped_vectors[self.cull_mask], True, (255, 255, 255), 1)

		colours = torch.clip(colour_source[self.cull_mask] * 255, 0, 255).to(torch.uint8)
		for i in range(len(self.cull_mask)):

			cell_index = self.cull_mask[i]				
			cv2.fillPoly(self.preview, [self.arr_mapped_vectors[cell_index]], colours[i].tolist())
		

		if readout:
			cv2.putText(self.preview, "step: %d | %s" %(self.model.age, label), (0, int(self.preview.shape[0] - self.preview.shape[0]*GRID_BORDER_AMOUNT)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 


	def update(self, mode, selected=None, adjusting_view=False):

		# Selection
		if selected is not None:
			selected_param  = (selected - self.view_centre) / self.view_radius
			self.model.set_selection(selected_param)


		output = self.preview

		if mode > int((self.model.num_layers-1)/3+1):

			layer_sets = []
			for i in range(6):

				if i < (mode-1):
					self.draw_state(i+1, not adjusting_view)
					layer_sets.append(self.preview.copy())

				else:
					layer_sets.append(np.zeros_like(self.preview))
			
			if mode < 6:
				upper = np.hstack((layer_sets[0], layer_sets[1]))
				lower = np.hstack((layer_sets[2], layer_sets[3]))
			else:
				upper = np.hstack((layer_sets[0], layer_sets[1], layer_sets[2]))
				lower = np.hstack((layer_sets[3], layer_sets[4], layer_sets[5]))
				
			output = np.vstack((upper, lower))

		else:
			self.draw_state(mode, readout=(mode > 0 and not adjusting_view))

		return output


