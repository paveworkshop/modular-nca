# TODO
# Update batch and reconstruction loaders to accomodate alpha channel

import os
import cv2
import random
import torch
import numpy as np

def rotate_image(src, angle):

	h, w, _ = src.shape
	centre = (w//2, h//2)
	M = cv2.getRotationMatrix2D(centre, angle, 1.0)

	return cv2.warpAffine(src, M, (w, h))

class HexDataset:

	def __init__(self, model):

		self.model = model
		self.samples = None
		self.masks = None

	def load_time_series(self, source_folder, start=0, end=1, stride=1, blur_strength=0.5):

		files = sorted(os.listdir(source_folder))

		time_files = []
		mask_files = []

		for file in files:
			
			if file.count("mask"):
				mask_files.append(file)

			else:
				time_files.append(file)

		param_locs = (1 + self.model.layout.locs) / 2
		sample_indices = np.arange(round(len(time_files)*start), round(len(time_files)*end), stride, dtype=np.int32)

		dataset_size = len(sample_indices)
	
		model_arr = self.model.state

		self.samples = torch.zeros((dataset_size, model_arr.shape[0], 4), dtype=model_arr.dtype)
		self.masks = torch.zeros((dataset_size, model_arr.shape[0], 3), dtype=model_arr.dtype)

		for i in range(dataset_size):

			sample_index = sample_indices[i]

			self.samples[i] = self.read_and_sample(source_folder+time_files[sample_index], param_locs, blur_strength, alpha=True)

			if sample_index < len(mask_files):
				self.masks[i] = self.read_and_sample(source_folder+mask_files[sample_index], param_locs, blur_strength)
			else:
				self.masks[i].copy_(self.masks[i-1])

		print("Created %d time series samples from dataset path %s." %(dataset_size, source_folder))

	def load_reconstruction_set(self, source_folder, dataset_coverage, blur_strength=0.5):

		pre_files = sorted([f for f in os.listdir(source_folder) if f.count("pre")])
		post_files = sorted([f for f in os.listdir(source_folder) if f.count("post")])
		mask_files = sorted([f for f in os.listdir(source_folder) if f.count("mask")])

		file_sets = []

		for i in range(min(len(pre_files), len(post_files))):
			file_sets.append((pre_files[i], post_files[i], mask_files[i]))

		param_locs = (1 + self.model.layout.locs) / 2

		pair_count = int(len(file_sets)*dataset_coverage)
		dataset_size = pair_count * 2
	
		model_arr = self.model.state

		self.samples = torch.zeros((dataset_size, model_arr.shape[0], 3), dtype=model_arr.dtype)
		self.masks = torch.zeros((pair_count, model_arr.shape[0]), dtype=model_arr.dtype)


		for i in range(pair_count):

			pre_file, post_file, mask_file = file_sets[i]

			pre_sample_vals = self.read_and_sample(source_folder+pre_file, param_locs, blur_strength)
			post_sample_vals = self.read_and_sample(source_folder+post_file, param_locs, blur_strength)
			mask_sample_vals = self.read_and_sample(source_folder+mask_file, param_locs, blur_strength, grey=True)

			self.samples[i*2] = pre_sample_vals
			self.samples[i*2+1] = post_sample_vals
			self.masks[i] = mask_sample_vals

		print("Created %d reconstruction set pairs from dataset path %s." %(pair_count, source_folder))

	def read_and_sample(self, filepath, param_locs, blur_strength, grey=False, alpha=False):

		raw = cv2.imread(filepath)

		# Take red channel
		if grey:
			raw = raw[:, :, 2]

		sample_w = int(raw.shape[1])
		sample_h = int(raw.shape[0])

		blur_size = int(blur_strength * min(sample_w, sample_h) / 2) * 2 + 1
		blurred = cv2.GaussianBlur(raw, (blur_size, blur_size), 0)

		min_dim = min(sample_w, sample_h)

		sample_locs = np.round(param_locs * min_dim).astype(np.int32)

		sample_vals = blurred[sample_locs[:, 1], sample_locs[:, 0]] / 255


		result = torch.tensor(sample_vals)
		# Infer alpha from non-zero
		if alpha:
			
			extended = torch.zeros((result.shape[0], 4))
			extended[:, :3] = result
			extended[:, 3][torch.mean(result, axis=1) > 0] = 1

			result = extended

		return result

	def load_images_and_sample(self, source_folder, samples_per_image=10, sample_coverage=0.5, blur_strength=0.5):
		
		param_locs = (1 + self.model.layout.locs) / 2

		files = os.listdir(source_folder)
		dataset_size = samples_per_image * len(files)
		
		model_arr = self.model.state
		self.samples = torch.zeros((dataset_size, model_arr.shape[0], 3), dtype=model_arr.dtype)

		for i in range(len(files)):

			raw = cv2.imread(source_folder+files[i])

			sample_w = int(raw.shape[1] * sample_coverage)
			sample_h = int(raw.shape[0] * sample_coverage)

			blur_size = int(blur_strength * min(sample_w, sample_h) / 2) * 2 + 1
			blurred = cv2.GaussianBlur(raw, (blur_size, blur_size), 0)

			for j in range(samples_per_image):

				sample_index = i * samples_per_image + j
				
				sx, sy = (random.randint(0, blurred.shape[1] - sample_w), random.randint(0, blurred.shape[0] - sample_h))
				cropped = blurred[sy:sy+sample_h, sx:sx+sample_w]

				rotated = cropped #rotate_image(cropped, random.choice([0, 90, 180, 270]))

				h, w, _ = rotated.shape
				min_dim = min(w, h)

				sample_locs = np.round(param_locs * min_dim).astype(np.int32)

				sample_vals = rotated[sample_locs[:, 0], sample_locs[:, 1]] / 255

				self.samples[sample_index, :] = torch.tensor(sample_vals)

		print("Created %d samples (%d source images) from dataset path %s." %(dataset_size, len(files), source_folder))
