# TODO
# Scrolling on when app focused

import numpy as np
import cv2

from pynput import mouse # Quick fix for scroll detection on Mac

WINDOW_NAME = "Preview"

PAN_SPEED = 0.001
SCROLL_SPEED = 0.001

class Viewer:

	def __init__(self, renderer):

		self.scale = 0

		self.view_offset = np.array((0, 0), dtype=np.float32)

		self.renderer = renderer
		self.renderer_view_out_of_date = False

		self.mousedown = False
		self.last_mouse_pos = np.array((0, 0))
		self.select_pos = None

		self.recording_settings = None
		self.recording = None

		self.preview_size = None

		self.frame_count = 0

	def set_recording_settings(self, name, frame_count, fps=12):

		self.recording_settings = (name, frame_count, fps)

	def save_frame(self, frame):

		file_name, frame_count, fps = self.recording_settings

		scaling=1.0

		if self.recording is None:

			resized = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling)
			height, width, _ = resized.shape

			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			self.recording = cv2.VideoWriter(file_name+".mp4", fourcc, fps, (width, height))

		resized = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
		self.recording.write(resized)
		
		if self.frame_count >= frame_count:
			self.recording.release()
			self.recording_settings = None
			print("Saved recording:", file_name)

	def start(self, mode, on_frame=None):

		cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
		cv2.setMouseCallback(WINDOW_NAME, self.on_mouse_click_event)

		mouse_listener = mouse.Listener(on_scroll=self.on_mouse_scroll_event)
		mouse_listener.start()

		while True:
			
			if self.renderer_view_out_of_date:
				self.renderer.recalculate_view(self.scale, self.view_offset)
				self.renderer_view_out_of_date = False

			preview = self.renderer.update(mode, self.select_pos, adjusting_view=self.mousedown)
			self.select_pos = None

			if self.recording_settings is not None:
				self.save_frame(preview)


			if self.preview_size != preview.shape:
				cv2.resizeWindow(WINDOW_NAME, preview.shape[1], preview.shape[0]) 
				self.preview_size = preview.shape

			cv2.imshow(WINDOW_NAME, preview)

			if on_frame is not None:
				on_frame()

			self.frame_count += 1

			key = cv2.waitKey(1)

			if key == ord("q"):
				break

			if key == ord("r"):
				self.view_offset[:] = 0
				self.renderer_view_out_of_date = True

		cv2.destroyAllWindows()
		mouse_listener.stop()

	def on_mouse_click_event(self, event, x, y, flags, param):
		
		if event == cv2.EVENT_MOUSEMOVE:
			
			if self.mousedown:
				cur_pos = np.array((x, y), dtype=np.float32)
				self.view_offset += (cur_pos - self.last_mouse_pos) * PAN_SPEED
				
				self.renderer_view_out_of_date = True
				self.last_mouse_pos = cur_pos

		elif event == cv2.EVENT_RBUTTONDOWN:

			if (not self.mousedown):
				self.last_mouse_pos = np.array((x, y), dtype=np.float32)
				self.mousedown = True

			else:
				self.mousedown = False

				self.renderer_view_out_of_date = True

		elif event == cv2.EVENT_LBUTTONDOWN:
			self.select_pos = np.array((x, y), dtype=np.float32)

	def on_mouse_scroll_event(self, x, y, dx, dy):

		self.scale = max(0, min(1, self.scale + dy * -SCROLL_SPEED))

		self.renderer_view_out_of_date = True
