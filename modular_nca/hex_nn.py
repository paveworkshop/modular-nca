import torch
import torch.nn as nn

activation = torch.nn.functional.gelu  #torch.tanh #torch.nn.functional.relu #

class HexNN(nn.Module):

	def __init__(self, input_size, output_size, hidden_layer_sizes):
		
		super(HexNN, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_layer_sizes = hidden_layer_sizes
		
		self.encoder = nn.Linear(self.input_size, self.hidden_layer_sizes[0])
		self.hidden = [nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]) for i in range(len(self.hidden_layer_sizes)-1)]
		self.decoder = nn.Linear(self.hidden_layer_sizes[-1], self.output_size)

		nn.init.zeros_(self.decoder.weight)
		nn.init.zeros_(self.decoder.bias)

	def forward(self, x):

		x = self.encoder(x)
		for h in self.hidden:
			x = h(x)
		x = activation(x)
		x = self.decoder(x)

		return x


