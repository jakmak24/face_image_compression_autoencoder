import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
	def __init__(self, cudaD = False):
		super(AutoEncoder, self).__init__()

		self.encoded = None
		self.cudaD = cudaD
		# ENCODER

		# 64x128x128
		self.e_conv_1 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
			nn.LeakyReLU()
		)

		# 128x64x64
		self.e_conv_2 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
			nn.LeakyReLU()
		)

		# 128x64x64
		self.e_block_1 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.e_pool_1 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		)

		# 128x32x32
		self.e_block_2 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.e_block_3 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 16x16x16
		self.e_conv_3 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(5, 5), stride=(2, 2)),
			nn.Tanh()
		)

		# DECODER

		# 128x32x32
		self.d_up_conv_1 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
		)

		# 128x32x32
		self.d_block_1 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x64x64
		self.d_up_1 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest')
		)

		# 128x64x64
		self.d_block_2 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x64x64
		self.d_block_3 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 256x128x128
		self.d_up_conv_2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
		)

		# 3x256x256
		self.d_up_conv_3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
			nn.LeakyReLU(),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2)),
			nn.Tanh()
		)

	def forward(self, x):
		# ENCODE
		ec1 = self.e_conv_1(x)
		ec2 = self.e_conv_2(ec1)
		eblock1 = self.e_block_1(ec2)
		eblock1 = self.e_pool_1(ec2 + eblock1)
		eblock2 = self.e_block_2(eblock1) + eblock1
		eblock3 = self.e_block_3(eblock2) + eblock2
		ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

		# stochastic binarization
		with torch.no_grad():
			if self.cudaD:
				rand = torch.rand(ec3.shape).cuda()
			else:
				rand = torch.rand(ec3.shape)
			prob = (1 + ec3) / 2
			
			if self.cudaD:
				eps = torch.zeros(ec3.shape).cuda()
			else:
				eps = torch.zeros(ec3.shape)
			eps[rand <= prob] = (1 - ec3)[rand <= prob]
			eps[rand > prob] = (-ec3 - 1)[rand > prob]

		# encoded tensor
		self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

		return self.decode(self.encoded)

	def decode(self, enc):
		y = enc * 2.0 - 1  # (0|1) -> (-1, 1)

		uc1 = self.d_up_conv_1(y)
		dblock1 = self.d_block_1(uc1) + uc1
		dup1 = self.d_up_1(dblock1)
		dblock2 = self.d_block_2(dup1) + dup1
		dblock3 = self.d_block_3(dblock2) + dblock2
		uc2 = self.d_up_conv_2(dblock3)
		dec = self.d_up_conv_3(uc2)

		return dec
