#!/usr/bin/env python

import sys
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import dill
from time import time
import seaborn as sns
import argparse

# dataloading apparatus
from dataset_TPC import dataset_TPC3d
from network_design import network_design
from visualize import plot_losses, plot_histograms, plot_histogram_2d, plot_errors, plot_mse, visualize_2d, visualize_3d

import torch
import torch.nn as nn
torch.cuda.is_available()

# Load model
model_path = '/sdcc/u/yhuang2/PROJs/EICLDRD/models'
assert Path(model_path).exists()
if model_path not in sys.path:
	sys.path.append(model_path)
from model_residual import CNNAE


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--layer_group', type=str, choices=['inner', 'middle', 'outer'], help='layer group')
	parser.add_argument('-c', '--oneOneConvCsz', type=int, default=None, help='number of channels in the one-by-one convolutional layer that ends the encoder')
	parser.add_argument('-r', '--use_residual', type=bool, default=False, help='whether to use residual blocks')
	parser.add_argument('-e', '--num_epochs', type=int, default=1000, help='number of epochs')
	parser.add_argument('-M', '--minimum_learning_rate', type=float, default=.0001, help='minimum learning rate')
	parser.add_argument('-S', '--starting_learning_rate', type=float, default=.01, help='staring learningrate')
	parser.add_argument('-R', '--learning_rate_decay_rate', type=float, default=.95, help='learning rate decay rate every 20 epochs')
	parser.add_argument('-E', '--train_valid_test', nargs=3, default=[960, 320, 320], help='number of train, validation, and test examples')
	parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('-d', '--description', type=str, required=True, help='decription of the run')
	parser.add_argument('-p', '--weight_pow', type=float, default=.1, help='the weight of voxel is ADC^pow')
	parser.add_argument('-g', '--focal_loss_gamma', type=float, default=2., help='the gamma value for the focal classification loss')
	args = parser.parse_args()


	# ========================================================== Parameters ========================================================== 
	residual, oneOneConvCsz = args.use_residual, args.oneOneConvCsz

	csz = [8, 16, 32, 32]
	rcsz = [oneOneConvCsz, 8, 4, 2]

	shape_map = {
		'inner': [96, 249, 16],
		'middle': [128, 249, 16],
		'outer': [192, 249, 16]
	}
	filter_stride_map = {
		'inner': {
			'filters': [[4, 5, 3], [3, 3, 3], [3, 3, 3], [3, 4, 3]],
			'strides': [[2, 2, 1], [2, 2, 1], [2, 2, 1], [1, 2, 1]]
		},
		'middle': {
			'filters': [[4, 5, 3], [3, 3, 3], [3, 3, 3], [3, 4, 3]],
			'strides': [[2, 2, 1], [2, 2, 1], [2, 2, 1], [1, 2, 1]]
		},
		'outer': {
			'filters': [[4, 5, 3], [3, 3, 3], [3, 3, 3], [3, 4, 3]],
			'strides': [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
		}
	}

	input_shape = shape_map[args.layer_group]
	filters = filter_stride_map[args.layer_group]['filters'] 
	strides = filter_stride_map[args.layer_group]['strides']
	output_paddings = network_design().get_output_padding(input_shape, filters, strides)


	# ========================================================== Trial ========================================================== 
	model = CNNAE(csz, rcsz, filters, strides, output_paddings, residual=residual, evaluation_mode=False, oneOneConvCsz=oneOneConvCsz)
	model.trial(input_shape)


	# ========================================================== Load Data ========================================================== 
	split_path = Path(f'/hpcgpfs01/scratch/yhuang2/TPC/highest_split_3d/{args.layer_group}/')
	framedata_path = Path(f'/hpcgpfs01/scratch/yhuang2/TPC/highest_framedata_3d/{args.layer_group}/')
	# we have maximumally 3200 train, 400 validation, and 400 test.
	# We can load a smaller portions of them (the list maximum) to make training faster.
	ds = dataset_TPC3d(split_path, framedata_path, batch_size=args.batch_size, maximum=args.train_valid_test)
	train_loader, valid_loader, test_loader = ds.get_splits()
	print(f'Number of batches:\n\ttrain = {len(train_loader)}\n\tvalidation = {len(valid_loader)}\n\ttest = {len(test_loader)}')


	# ========================================================== Train Parameters ========================================================== 
	model = CNNAE(csz, rcsz, filters, strides, output_paddings, evaluation_mode=False, residual=residual, oneOneConvCsz=oneOneConvCsz)
	model = model.cuda()
	optimizer = torch.optim.AdamW(model.parameters())

	lr_initial, ratio_initial = args.starting_learning_rate, 20000
	ratio, epochs = ratio_initial, args.num_epochs

	def get_lr(epoch, lr, rate=args.learning_rate_decay_rate, every_num_epochs=20, minimum_lr=args.minimum_learning_rate):
		lr = lr * (rate ** (epoch // every_num_epochs))
		if lr < minimum_lr:
			lr = minimum_lr
		return lr

	def transform(x):
		return torch.exp(x) * 6 + 64

	def loss_regression(input_regression, input_classification, target, transform, weight_pow=args.weight_pow, threshold=.5):
		input_regression_transformed = transform(input_regression)
		input_combined = input_regression_transformed * (input_classification > threshold)
		weight = torch.pow(target, weight_pow)
		return torch.sum(torch.square(input_combined - target) * weight) / torch.sum(weight)

	class SigmoidStep(nn.Module):
		"""
		SigmoidStep can also be used to implement a soft-classfication. 
		"""
		def __init__(self, mu, alpha):
			super(SigmoidStep, self).__init__()
			self.mu, self.alpha = mu, alpha

		def forward(self, x):
			y = self.alpha * (x - self.mu)
			return torch.sigmoid(y)

	def loss_classification(input_classification, target, gamma=args.focal_loss_gamma, eps=1e-8):
		target = torch.log2(target + 1)
		step = SigmoidStep(6, 20)
		soft_label_target = step(target)
		I = input_classification
		J = 1 - input_classification
		# guarding the log against too small values
		I, J = I + eps, J + eps

		focal_loss = soft_label_target * torch.log2(I) * torch.pow(J, gamma) + (1 - soft_label_target) * torch.log2(J) * torch.pow(I, gamma)	
		return -torch.mean(focal_loss)


	# ========================================================== Training ========================================================== 
	current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
	run_name = current_time + '_' + args.description
	print(f'Run name: {run_name}')
	path_prefix = f'/sdcc/u/yhuang2/PROJs/EICLDRD/results_3d/{run_name}/'
	if Path(path_prefix).exists():
		print(f'{path_prefix} exists， if you continue, the content in the folder will be removed.')
		overwritten = input('Continue?[Y/n]')
		if overwritten == 'Y':
			shutil.rmtree(path_prefix)
			Path(path_prefix).mkdir(parents=True)
		else:
			exit(1)
	else:
		Path(path_prefix).mkdir(parents=True)

	train_losses, valid_losses = [], []
	time0 = time()
	for e in range(epochs):
		# Adjust learning rate
		lr = get_lr(e, lr_initial)
		for g in optimizer.param_groups:
			g['lr'] = lr

		# training
		for x in train_loader:
			x = x.cuda()
			y_c, y_r = model(x)
			loss_c = loss_classification(y_c, x)
			loss_r = loss_regression(y_r, y_c, x, transform)
			loss = loss_r +  ratio * loss_c
			train_losses.append([loss_c.item(), loss_r.item(), loss.item()])

			# back-propagate
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# validation
		for x in valid_loader:
			x = x.cuda()
			y_c, y_r = model(x)		
			loss_c = loss_classification(y_c, x)
			loss_r = loss_regression(y_r, y_c, x, transform)
			loss = loss_r +  ratio * loss_c
			valid_losses.append([loss_c.item(), loss_r.item(), loss.item()])

		train_loss_avg = np.mean(train_losses[-len(train_loader):], axis=0)
		valid_loss_avg = np.mean(valid_losses[-len(valid_loader):], axis=0)

		print(f'Epoch {e}: {lr:.6f}, {ratio:.6f}')
		print(f'\ttrain loss = {train_loss_avg[0]: .6f}, {train_loss_avg[1]: .6f}, {train_loss_avg[2]: .6f}')
		print(f'\tvalid loss = {valid_loss_avg[0]: .6f}, {valid_loss_avg[1]: .6f}, {valid_loss_avg[2]: .6f}\n')

		ratio = train_loss_avg[1] / train_loss_avg[0]
	time1 = time()
	tpe = (time1 - time0) / epochs
	print(f'average time per epoch {tpe:.6f} seconds')

	# ========================================================== Evaluation ==========================================================
	model_state_dict = model.state_dict()
	## Evaluation with high-precision (float32) data
	model = CNNAE(csz, rcsz, filters, strides, output_paddings, residual=residual, evaluation_mode=False, oneOneConvCsz=oneOneConvCsz)
	model = model.cuda()
	model.load_state_dict(model_state_dict)
	model.eval()

	test_losses_float32 = []
	for x in test_loader:
		x = x.cuda()
		y_c, y_r = model(x)
		loss_c = loss_classification(y_c, x)
		loss_r = loss_regression(y_r, y_c, x, transform)
		test_losses_float32.append([loss_c.item(), loss_r.item()])

	tl_mean_float32, tl_std_float32 = np.mean(test_losses_float32, axis=0), np.std(test_losses_float32, axis=0)
	print(f'Test loss with float32:')
	print(f'\tmean = {tl_mean_float32[0]:.6f}, {tl_mean_float32[1]:.6f}')
	print(f'\tstd = {tl_std_float32[0]:.6f}, {tl_std_float32[1]:.6f}')

	## Evaluation with low-precision (float16) data
	model = CNNAE(csz, rcsz, filters, strides, output_paddings, residual=residual, evaluation_mode=True, oneOneConvCsz=oneOneConvCsz)
	model = model.cuda()
	model.load_state_dict(model_state_dict)
	model.eval()

	test_losses_float16 = []
	for x in test_loader:
		x = x.cuda()
		y_c, y_r = model(x)
		y_c, y_r = y_c.type(torch.float32), y_r.type(torch.float32)
		# Guarding values too close to 0 or 1 for the logarithmics
		y_c[y_c == 0] = 1e-6
		y_c[y_c == 1] = 1 - 1e-6

		loss_c = loss_classification(y_c, x)
		loss_r = loss_regression(y_r, y_c, x, transform)
		test_losses_float16.append([loss_c.item(), loss_r.item()])

	tl_mean_float16, tl_std_float16 = np.mean(test_losses_float16, axis=0), np.std(test_losses_float16, axis=0)
	print(f'Test loss with float16:')
	print(f'\tmean = {tl_mean_float16[0]:.6f}, {tl_mean_float16[1]:.6f}')
	print(f'\tstd = {tl_std_float16[0]:.6f}, {tl_std_float16[1]:.6f}')

	save_fname = f'{path_prefix}/loss.png'
	plot_losses(train_losses, valid_losses, test_loss=tl_mean_float32, save_fname=save_fname)


	# ========================================================== Save ========================================================== 
	model_run_dict = {
	   'serialized_model': str(model), # In case you loss the model, you can still type it in 
	   'model_state_dict': model.state_dict(),
	   'optimizer_state_dict': optimizer.state_dict(),
	   'epoch': args.num_epochs,
	   'batch_size': args.batch_size,
		# use dill.loads() to load the function
	   'regression_transform': dill.dumps(transform),
	   'loss_classification': dill.dumps(loss_classification), 
	   'loss_regression': dill.dumps(loss_regression),
	   'learning_rate_initial': lr_initial, 
	   'learning_rate_adaptive': dill.dumps(get_lr),
	   'clf_regr_loss_ratio_initial': ratio_initial,
	   'clf_regr_loss_ratio_adaptive': 'error balanced',
	   # Existing run results:
	   'train_losses': train_losses,
	   'validation_losses': valid_losses,
	   'test_losses_float32': test_losses_float32,
	   'test_losses_float16': test_losses_float16,
	   'run_name': run_name
	}

	save_fname = f'{path_prefix}/model_run_dict.pt'
	torch.save(model_run_dict, save_fname)
	# Sanity check
	# Load model
	model = CNNAE(csz, rcsz, filters, strides, output_paddings, residual=residual, evaluation_mode=False, oneOneConvCsz=oneOneConvCsz)
	model = model.cuda()
	checkpoint = torch.load(save_fname)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	# Load functions
	transform = dill.loads(checkpoint['regression_transform'])
	loss_classification = dill.loads(checkpoint['loss_classification'])
	loss_regression = dill.loads(checkpoint['loss_regression'])
	# Test run
	test_run_input = next(iter(test_loader))
	x = test_run_input.cuda()
	y_c, y_r = model(x)
	loss_c = loss_classification(y_c, x)
	loss_r = loss_regression(y_r, y_c, x, transform)
	print('================== Test run successful ==================')
	print(f'Test run loss = {loss_c.item(): .3f}, {loss_r.item(): .3f}')
	tl_mean = np.mean(checkpoint['test_losses_float32'], axis=0)
	print(f'As a reference, existing test losses mean = {tl_mean[0]:.3f}, {tl_mean[1]:.3f}')
	print('=========================================================')


	# ========================================================== Metrics ========================================================== 
	model = CNNAE(csz, rcsz, filters, strides, output_paddings, residual=residual, evaluation_mode=False, oneOneConvCsz=oneOneConvCsz)
	model = model.cuda()
	checkpoint = torch.load(f'{path_prefix}/model_run_dict.pt')
	model_state_dict = checkpoint['model_state_dict']
	model.load_state_dict(model_state_dict)
	model_str = model.eval()

	x = next(iter(train_loader)).cuda()
	x_log = torch.log2(x + 1).cpu().detach().numpy().flatten()
	y_c, y_r = model(x)

	## Classification Error rates
	save_fname = f'{path_prefix}/error.png'
	plot_errors(x, y_c, save_fname=save_fname)

	## Mean squared error
	save_fname = f'{path_prefix}/mse.png'
	plot_mse(x, y_r, y_c, transform, save_fname=save_fname)

	## 2d histogram, histogram, and visual comparison
	thresholds = [.35, .40, .45, .50, .55, .60]
	for threshold in thresholds:
		threshold_str = f'threshold-{int(threshold * 100)}'
		print(threshold_str)

		y = transform(y_r) * (y_c > threshold)

		# 2d histogram
		# in log-scaled and with only the true postive part
		y_log = torch.log2(y + 1).cpu().detach().numpy().flatten()
		X_log, Y_log = y_log[(x_log > 0) & (y_log > 0)], x_log[(x_log > 0) & (y_log > 0)]

		try:
			save_fname = f'{path_prefix}/{threshold_str}_histogram-2d.png'
			plot_histogram_2d(X_log, Y_log, save_fname=save_fname)
		except:
			print('histogram_2d error')

		# Histograms
		X = x.cpu().detach().numpy().flatten()
		Y_c = y_c.cpu().detach().numpy().flatten()
		Y = y.cpu().detach().numpy().flatten()

		
		save_fname = f'{path_prefix}/{threshold_str}_histogram.png'
		plot_histograms(X, Y_c, Y, save_fname=save_fname)
		
		# 2d and 3d visual comparison
		X_frame, Y_frame = x[0].squeeze(), y[0].squeeze()

	
		save_fname = f'{path_prefix}/{threshold_str}_sections-LA.png'
		visualize_2d(X_frame, Y_frame, frame_axis=1, max_frames=3, figure_width=15, cmap='viridis', save_fname=save_fname)
		
		save_fname = f'{path_prefix}/{threshold_str}_sections-AZ.png'
		visualize_2d(X_frame, Y_frame, frame_axis=2, max_frames=2, figure_width=7.5, cmap='viridis', save_fname=save_fname)

		save_fname = f'{path_prefix}/{threshold_str}_sections-LZ.png'
		visualize_2d(X_frame, Y_frame, frame_axis=0, max_frames=3, figure_width=15, cmap='viridis', save_fname=save_fname)

		save_fname = f'{path_prefix}/{threshold_str}_3d.png'
		visualize_3d(X_frame, Y_frame, figsize=(20, 10), permute=[1, 0, 2], save_fname=save_fname)
