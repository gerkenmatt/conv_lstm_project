import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, LSTM, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
import matplotlib.pyplot as plt
from enum import Enum

class ModelType(Enum):
	FUNCTIONAL = 0
	SEQUENTIAL = 1


class MyModel():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.seq_model = Sequential()
		self.conv1d_input_shape = (None, 49, 2)

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.seq_model = load_model(filepath)

	def build_functional_model(self, configs):
		"""Build Functional keras model"""

		timer = Timer()
		timer.start()

		sequence_len = configs['data']['sequence_length'] - 1

		# input tensor
		inputs = Input(shape=(sequence_len,2))

		filter_num = 64
		# conv1d layer: feature extraction
		feat_extract = Conv1D(
			filters=filter_num, 
			kernel_size=5, 
			input_shape=(sequence_len, 2), 
			padding='same' )(inputs)

		dropout0 = Dropout(0.3)(feat_extract)
		# lstm network
		lstm1 = LSTM(100, input_shape=(sequence_len,filter_num), return_sequences=False)(dropout0)
		dropout1 = Dropout(0.2)(lstm1)
		# lstm2 = LSTM(100, return_sequences=True)(dropout1)
		# lstm3 = LSTM(100, return_sequences=False)(dropout1)
		# dropout2 = Dropout(0.2)(lstm3)
		predictions = Dense(1, activation='linear')(dropout1)

		# build the model
		self.func_model = Model(inputs=inputs,outputs=predictions)
		self.func_model.compile(optimizer='adam', loss='mse')

		# self.aux_model = Model(inputs=inputs, outputs=feat_extract)

		print("BUILT FUNCTIONAL MODEL: ")
		print(self.func_model.summary())

		plot_model(self.func_model, to_file='conv_lstm_plot.png')
		timer.stop()



	def build_sequential_model(self, configs):
		timer = Timer()
		timer.start()
		sequence_len = configs['data']['sequence_length'] 

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			filters = layer['filters'] if 'filters' in layer else None
			kernel_size = layer['kernel_size'] if 'kernel_size' in layer else None
			padding = layer['padding'] if 'padding' in layer else "same"

			if layer['type'] == 'dense':
				self.seq_model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.seq_model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.seq_model.add(Dropout(dropout_rate))
			if layer['type'] == 'conv1d':
				self.conv1d_input_shape = (None, sequence_len-1, input_dim)
				self.seq_model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(sequence_len-1, input_dim), padding=padding ))

		self.seq_model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		for layer in self.seq_model.layers:
			print("layer: ")
			print("   input shape: ", str(layer.input_shape))
			print("   output shape: ", str(layer.output_shape))
		timer.stop()


	def train(self, x, y, epochs, batch_size, save_dir, modelType):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			# EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		if modelType == ModelType.FUNCTIONAL: 
			self.func_model.fit(
				x,
				y,
				epochs=epochs,
				batch_size=batch_size,
				callbacks=callbacks
			)
			self.func_model.save(save_fname)
		elif modelType == ModelType.SEQUENTIAL: 
			self.seq_model.fit(
				x,
				y,
				epochs=epochs,
				batch_size=batch_size,
				callbacks=callbacks
			)
			self.seq_model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def eval(self, x, y, batch_size, modelType):
		timer = Timer()
		timer.start()
		print('[Model] Evaluation Started')
		
		perf = []
		if modelType == ModelType.FUNCTIONAL: 
			perf = self.func_model.evaluate(x, y)
			print("TEST PERF: ", str(perf))
		elif modelType == ModelType.SEQUENTIAL: 
			perf = self.seq_model.evaluate(x, y)

		print('[Model] Evaluation Completed')
		timer.stop()
		return perf

	def eval_generator(self, data_gen, batch_size,save_dir, modelType):
		timer = Timer()
		timer.start()
		print('[Model] Evaluation Started')
		
		perf = []
		if modelType == ModelType.FUNCTIONAL: 
			print("     EVALUATE FUNCTIONAL PERFORMANCE")
			perf = 	self.func_model.evaluate_generator(
						data_gen,
						workers=1,
						steps=100
					)
			print("***********PERFORMANCE: ", str(perf))
		elif modelType == ModelType.SEQUENTIAL: 
			print("     EVALUATE SEQUENTIAL PERFORMANCE")
			perf = 	self.seq_model.evaluate_generator(
						data_gen,
						workers=1,
						steps=100
					)
		print('[Model] Evaluation Completed. ')
		timer.stop()
		return perf

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, modelType):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		if modelType == ModelType.FUNCTIONAL: 
			self.func_model.fit_generator(
				data_gen,
				steps_per_epoch=steps_per_epoch,
				epochs=epochs,
				callbacks=callbacks,
				workers=1
			)
		elif modelType == ModelType.SEQUENTIAL: 
			self.seq_model.fit_generator(
				data_gen,
				steps_per_epoch=steps_per_epoch,
				epochs=epochs,
				callbacks=callbacks,
				workers=1
			)
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def predict_point_by_point(self, data, modelType):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		if modelType == ModelType.FUNCTIONAL:
			predicted = self.func_model.predict(data)
		elif modelType == ModelType.SEQUENTIAL: 
			predicted = self.seq_model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

	def conv_layer_analysis(self, data, window_size, prediction_len):
		"""Visualization of convolutional layers and filters"""
		print('[Model] Predicting Aux Sequences Multiple...')
		prediction_seqs = []
		print("********data length: ", str(len(data)))
		weights = self.aux_model.get_weights()
		print("WEIGHTS.shape: ", str(weights[0].shape))
		print("WEIGHTS: ", str(weights))
		w1 = weights[0][:,0,0]
		w2 = weights[0][:,0,1]
		w3 = weights[0][:,0,2]
		print("w: ", str(w1))
		fig = plt.figure(facecolor='white')
		ax = fig.add_subplot(111)
		ax.plot(w1, label='w')
		# plt.plot(w2, label="w2")
		# plt.plot(w3,label="w3")
		plt.legend()
		plt.show()

		sequence_i = 15
		curr_frame = data[sequence_i*prediction_len]
		# print("CURR FRAME SHAPE: ", str(curr_frame.shape))
		# print("***********curr_frame: ", str(curr_frame))
		x = curr_frame[:,0]
		# print("x: ", str(x))
		# print("X.shape: ", str(x.shape))

		output = self.aux_model.predict(curr_frame[newaxis,:,:])
		# print("OUTPUT SHAPE: ", str(output.shape))
		# print("OUTPUT: ", str(output))
		y1 = output[0,:,0]
		y2 = output[0,:,1]
		y3 = output[0,:,2]
		# print("y: ", str(y))
		# print("y.shape: ", str(y.shape))
		fig = plt.figure(facecolor='white')
		ax = fig.add_subplot(111)
		ax.plot(x, label='x')
		plt.plot(y1, label="y1")
		# plt.plot(y2,label="y2")
		# plt.plot(y3, label="y3")
		plt.legend()
		plt.show()

		return
		for i in range(int(len(data)/prediction_len)):

			curr_frame = data[i*prediction_len]
			# print("CURR FRAME SHAPE: ", str(curr_frame.shape))
			# print("***********curr_frame: ", str(curr_frame))
			x = curr_frame[:,0]
			# print("x: ", str(x))
			# print("X.shape: ", str(x.shape))

			output = self.aux_model.predict(curr_frame[newaxis,:,:])
			# print("OUTPUT SHAPE: ", str(output.shape))
			# print("OUTPUT: ", str(output))
			y1 = output[0,:,0]
			y2 = output[0,:,1]
			y3 = output[0,:,2]
			# print("y: ", str(y))
			# print("y.shape: ", str(y.shape))
			fig = plt.figure(facecolor='white')
			ax = fig.add_subplot(111)
			ax.plot(x, label='x')
			plt.plot(y1, label="y1")
			# plt.plot(y2,label="y2")
			# plt.plot(y3, label="y3")
			plt.legend()
			plt.show()


			return 

	def predict_sequences_multiple(self, data, window_size, prediction_len, modelType):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		# print("********data length: ", str(len(data)))
		# print("******data: ", str(data))
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			# print("***********curr_frame shape: ", str(curr_frame.shape))
			predicted = []
			for j in range(prediction_len):
				if modelType == ModelType.FUNCTIONAL: 
					pred = self.func_model.predict(curr_frame[newaxis,:,:])[0,0]
				elif modelType == ModelType.SEQUENTIAL: 
					pred = self.seq_model.predict(curr_frame[newaxis,:,:])[0,0]
				predicted.append(pred)
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			# print("******prediction list len: ", len(predicted))
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size, modelType):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			if modelType == ModelType.FUNCTIONAL: 
				predicted.append(self.func_model.predict(curr_frame[newaxis,:,:])[0,0])
			elif modelType == ModelType.SEQUENTIAL: 
				predicted.append(self.seq_model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted