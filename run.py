__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.mymodel import MyModel


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_conv(predicted_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len, normalised):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    index = 0
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, d in enumerate(predicted_data):
        data = d.copy()
        if not normalised:
            data[:] += true_data[index]
            index += prediction_len
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data)#, label='Prediction')
        plt.legend()
    plt.show()

def build_conv_layer_model(conv_layer):
    model = Sequential()

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = MyModel()
    model.build_functional_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    '''
    # in-memory training
    model.train(
        x,
        y,
        epochs = configs['training']['epochs'],
        batch_size = configs['training']['batch_size'],
        save_dir = configs['model']['save_dir']
    )
    '''
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    
    #set up raw input for feeding into conv1d layer
    # print("x_test shape: ", str(x_test.shape))
    # # raw_input = one sequence length 
    # raw_input = x_test[1]
    # print("raw input shape: ", str(raw_input.shape))
    # print("")
    # conv_output = model.get_output_of_conv_layer(raw_input)
    # print("conv_output : ", str(conv_output))

    # see how accurate the predictions are on training data
    # visualize 
    print("*****x shape: ", str(x.shape))
    conv_predictions = model.conv_layer_analysis(x, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # print("AUX PREDICTIONS: ", str(conv_predictions[0][0][0].shape))
    # plot_results(conv_predictions[0][0][0], x[0][1])




    # train_predictions = model.predict_sequences_multiple(x, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # plot_results_multiple(train_predictions, y, configs['data']['sequence_length'], True)

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # # predictions = model.predict_point_by_point(x_test)

    # plot_results_multiple(predictions, data.data_test[:,0], configs['data']['sequence_length'], False)
    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'], True)
    # # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()