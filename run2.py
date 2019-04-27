import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.data_fetcher import DataFetcher
from core.mymodel import MyModel
from core.mymodel import ModelType
import numpy as np
from scipy.ndimage.interpolation import shift

useSeqModel = False
useFuncModel = True
visualizeConvolution = False
plotPredictions = True
plotData = False
evaluatePerformance = False

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple_over_total(predicted_data, true_data, prediction_len, normalised, start_index):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    ax.set_xticks(range(200), minor=True)
    print("start index: ", str(start_index))
    print("prediction_len: ")
    index = start_index
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, d in enumerate(predicted_data):
        data = d.copy()
        if not normalised:
            data[:] += true_data[index]
            index += prediction_len
        padding = [None for p in range(i * prediction_len + start_index)]
        plt.plot(padding + data)#, label='Prediction')
        plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len, normalised):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    index = 0

    ax.plot(true_data, label='True Data')
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

def plot_data(x):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(x)
    plt.show()

def plot_train_test_total(train, test, total, seq_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(train)
    print("test shape: ", str(test.shape))
    new_test = np.full((train.size + seq_len,1), np.NaN)
    print("new test shape: ", str(new_test.shape))

    new_test = np.concatenate([new_test, test])
    plt.plot(new_test)
    plt.plot(total)
    plt.show()

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataFetcher(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # Get data and normalise it first
    data_total = data.get_total_data(
        seq_len=configs['data']['sequence_length'], 
        normalise=False)
    i_split = int(len(data_total) * configs['data']['train_test_split'])
    raw_train_data = data_total[:i_split]
    last_ob = data_total[0][0]
    print("last_ob: ", str(last_ob))
    print("data_tatal shape: ", str(data_total.shape))
    scaler, normalised_data = data.transform_data(data_total)
    print("norm data shape: ", str(normalised_data.shape))
    if plotData:
        plot_data(normalised_data)

    data.update_data(normalised_data)
    
    # Get training and test data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=False
    )
    print("x shape: ", str(x.shape))
    print("y shape: ", str(y.shape))
    if plotData:
        plot_data(y.flatten())

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=False
    )
    print("x_test shape: ", str(x_test.shape))
    print("y_test shape: ", str(y_test.shape))
    if plotData:
        plot_data(y_test.flatten())
    
    #inverse transform to verify it works
    inv_trans = data.inverse_transform(last_ob, normalised_data, scaler)
    if plotData:
        plot_data(inv_trans)


    # Build the model(s)
    model = MyModel()
    if useFuncModel:
        model.build_functional_model(configs)
    if useSeqModel: 
        model.build_sequential_model(configs)

    # x, y = data.get_train_data(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']
    # )

    if plotData: 
        print("y.shape: ", str(y.shape))
        print("y_test.shape: ", str(y_test.shape))
        print("data_total.shape: ", str(data_total.shape))
        plot_data(data_total)
        # plot_train_test_total(y, y_test, data_total + 1, configs['data']['sequence_length'])

    
    # in-memory training
    if useFuncModel:
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir'],
            modelType=ModelType.FUNCTIONAL
        )
    if useSeqModel: 
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir'],
            modelType=ModelType.SEQUENTIAL
        )    
 
    # Plot predictions on each of the models
    if plotPredictions:
        if useFuncModel:
            # Run predictions on Functional model (with conv layers)
            func_predictions = model.predict_sequences_multiple(
                x, 
                configs['data']['sequence_length'], 
                configs['data']['sequence_length'], 
                ModelType.FUNCTIONAL)
            # print("func_pred shape: ", str(np.array(func_predictions).shape))
            raw_func_preds = data.inverse_transform_forecasts(
                normalised_data, 
                func_predictions, 
                scaler,
                configs['data']['sequence_length'])
            # print("raw_func_preds.shape: ", str(raw_func_preds.shape))
            print("normalised data shape: ", str(normalised_data.shape))
            plot_results_multiple(
                raw_func_preds, 
                raw_train_data, 
                configs['data']['sequence_length'], 
                False)
            return 

            # Predict on test data
            func_predictions_test = model.predict_sequences_multiple(
                x_test, 
                configs['data']['sequence_length'], 
                configs['data']['sequence_length'], 
                ModelType.FUNCTIONAL)
            plot_results_multiple_over_total(
                func_predictions, 
                data_total, 
                configs['data']['sequence_length'], 
                True, 
                0)
            plot_results_multiple(
                func_predictions_test, 
                y_test, 
                configs['data']['sequence_length'], 
                True)
    

        # Run predictions on Sequential model
        if useSeqModel:
            seq_predictions = model.predict_sequences_multiple(
                x, 
                configs['data']['sequence_length'], 
                configs['data']['sequence_length'], 
                ModelType.SEQUENTIAL, )
            plot_results_multiple_over_total(
                seq_predictions, 
                data_total, 
                configs['data']['sequence_length'], 
                True, 
                0)




if __name__ == '__main__':
    main()