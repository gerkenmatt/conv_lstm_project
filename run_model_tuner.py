import os
import json
import time
import math
from core.data_fetcher import DataFetcher
from core.mymodel import MyModel
from core.mymodel import ModelType
import numpy as np
from scipy.ndimage.interpolation import shift
import core.plot_utils as plt

useSeqModel = False
useFuncModel = True
visualizeConvolution = False
plotPredictions = True
plotData = False
evaluatePerformance = False


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataFetcher(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # Get data for plotting later
    raw_train_data = data.train_data(norm=False)
    normalised_data = data.total_data(norm=True)

    # Get training and test data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=True, 
        shuffle=False
    )
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=True
    )
    x_shuffled, y_shuffled = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=True, 
        shuffle=True
    )
    # plt.plot_data(y, "train")

    #inverse transform to verify it works
    if plotData:
        plt.plot_inverse_transform(data, normalised_data)


    # Build the model(s)
    model = MyModel()
    model.build_functional_model(configs, configs['func_model']['kernel'])
    
    epochs = [5, 10, 5]#[20, 30, 40, 50, 60]
    kernel_sizes = [1, 2, 4, 6, 8, 10, 15]
    error_scores = list()
    repeats = 2

    for kernel in kernel_sizes:
        errs = list()
        for i in range(repeats):

            #rebuild new model 
            model.build_functional_model(configs, kernel)
            
            # train the model
            model.train(
                x_shuffled,
                y_shuffled,
                epochs = configs['training']['epochs'],
                batch_size = configs['training']['batch_size'],
                save_dir = configs['model']['save_dir'],
                modelType=ModelType.FUNCTIONAL
            )
            # evaluate performance, save metrics
            performance = model.eval(
                x=x,
                y=y, 
                batch_size=configs['training']['batch_size'], 
                modelType=ModelType.FUNCTIONAL
            )
            print("PERFORMANCE: ", str(performance))
            print("")
            errs.append(performance)
        error_scores.append(errs)



    for i in range(len(kernel_sizes)):
        print("kernel_size: ", str(kernel_sizes[i]))
        for j in range(repeats):
            print("---", str(error_scores[i][j]))
    # print("error scores: ", str(error_scores))
    return 

    


    # Train the model(s)
    if useFuncModel:
        model.train(
            x_shuffled,
            y_shuffled,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir'],
            modelType=ModelType.FUNCTIONAL
        )
    if useSeqModel: 
        model.train(
            x_shuffled,
            y_shuffled,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir'],
            modelType=ModelType.SEQUENTIAL
        )    
 
    # Plot predictions on each of the models
    if plotPredictions:
        if useFuncModel:
            plot_predictions(model, data, ModelType.FUNCTIONAL, configs, x, normalised_data, raw_train_data, x_test, y_test, y)

        # Run predictions on Sequential model
        if useSeqModel:
            plot_predictions(model, data, ModelType.SEQUENTIAL, configs, x, normalised_data, raw_train_data, x_test, y_test, y)
            



if __name__ == '__main__':
    main()