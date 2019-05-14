import os
import json
import time
import math
from core.data_fetcher import DataFetcher
from core.mymodel import MyModel
from core.mymodel import ModelType
import numpy as np
from scipy.ndimage.interpolation import shift
import plot_utils as plt

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




    # Get data and normalise it first
    # data_total = data.get_total_data(
    #     seq_len=configs['data']['sequence_length'], 
    #     normalise=False)

    # print("data_total: ", str(data_total.shape))
    # i_split = int(len(data_total) * configs['data']['train_test_split'])
    # raw_train_data = data_total[:i_split]
    # last_ob = data_total[0][0]
    # scaler, normalised_data = data.transform_data(data_total)
    raw_train_data = data.train_data(norm=False)
    scaler = data.get_scaler()
    normalised_data = data.total_data(norm=True)



    if plotData:
        plot_data(normalised_data)

    # data.update_data(normalised_data)




    
    # Get training and test data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=False
    )
    if plotData:
        plot_data(y.flatten())

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=False
    )
    if plotData:
        plot_data(y_test.flatten())
    

    #inverse transform to verify it works
    # inv_trans = data.inverse_transform(last_ob, normalised_data, scaler)
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
            raw_func_preds = data.inverse_transform_forecasts(
                normalised_data, 
                func_predictions, 
                scaler,
                configs['data']['sequence_length'])
            plt.plot_results_multiple(
                raw_func_preds, 
                raw_train_data, 
                configs['data']['sequence_length'], 
                True,
                "Raw Train Predictions")

            # Predict on test data
            func_predictions_test = model.predict_sequences_multiple(
                x_test, 
                configs['data']['sequence_length'], 
                configs['data']['sequence_length'], 
                ModelType.FUNCTIONAL)
            raw_func_preds_test = data.inverse_transform_forecasts(
                normalised_data, 
                func_predictions_test, 
                scaler,
                configs['data']['sequence_length'])
            # plot_results_multiple_over_total(
            #     func_predictions, 
            #     data_total, 
            #     configs['data']['sequence_length'], 
            #     True, 
            #     0)
            plt.plot_results_multiple(
                func_predictions_test, 
                y_test, 
                configs['data']['sequence_length'], 
                True,
                "Test Predictions")
            return
    

        # Run predictions on Sequential model
        if useSeqModel:
            seq_predictions = model.predict_sequences_multiple(
                x, 
                configs['data']['sequence_length'], 
                configs['data']['sequence_length'], 
                ModelType.SEQUENTIAL, )
            plt.plot_results_multiple_over_total(
                seq_predictions, 
                data_total, 
                configs['data']['sequence_length'], 
                True, 
                0)




if __name__ == '__main__':
    main()