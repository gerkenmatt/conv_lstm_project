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

def plot_inverse_transform(data, normalised_data):
    last_ob = data.total_data(False)[0][0]
    inv_trans = data.inverse_transform(last_ob, normalised_data)
    plt.plot_data(inv_trans, "inverse transform")

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
        normalise=False
    )
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=False
    )

    #inverse transform to verify it works
    if plotData:
        plot_inverse_transform(data, normalised_data)


    # Build the model(s)
    model = MyModel()
    if useFuncModel:
        model.build_functional_model(configs)
    if useSeqModel: 
        model.build_sequential_model(configs)
    
    # Train the model(s)
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
                # scaler,
                configs['data']['sequence_length'])
            plt.plot_results_multiple(
                raw_func_preds, 
                raw_train_data, 
                configs['data']['sequence_length'], 
                True,
                "Raw Train Predictions")
            return 

            # Predict on test data
            func_predictions_test = model.predict_sequences_multiple(
                x_test, 
                configs['data']['sequence_length'], 
                configs['data']['sequence_length'], 
                ModelType.FUNCTIONAL)
            raw_func_preds_test = data.inverse_transform_forecasts(
                normalised_data, 
                func_predictions_test, 
                # scaler,
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