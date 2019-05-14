
import matplotlib.pyplot as plt


def plot_results(predicted_data, true_data):
    """plot prediction over true data"""

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple_over_total(predicted_data, true_data, prediction_len, normalised, start_index):
    """plot the predictions over the entire dataset (train + test)"""

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    ax.set_xticks(range(200), minor=True)
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

def plot_results_multiple(predicted_data, true_data, prediction_len, normalised, title):
    """plot multiple predictions over true data"""

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    index = 0

    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, d in enumerate(predicted_data):
        data = d.copy()
        if not normalised:
            print("NORMALIZING")
            data[:] += true_data[index]
            index += prediction_len
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data)#, label='Prediction')
        plt.legend()
        plt.title(title)
    plt.show()

def plot_data(x):
    """plot line"""
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(x)
    plt.show()

def plot_train_test_total(train, test, total, seq_len):
    """for debugging purposes: plot only the test and train data, no predictions"""

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(train)
    new_test = np.full((train.size + seq_len,1), np.NaN)

    new_test = np.concatenate([new_test, test])
    plt.plot(new_test)
    plt.plot(total)
    plt.show()
