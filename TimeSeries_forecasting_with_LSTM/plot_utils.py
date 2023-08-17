import numpy as np
import matplotlib.pyplot as plt



def plot_losses(history) -> None:
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    return


def predicted_plots(train_predict: np.array, test_predict: np.array, data: np.array, time_steps: int) -> None:
    # train predictions
    train_predictions = np.empty_like(data)
    train_predictions[:,:] = np.nan
    train_predictions[time_steps:len(train_predict)+time_steps, :] = train_predict
    # test predictions
    test_predictions = np.empty_like(data)
    test_predictions[:,:] = np.nan
    test_predictions[len(train_predict)+(time_steps*2)+1:len(data)-1, :] = test_predict
    # plot groundtruth and predictions
    plt.figure(figsize=(14,8))
    plt.plot(data, label='ground_truth')
    plt.plot(train_predictions, label='train')
    plt.plot(test_predictions, label='test')
    plt.legend()
    plt.show()


def plot_30_days_predict(data: np.array, predictions:list, scaler_model, offset=200) -> None:
    op_predictions = scaler_model.inverse_transform(predictions)
    data_updated = data.tolist()
    data_updated.extend(op_predictions)
    plt.figure(figsize=(12,8))
    plt.plot(data_updated[-offset:])
    plt.axvline(offset-30, color='red', lw=4, ls=':')

    final_predictions = np.empty((offset,1))
    final_predictions[:,:] = np.nan
    final_predictions[offset-30:, :] = op_predictions

    plt.plot(final_predictions, lw=4, color='green')
    plt.tight_layout()
    plt.show()
    return