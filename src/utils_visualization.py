import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_prediction(y, y_hat, forecast=True):
    """
    """
    n_y = len(y)
    n_yhat = len(y_hat)
    ds_y = np.array(range(n_y))
    ds_yhat = np.array(range(n_y, n_y+n_yhat))

    plt.plot(ds_y, y, label = 'y')
    plt.plot(ds_yhat, y_hat, label='y_hat')
    plt.legend(loc='upper left')
    plt.show()
