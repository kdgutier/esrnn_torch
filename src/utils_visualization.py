import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_prediction(y, y_hat):
    """
    """
    pd.plotting.register_matplotlib_converters()

    plt.plot(y.ds, y.y, label = 'y')
    plt.plot(y_hat.ds, y_hat.y_hat, label='y_hat')
    plt.legend(loc='upper left')
    plt.show()
