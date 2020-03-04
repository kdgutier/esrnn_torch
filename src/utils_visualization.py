import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns


def plot_prediction(y, y_hat):
  """
  y: pandas df
    panel with columns unique_id, ds, y
  y_hat: pandas df
    panel with columns unique_id, ds, y_hat
  """
  pd.plotting.register_matplotlib_converters()

  plt.plot(y.ds, y.y, label = 'y')
  plt.plot(y_hat.ds, y_hat.y_hat, label='y_hat')
  plt.legend(loc='upper left')
  plt.show()

def plot_distributions(distributions_dict, fig_title=None, xlabel=None):
  n_distributions = len(distributions_dict.keys())
  fig, ax = plt.subplots(1, figsize=(7, 5.5))
  plt.subplots_adjust(wspace=0.35)
  
  n_colors = len(distributions_dict.keys())
  colors = sns.color_palette("hls", n_colors)
  
  for idx, dist_name in enumerate(distributions_dict.keys()):
      train_dist_plot = sns.kdeplot(distributions_dict[dist_name],
                                    bw='silverman',
                                    label=dist_name,
                                    color=colors[idx])
      if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14)
      ax.set_ylabel('Density', fontsize=14)
      ax.set_title(fig_title, fontsize=15.5)
      ax.grid(True)
      ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
  
  fig.tight_layout()
  if fig_title is not None:
    fig_title = fig_title.replace(' ', '_')
    plot_file = "./results/plots/{}_distributions.png".format(fig_title)
    plt.savefig(plot_file, bbox_inches = "tight", dpi=300)
  plt.show()

def plot_cat_distributions(df, cat, var):
  unique_cats = df[cat].unique()
  cat_dict = {}
  for c in unique_cats:
      cat_dict[c] = df[df[cat]==c][var].values
  
  plot_distributions(cat_dict, xlabel=var)

def plot_single_cat_distributions(distributions_dict, ax, fig_title=None, xlabel=None):
    n_distributions = len(distributions_dict.keys())

    n_colors = len(distributions_dict.keys())
    colors = sns.color_palette("hls", n_colors)

    for idx, dist_name in enumerate(distributions_dict.keys()):
        train_dist_plot = sns.distplot(distributions_dict[dist_name],
                                      #bw='silverman',
                                      #kde=False,
                                      rug=True,
                                      label=dist_name,
                                      color=colors[idx],
                                      ax=ax)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(fig_title, fontsize=15.5)
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_grid_cat_distributions(df, cats, var):
    cols = int(np.ceil(len(cats)/2))
    fig, axs = plt.subplots(2, cols, figsize=(4*cols, 5.5))
    plt.subplots_adjust(wspace=0.95)
    plt.subplots_adjust(hspace=0.5)
    
    for idx, cat in enumerate(cats):
        unique_cats = df[cat].unique()
        cat_dict = {}
        for c in unique_cats:
            values = df[df[cat]==c][var].values
            values = values[~np.isnan(values)]
            if len(values)>0:
              cat_dict[c] = values
        
        row = int(np.round((idx/len(cats))+0.001, 0))
        col = idx % cols
        plot_single_cat_distributions(cat_dict, axs[row, col],
                                      fig_title=cat, xlabel=var)
    
    min_owa = math.floor(df.min_owa.min() * 1000) / 1000
    suptitle = var + ': ' + str(min_owa)
    fig.suptitle(suptitle, fontsize=18)
    plt.show()
