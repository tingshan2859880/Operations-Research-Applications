import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from dir_config import DirConfig
from .data_preprocessing import *

path = DirConfig()


def plot_demand(filename, training: pd.Series, testing: pd.Series, pred: pd.DataFrame):
    """
    畫出銷售量趨勢圖
        Args:
            training: index 是時間的 training data
            testing: index 是時間的 testing data
            pred: index 是時間、每個 column 代表一種預測方法的預測值
        Returns: None
    """
    data = pd.DataFrame(index=pd.date_range(
        min(training.index), max(pred.index)))
    for p in pred.columns:
        data[p] = training.append(pred[p])
    data['origin'] = training.append(testing)

    sns_plot = sns.lineplot(data=data)
    sns_plot.figure.savefig(os.path.join(path.new_output_path,
                                         'fig', filename+".png"), dpi=200)
    plt.close()


def plot_dp_result(result, type='action', state=None, period=None, save=True, name='dp'):
    if period != None:
        plt.plot(result[period], '-o')
        plt.xlabel('state')
        if type == 'action':
            plt.ylabel("best discount rate")
        else:
            plt.ylabel("expected profit")
        plt.title("inventory level = " + str(state) + ", inventory cost = 0.1")
        # plt.show()
        plt.savefig(path.to_dp_fig_file(name+'-'+type+'_state_inv_'+str(state)+'.png'))
        plt.close('all')
    if state != None:
        plt.plot(result.loc[state], '-o')
        plt.xlabel('period')
        if type == 'action':
            plt.ylabel("best discount rate")
        else:
            plt.ylabel("expected profit")
        plt.title("period = " + str(period) + ", inventory cost = 0.1")
        # plt.show()
        plt.savefig(path.to_dp_fig_file(name+'-'+type+'_period_inv_'+str(state)+'.png'))
        plt.close('all')
    return


if __name__ == '__main__':
    flow_dic, trans_dic, trans_data = read_data()
    train, test = train_test_split(trans_data)
