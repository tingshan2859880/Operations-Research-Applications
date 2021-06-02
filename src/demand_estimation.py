import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import mse
from matplotlib import pyplot as plt

from data_preprocessing import *


class TimeSeries:
    def __init__(self, data):
        """
        Args:
            data: index 為時間、只有一個 column 的資料集
        """
        self.data = data
        return

    def ADF_test(self, data, alpha=0.05, find_d=False):
        """
        Args:
            data: 傳入要判定是否 stationary 的資料集
            alpha: 信心水準
            find_d: 是否為尋找 ARIMA 參數 d 的模式
        """
        if find_d:
            t = adfuller(data)
        else:
            t = adfuller(self.data)

        # H0: 這組 time series 是 non-stationary
        if t[1] >= alpha:
            if not find_d:
                self.type = 'non-stationary'  # 使用 ARIMA

            diff_data = self.data.copy().diff().dropna()  # 對原始資料做一階微分
            return 1+self.ADF_test(diff_data, find_d=True)

        else:
            if not find_d:
                self.type = 'stationary'  # 使用 ARMA
            return 0

    def ACF_PACF(self):
        """
        透過畫圖觀察，幫助找到合適的 p 和 q 範圍
        """

        lag_acf = acf(self.data, nlags=20)
        lag_pacf = pacf(self.data, nlags=20, method='ols')

        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        sm.graphics.tsa.plot_acf(self.data, lags=100, ax=axes[0])
        sm.graphics.tsa.plot_pacf(self.data, lags=100, ax=axes[1])
        plt.show()
        return

    def fit(self, p_range, d_range, q_range):
        """
        Args:
            p_range: 要進行測試的參數 p 範圍
            d_range: 要進行測試的參數 d 範圍
            q_range: 要進行測試的參數 q 範圍
        Returns:
            best_model: 最佳 ARMA 模型
            best_aic: 最佳 ARMA 模型的 AIC
        """
        best_model = None
        best_aic = 100000000000000000
        best_pdq_set = None
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    model = ARIMA(
                        self.data, order=(p, d, q)).fit()  # 如果是 ARMA 只要 d=0 即可

                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                        best_pdq_set = (p, d, q)
        self.model = best_model
        self.aic = best_aic
        self.best_param_set = best_pdq_set
        return best_model, best_aic, best_pdq_set

    def predict(self, start_date, end_date):
        """
        輸入未來一段時間區間做預測
        Args:
            start_date: 起始日
            end_date: 結束日
        Returns: 預測結果
        """
        pred = self.model.predict(start_date, end_date)
        return pred

    def MSE(self, pred, real):
        """
        輸入預測值與實際值計算 MSE
        Args:
            pred: 預測值
            real: 實際值
        Returns: MSE
        """
        mean_squared_error = mse(pred, real)
        return mean_squared_error

    def box_pierce_test(self, alpha=0.05):
        # H0: 殘差是獨立的，模型有效
        lbq = sm.stats.acorr_ljungbox(self.model.resid, lags=[
                                      10], return_df=True, boxpierce=True)
        if lbq.loc[10, 'bp_pvalue'] <= alpha:
            return False
        else:
            return True


def main():
    low, sales_data, sales_all = read_data()

    weekly_sales = {}
    for k, v in sales_data.items():
        sales_data[k] = fill_daily_na(agg_daily_data(sales_data[k]))
        sales_data[k], weekly_sales[k] = agg_weekly_data(sales_data[k])
        sales_data[k].set_index('SlipDate', inplace=True)

    sales_all = fill_daily_na(agg_daily_data(sales_all))
    sales_all.set_index('SlipDate', inplace=True)

    ts_test = TimeSeries(sales_all['數量'])
    ts_test.ACF_PACF()
    d = ts_test.ADF_test(None)

    p_range = list(range(2, 6))
    d_range = [0]
    q_range = list(range(2, 6))
    model, aic, params = ts_test.fit(p_range, d_range, q_range)
    print("model is good (not lack of fit)?", ts_test.box_pierce_test())

    return


if __name__ == '__main__':
    main()
