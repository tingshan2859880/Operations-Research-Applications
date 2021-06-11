import pandas as pd
import numpy as np
import os
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import mse
from matplotlib import pyplot as plt

from .data_preprocessing import *


class TimeSeries:
    def __init__(self, data):
        """
        Args:
            data: index 為時間、只有一個 column （需求量或瀏覽數）的資料集
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


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


class LinearModel:
    def __init__(self, data):
        """
        Args:
            data: 包含日期、需求量與瀏覽數三個 columns 的資料集
        """
        self.data = data
        return

    def fit_traffic(self, holiday: list):
        """
        建立預測日瀏覽數的模型
            Args:
                channel: 單一通路名稱
                holiday: 一個記錄特殊節慶的list
            Returns: 迴歸模型
        """
        # print("____clean traffic file")
        traffic = self.data.copy()
        # traffic['seq_no'] = traffic.index + 1

        year = traffic['日期'].dt.year.unique()
        holiday_date = []
        for h in holiday:
            for y in year:
                holiday_date.append(
                    datetime.strptime(str(y)+'/'+h, '%Y/%m/%d'))
        traffic['前後7天內有特殊節日'] = 0
        traffic['temp'] = 0
        for d in holiday_date:
            traffic['temp'] = (traffic['日期'] - d).dt.days
            traffic['temp'] = abs(traffic['temp'])
            traffic.loc[(traffic['temp'] <= 7), '前後7天內有特殊節日'] = 1
        traffic.drop(columns=['日期', 'temp'], inplace=True)

        # 清理資料
        traffic.dropna(inplace=True)
        traffic['瀏覽數'] = traffic['瀏覽數'].astype(int)
        traffic = traffic[['瀏覽數', 'Seq_no', '前後7天內有特殊節日']]

        cols = list(traffic.columns)
        cols.remove('瀏覽數')

        # print("___build traffic model")
        all_columns = "+".join(cols)
        mod = smf.ols(formula="瀏覽數~" + all_columns, data=traffic)
        res = mod.fit()

        self.traffic_model = res

        return res

    def predict_traffic(self, model, start_date: datetime, end_date: datetime, holiday: list) -> pd.DataFrame:
        """
        預測每日瀏覽數
            Args:
                model: 預測模型
                start_date: 預測瀏覽數起始日期
                end_date: 預測瀏覽數結束日期
                holiday: 
            Returns: 包含日期與預測瀏覽數兩個欄位的 dataframe
        """
        new_data = pd.DataFrame()
        # print(pd.date_range(start_date, end_date))
        new_data['日期'] = pd.date_range(start_date, end_date)

        new_data['seq_no'] = range(
            max(self.data.index) + 2, max(self.data.index) + 2 + len(new_data['日期']))
        year = new_data['日期'].dt.year.unique()
        holiday_date = []
        for h in holiday:
            for y in year:
                holiday_date.append(
                    datetime.strptime(str(y)+'/'+h, '%Y/%m/%d'))
        new_data['前後7天內有特殊節日'] = 0
        new_data['temp'] = 0
        for d in holiday_date:
            new_data['temp'] = (new_data['日期'] - d).dt.days
            new_data['temp'] = abs(new_data['temp'])
            new_data.loc[(new_data['temp'] <= 7), '前後7天內有特殊節日'] = 1
        new_data.drop(columns=['日期', 'temp'], inplace=True)

        pred = pd.DataFrame()
        pred['日期'] = pd.date_range(start_date, end_date)
        pred['瀏覽數_pred'] = model.predict(new_data)

        return pred

    def fit_sales_daily(self, sales_data: pd.DataFrame, traffic_data: pd.DataFrame):
        """
        建立預測日銷售量的模型
            Args:
                sales_data: 銷售資料，包含日期、單價、數量三個 columns
                traffic_data: 瀏覽數資料
                channel: 單一通路名稱
            Returns: 迴歸模型
        """
        data = pd.merge(sales_data, traffic_data, left_on='單據日期',
                        right_on='日期').drop(columns='日期')
        # data = data[(np.abs(data['數量']-data['數量'].mean()) <= (3*data['數量'].std()))]
        # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
        # data = data[~(np.abs(data['數量']-data['數量'].mean()) > (3*data['數量'].std()))]
        data.dropna(inplace=True)
        data, _ = agg_weekly_data(data)
        data = data[['數量', '單價', '瀏覽數', 'week_day']]
        # data['seq_no'] = data.index + 1
        data['數量'] = data['數量'].astype(int)
        data['單價'] = data['單價'].astype(float)
        data['瀏覽數'] = data['瀏覽數'].astype(int)

        data['week_day'] = data['week_day'].astype(str)
        # print("____get dummies")
        data = pd.get_dummies(data, drop_first=True)

        cols = list(data.columns)
        cols.remove('數量')

        # print("___fit sales data")
        all_columns = "+".join(cols)
        mod = smf.ols(formula="數量~" + all_columns, data=data)
        res = mod.fit()

        self.sales_model = res

        return res

    def predict_sales_daily(self, price, model, sales_data: pd.DataFrame, traffic_prediction: pd.DataFrame,
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        預測日需求
            Args:
                price: 價格帶
                model: 迴歸模型
                sales_data: 以日為單位的銷售紀錄
                traffic_prediction: 以日為單位的未來瀏覽數預測
                start_date: 未來資料起始日
                end_date: 未來資料結束日
            Returns: 包含日期與預測銷售量兩個欄位的 dataframe
        """
        new_data = pd.DataFrame()
        new_data['單據日期'] = pd.date_range(start_date, end_date)
        new_data['單價'] = price
        new_data['瀏覽數'] = traffic_prediction['瀏覽數_pred']
        new_data, _ = agg_weekly_data(new_data)
        # new_data['seq_no'] = range((start_date - min(sales_data['單據日期'])).days + 1,
        #                            (start_date - min(sales_data['單據日期'])).days + len(new_data['單據日期']) + 1)
        new_data['week_day'] = new_data['week_day'].astype(str)
        new_data = pd.get_dummies(new_data, drop_first=True)
        new_data.drop(columns=['單據日期', 'week'], inplace=True)
        pred = pd.DataFrame()
        pred['日期'] = pd.date_range(start_date, end_date)
        pred['數量_pred'] = model.predict(new_data)
        return pred


def weighted_average(weights, data):
    """
    計算每一天加權平均後的預測值
        Args:
            weights: 1xn matrix，代表 n 個模型的權重
            data: nxk matrix，代表 n 個模型中各自有 k 個預測值
        Returns: 1xk matrix，代表加權後的 k 個預測結果
    """
    avg = np.dot(weights, data)
    return avg


def main():
    '''
    1. 拔掉 outlier 
    2. train-test
    3. cluster(目前資料怎麼做)
    4. data 要重新抓
    '''
    low, sales_data, sales_all = read_data()

    weekly_sales = {}
    for k, v in sales_data.items():
        sales_data[k] = fill_daily_na(
            agg_daily_data(sales_data[k]))  # 補沒有販售的時間點
        sales_data[k], weekly_sales[k] = agg_weekly_data(sales_data[k])  # 整理成週
        sales_data[k].set_index('單據日期', inplace=True)

    sales_all = fill_daily_na(agg_daily_data(sales_all))
    sales_all.set_index('單據日期', inplace=True)

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
