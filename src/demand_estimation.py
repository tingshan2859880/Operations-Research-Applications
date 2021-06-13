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
from pmdarima.arima import auto_arima

from .data_preprocessing import *


def weighted_average(mse_list, data):
    """
    計算每一天加權平均後的預測值
        Args:
            mse_list: 1xn matrix，代表 n 個模型的mse
            data: nxk matrix，代表 n 個模型中各自有 k 個預測值
        Returns: 1xk matrix，代表加權後的 k 個預測結果
    """
    inverse_mse = []
    for i in mse_list:
        inverse_mse.append(1/i)
    weight = []
    for i in inverse_mse:
        weight.append(i/sum(inverse_mse))
    print("weight of prediction:", weight)
    avg = np.dot(weight, data)
    return avg


class TimeSeries:
    def __init__(self, data, no_outiler=True):
        """
        Args:
            data: index 為時間、只有一個 column （需求量或瀏覽數）的資料集
        """
        if no_outiler:
            self.data = pd.Series(remove_outlier(data), index=data.index)
        else:
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

    def fit(self, price=None, p_range=range(5), d_range=range(3), q_range=range(5),  method='auto', print_trace=False):
        """
        Args:
            p_range: 要進行測試的參數 p 範圍
            d_range: 要進行測試的參數 d 範圍
            q_range: 要進行測試的參數 q 範圍
            method:
        Returns:
            best_model: 最佳 ARMA 模型
            best_aic: 最佳 ARMA 模型的 AIC
            best_pdq_set: 最佳 (p, d, q) 組合
        """
        if method == 'auto':
            self.auto_arima = True
            self.normal_arima = False
            n_diffs = self.ADF_test(None)
            best_model = auto_arima(self.data, X=price, d=n_diffs, seasonal=False, stepwise=True, error_action="ignore", max_p=max(p_range), max_q=max(q_range),
                                    max_order=None, trace=print_trace)
            best_aic = best_model.aic()
            best_pdq_set = best_model.order
        if method == 'normal':
            self.auto_arima = False
            self.normal_arima = True
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

    def predict(self, price=None, start_date=None, end_date=None, period=None):
        """
        輸入未來一段時間區間做預測
        Args:
            start_date: 起始日
            end_date: 結束日
        Returns: 預測結果
        """
        if self.normal_arima:
            pred = self.model.predict(start_date, end_date)
        if self.auto_arima:
            pred = self.model.predict(
                n_periods=period, X=np.array([price]*period).reshape(-1, 1))
            pred = [max(0, x) for x in pred]
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
        if self.normal_arima:
            lbq = sm.stats.acorr_ljungbox(self.model.resid(), lags=[
                7], return_df=True, boxpierce=True)
        if self.auto_arima:
            lbq = sm.stats.acorr_ljungbox(self.model.resid(), lags=[
                7], return_df=True, boxpierce=True)
        if lbq.loc[7, 'bp_pvalue'] <= alpha:
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
        self.data = data.copy()
        self.data.reset_index(inplace=True)
        return

    def fit_traffic_lm(self, holiday=['12/31', '8/31']):
        """
        以 linear regression 建立預測日瀏覽數的模型
            Args:
                holiday: 一個記錄特殊節慶的list
            Returns: 迴歸模型
        """
        traffic = self.data.copy()

        year = traffic['單據日期'].dt.year.unique()
        holiday_date = []
        for h in holiday:
            for y in year:
                holiday_date.append(
                    datetime.strptime(str(y)+'/'+h, '%Y/%m/%d'))
        traffic['前後7天內有特殊節日'] = 0
        traffic['temp'] = 0
        for d in holiday_date:
            traffic['temp'] = (traffic['單據日期'] - d).dt.days
            traffic['temp'] = abs(traffic['temp'])
            traffic.loc[(traffic['temp'] <= 7), '前後7天內有特殊節日'] = 1
        traffic.drop(columns=['單據日期', 'temp'], inplace=True)

        # 清理資料
        traffic.dropna(inplace=True)
        traffic['瀏覽數'] = traffic['瀏覽數'].astype(int)
        traffic = traffic[['瀏覽數', 'seq_no', '前後7天內有特殊節日']]

        cols = list(traffic.columns)
        cols.remove('瀏覽數')

        # print("___build traffic model")
        all_columns = "+".join(cols)
        mod = smf.ols(formula="瀏覽數~" + all_columns, data=traffic)
        res = mod.fit()

        self.traffic_model_lm = res

        return res

    def fit_traffic_arima(self):
        """
        用 ARIMA 建立預測日瀏覽數的模型
            Returns: 迴歸模型
        """
        traffic = self.data.copy()['瀏覽數']

        res = TimeSeries(traffic)
        res.fit()
        self.traffic_model_arima = res

        return res

    def predict_traffic_lm(self, testing, start_date, end_date, holiday=['12/31', '8/31']) -> pd.DataFrame:
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
        new_data['單據日期'] = pd.date_range(start_date, end_date)

        new_data['seq_no'] = range(
            max(self.data.index) + 2, max(self.data.index) + 2 + len(new_data['單據日期']))
        year = new_data['單據日期'].dt.year.unique()
        holiday_date = []
        for h in holiday:
            for y in year:
                holiday_date.append(
                    datetime.strptime(str(y)+'/'+h, '%Y/%m/%d'))
        new_data['前後7天內有特殊節日'] = 0
        new_data['temp'] = 0
        for d in holiday_date:
            new_data['temp'] = (new_data['單據日期'] - d).dt.days
            new_data['temp'] = abs(new_data['temp'])
            new_data.loc[(new_data['temp'] <= 7), '前後7天內有特殊節日'] = 1
        new_data.drop(columns=['單據日期', 'temp'], inplace=True)

        pred = pd.DataFrame()
        pred['單據日期'] = pd.date_range(start_date, end_date)
        pred['瀏覽數_pred'] = self.traffic_model_lm.predict(new_data)
        mse_lm = mse(pred['瀏覽數_pred'][:len(testing)], testing)

        return pred, mse_lm

    def predict_traffic_arima(self, testing, start_date, end_date):
        if self.traffic_model_arima.normal_arima:
            pred = self.traffic_model_arima.predict(
                start_date=start_date, end_date=end_date)
        if self.traffic_model_arima.auto_arima:
            pred = self.traffic_model_arima.predict(
                period=(end_date-start_date).days+1)
        mse_arima = self.traffic_model_arima.MSE(pred[:len(testing)], testing)
        return pred, mse_arima

    def fit_predict_traffic(self, testing, start_date, end_date, holiday=['12/31', '8/31']):
        self.fit_traffic_lm()
        self.fit_traffic_arima()
        pred_lm, mse_lm = self.predict_traffic_lm(
            testing, start_date, end_date)
        pred_arima, mse_arima = self.predict_traffic_arima(
            testing, start_date, end_date)

        return weighted_average(np.array([mse_lm, mse_arima]), np.array([pred_lm['瀏覽數_pred'], pred_arima]))

    def fit_sales_daily(self):
        """
        建立預測日銷售量的模型
            Returns: 迴歸模型
        """
        sales_data = self.data.copy()
        sales_data = sales_data[['數量', '折數', '瀏覽數', 'week_day']]
        sales_data['數量'] = sales_data['數量'].astype(int)
        sales_data['折數'] = sales_data['折數'].astype(float)
        sales_data['瀏覽數'] = sales_data['瀏覽數'].astype(int)
        sales_data['week_day'] = sales_data['week_day'].astype(str)
        sales_data = pd.get_dummies(sales_data, drop_first=True)

        cols = list(sales_data.columns)
        cols.remove('數量')

        all_columns = "+".join(cols)
        mod = smf.ols(formula="數量~" + all_columns, data=sales_data)
        res = mod.fit()

        self.sales_model = res

        return res

    def predict_sales_daily(self, testing, price, traffic_prediction, start_date, end_date):
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
        new_data['折數'] = price
        new_data['瀏覽數'] = traffic_prediction
        new_data, _ = agg_weekly_data(new_data, False)
        new_data['week_day'] = new_data['week_day'].astype(str)
        new_data = pd.get_dummies(new_data, drop_first=True)
        new_data.drop(columns=['單據日期', 'week'], inplace=True)
        pred = pd.DataFrame()
        pred['單據日期'] = pd.date_range(start_date, end_date)
        pred['數量_pred'] = self.sales_model.predict(new_data)
        pred['數量_pred'] = [max(0, x) for x in pred['數量_pred']]

        mse_lm = mse(pred['數量_pred'][:len(testing)], testing)
        return pred, mse_lm


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
    # ts_test.ACF_PACF()
    # d = ts_test.ADF_test(None)

    # p_range = list(range(2, 6))
    # d_range = [0]
    # q_range = list(range(2, 6))
    # model, aic, params = ts_test.fit(p_range, d_range, q_range)
    # print("model is good (not lack of fit)?", ts_test.box_pierce_test())

    ts_test.auto_arima_fit()
    print("model is good (not lack of fit)?", ts_test.box_pierce_test())

    return


if __name__ == '__main__':
    main()
