from datetime import datetime
from scipy.stats.mstats import winsorize
import pandas as pd
import os
import numpy as np
from hampel import hampel


def read_data(channel=['A', 'B', 'C']):
    '''
    read channel data
    '''
    flow = {}
    sales_data = {}
    sales_all = pd.read_excel(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data', 'Sales_data.xlsx'))
    for c in channel:
        flow[c] = pd.read_excel(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', c+'_流量資料.xlsx'))

        sales_data[c] = sales_all.loc[sales_all['CustCode'] == c]

    return flow, sales_data, sales_all


def agg_daily_data(data):
    data_pivot = data.pivot_table(index='SlipDate', aggfunc={
                                  '單價': np.mean, '數量': np.sum})
    data_pivot.reset_index(inplace=True)
    return data_pivot


def fill_daily_na(data):
    """
    將當天沒有販售紀錄的鞋款，以前一天的價格作為當天沒有販售的價格
    """
    date = pd.DataFrame()
    date['SlipDate'] = pd.date_range(
        start=min(data['SlipDate']), end=max(data['SlipDate']), freq='D')

    daily_sales = date.merge(data, on="SlipDate", how="left")
    daily_sales["數量"] = daily_sales["數量"].fillna(0)
    daily_sales["單價"] = daily_sales["單價"].fillna(
        method="ffill")  # 用前一天的單價當作沒有銷售量的那一天的單價
    daily_sales["單價"] = daily_sales["單價"].fillna(
        method="bfill")  # 還沒有fill的那些在用第一次有銷售紀錄的fill
    return daily_sales


def agg_weekly_data(data):
    year = data['SlipDate'].dt.year.unique()
    week_accumulate = 0  # 如果資料跨年，要累積計算
    data['week'] = data['SlipDate'].dt.isocalendar().week
    data['week_day'] = data['SlipDate'].dt.weekday
    data['year'] = data['SlipDate'].dt.isocalendar().year

    for y in year:
        data.loc[data['year'] == y, 'week'] += week_accumulate
        week_accumulate += int(datetime(y, 12, 31).strftime("%W"))
    data.loc[data['week_day'] == 0,
             'week'] = data.loc[data['week_day'] == 0, 'week'] - 1

    weekly_data = data.pivot_table(
        index='week', aggfunc={'單價': np.mean, '數量': np.sum})

    return data, weekly_data


def remove_outlier(data: list, method='winsorizie'):
    """
    移除 data 裡面的 Outlier
        Args:
            data: 一個 list 的 資料，可以偵測 outlier 並取代成合適的值
            method: winsorizie/hampel，選擇要做 imputation 的方法
    """
    if method == 'winsorizie':
        result = winsorize(data, limits=[0.05, 0.05], inclusive=(False, False))
    if method == 'hampel':
        result = hampel(data, window_size=5, n=3, imputation=True)
    return list(result)


def main():
    # import data
    flow, sales_data = read_data()

    weekly_sales = {}
    for k, v in sales_data.items():
        sales_data[k] = fill_daily_na(agg_daily_data(sales_data[k]))

        sales_data[k], weekly_sales[k] = agg_weekly_data(sales_data[k])
    print(weekly_sales)


if __name__ == '__main__':
    main()
