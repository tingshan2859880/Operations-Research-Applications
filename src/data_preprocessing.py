from datetime import datetime
from scipy.stats.mstats import winsorize
import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from hampel import hampel
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from dir_config import DirConfig

path = DirConfig()

# 讀入個人設定檔
with open("_config.yml", "rb") as stream:
    data = yaml.load(stream, Loader=yaml.FullLoader)
# data
plt.rcParams['font.family'] = [data['FontFamily']]


# train test
def train_test_split(data):
    train = data[data['單據日期'].dt.year == 2020]
    test = data[data['單據日期'].dt.year == 2021]
    return train, test


def cluster(transaction_data, num=3):
    transaction_data['建議售價'] = transaction_data['建議售價'].apply(str)
    # transaction_data['group_name'] = transaction_data['貨號']
    transaction_data['group_name'] = transaction_data[[
        '品牌', '性別', '建議售價']].agg('-'.join, axis=1)
    transaction_data['建議售價'] = transaction_data['建議售價'].apply(int)
    ttl_num_df = transaction_data[['group_name', '數量', '單據日期', '建議售價']].groupby(['group_name']).agg(
        {'數量': ['count', 'sum'], '單據日期': ['min', 'max'], '建議售價': 'mean'}).reset_index()
    ttl_num_df['販售時間'] = (ttl_num_df['單據日期']['max'] -
                          ttl_num_df['單據日期']['min']).dt.days
    filters = ['數量', '販售時間', '建議售價']
    X = ttl_num_df[filters].values
    kmeans = KMeans(n_clusters=num, random_state=0).fit(X)

    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=False, sharex=False)
    df = pd.DataFrame(X, columns=['售出次數', '售出總數', '販售時間', '建議售價'])
    df['cluster'] = kmeans.labels_
    # print(df)
    sns.scatterplot(y="售出次數", x="售出總數", data=df,  hue='cluster', ax=axes[0])
    sns.scatterplot(y="售出次數", x="販售時間", data=df,  hue='cluster', ax=axes[1])
    sns.scatterplot(y="售出總數", x="建議售價", data=df,  hue='cluster', ax=axes[2])
    fig.savefig(path.to_new_output_file('test.png'))

    ttl_num_df['cluster_kind'] = kmeans.labels_
    return (ttl_num_df)


def read_data(channel=['A', 'B', 'C']):
    '''
    read channel data
    '''
    flow = {}
    sales_data = {}
    sales_all = pd.read_excel(path.to_new_file('Sales_data.xlsx'))
    sales_all['折數'] = round(sales_all['單價'] / sales_all['建議售價'], 1)
    sales_all = sales_all.loc[sales_all['建議售價'] < 5000]
    print("折數", sales_all['折數'])
    print(np.mean(sales_all['折數']))
    for c in channel:
        flow[c] = pd.read_excel(path.to_new_file(c+'_流量資料.xlsx'))

        sales_data[c] = sales_all.loc[sales_all['客戶名稱'] == c]

    return flow, sales_data, sales_all


def agg_daily_data(data):
    data_pivot = data.pivot_table(
        index='單據日期', aggfunc={'折數': np.mean, '建議售價': np.mean, '數量': np.sum})
    data_pivot.reset_index(inplace=True)
    return data_pivot


def fill_daily_na(data):
    """
    將當天沒有販售紀錄的鞋款，以前一天的價格作為當天沒有販售的價格
    """
    date = pd.DataFrame()
    date['單據日期'] = pd.date_range(
        start=min(data['單據日期']), end=max(data['單據日期']), freq='D')

    daily_sales = date.merge(data, on="單據日期", how="left")
    daily_sales["數量"] = daily_sales["數量"].fillna(0)
    daily_sales["折數"] = daily_sales["折數"].fillna(
        method="ffill")  # 用前一天的單價當作沒有銷售量的那一天的單價
    daily_sales["折數"] = daily_sales["折數"].fillna(
        method="bfill")  # 還沒有fill的那些在用第一次有銷售紀錄的fill
    daily_sales["建議售價"] = daily_sales["建議售價"].fillna(
        method="ffill")  # 用前一天的單價當作沒有銷售量的那一天的單價
    daily_sales["建議售價"] = daily_sales["建議售價"].fillna(
        method="bfill")  # 還沒有fill的那些在用第一次有銷售紀錄的fill
    return daily_sales


def agg_weekly_data(data, return_weekly_pivot=True, year=[2020, 2021]):
    # year = data['單據日期'].dt.year.unique()
    week_accumulate = 0  # 如果資料跨年，要累積計算
    data['week'] = data['單據日期'].dt.isocalendar().week
    data['week_day'] = data['單據日期'].dt.weekday
    data['year'] = data['單據日期'].dt.isocalendar().year

    for y in year:
        data.loc[data['year'] == y, 'week'] += week_accumulate
        week_accumulate += int(datetime(y, 12, 31).strftime("%W"))
    data.loc[data['week_day'] == 0,
             'week'] = data.loc[data['week_day'] == 0, 'week'] - 1

    weekly_data = None
    if return_weekly_pivot:
        weekly_data = data.pivot_table(
            index='week', aggfunc={'折數': np.mean, '建議售價': np.mean, '數量': np.sum})

    return data, weekly_data


def remove_outlier(data: list, method='hampel'):
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
    flow, sales_data, sales_all = read_data()

    weekly_sales = {}
    for k, v in sales_data.items():
        sales_data[k] = fill_daily_na(agg_daily_data(sales_data[k]))

        sales_data[k], weekly_sales[k] = agg_weekly_data(sales_data[k])
    print(weekly_sales)


if __name__ == '__main__':
    # main()
    flow_dic, trans_dic, trans_data = read_data()
    print(flow_dic)
    # cluster(trans_data)
    # train, test = train_test_split(trans_data)
    # print(train)
    # print(test)
