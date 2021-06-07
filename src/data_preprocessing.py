from datetime import datetime
from scipy.stats.mstats import winsorize
import pandas as pd
import os
import numpy as np
from hampel import hampel

from dir_config import DirConfig

# old

def read_new_traffic(update=True, use_old=False):
    """
    讀取新的（202010-202102）流量資料
    input:
        update: 是否將新流量資料檔合併至 traffic_all 中
    output:
        顯示更新到最新日期的所有通路的流量資料，
        若update=True，就更新output/traffic_all.pkl
    """
    path = DirConfig()
    if use_old:
        current_traffic = pd.read_pickle(
            path.to_output_file("traffic_all.pkl"))
        return current_traffic
    else:
        # read momo data
        input_dir = path.to_traffic_folder("MOMO")
        momo_traffic = pd.DataFrame(columns=['日期', '造訪數', '瀏覽數'])
        for parents, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                df = pd.read_excel(os.path.join(parents, filename), sheet_name='流量趨勢', usecols=[
                    '日期', '造訪數', '瀏覽數'], skiprows=range(2))
                df['日期'] = pd.to_datetime(
                    filename[5:9] + '/' + df['日期'].astype(str))
                momo_traffic = momo_traffic.append(df, ignore_index=True)

        momo_traffic = momo_traffic.sort_values(by='日期')
        momo_traffic = momo_traffic.rename(
            columns={'造訪數': '造訪數_MOMO', '瀏覽數': '瀏覽數_MOMO'})
        momo_traffic = momo_traffic.reset_index(drop=True)

        # read yahoo data
        yahoo_traffic = pd.read_excel(path.to_traffic_file(
            "YAHOO", "yahoo流量.xlsx"), header=None, usecols=[0, 6], names=['日期', '瀏覽數_YAHOO'])
        yahoo_traffic = yahoo_traffic[yahoo_traffic['日期'].map(
            type) == datetime]
        yahoo_traffic['日期'] = pd.to_datetime(yahoo_traffic['日期'])
        yahoo_traffic['日期'] = yahoo_traffic['日期'].dt.strftime('%Y-%m-%d')
        yahoo_traffic['日期'] = pd.to_datetime(yahoo_traffic['日期'])

        # read shopee data
        input_dir = path.to_traffic_folder("SHOPEE")
        shopee_traffic = pd.DataFrame(columns=['日期', '頁面瀏覽數', '訪客數'])
        for parents, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                df = pd.read_excel(os.path.join(parents, filename), sheet_name='已付款訂單', usecols=[
                    '日期', '頁面瀏覽數', '訪客數'], skiprows=range(4))
                shopee_traffic = shopee_traffic.append(df, ignore_index=True)
        shopee_traffic = shopee_traffic.sort_values(by='日期')
        shopee_traffic = shopee_traffic.rename(
            columns={'頁面瀏覽數': '瀏覽數_SHOPEE', '訪客數': '造訪數_SHOPEE'})
        shopee_traffic = shopee_traffic.reset_index(drop=True)
        shopee_traffic['日期'] = pd.to_datetime(shopee_traffic['日期'])

        # 合併三個通路的流量
        new_traffic_all = pd.merge((pd.merge(
            momo_traffic, yahoo_traffic, on='日期', how='outer')), shopee_traffic, on='日期', how='outer')
        new_traffic_all

        # 將新資料加回舊資料中
        current_traffic = pd.read_pickle(
            path.to_output_file("traffic_all.pkl"))
        traffic_all = current_traffic.append(new_traffic_all)
        traffic_all.reset_index(drop=True, inplace=True)
        if update == True:
            traffic_all.to_pickle(path.to_output_file("traffic_all.pkl"))
            return traffic_all

        # 若不update, 則同樣return所有資料但不更新traffic_all.pkl
        return traffic_all


def read_transaction_data(update=False, use_old=False):
    """
    讀取交易紀錄檔案並清理資料
    input:
        file_path: 新的交易紀錄檔
        update: 是否將新交易紀錄檔合併至 transaction_all.xlsx 中
    output:
        包含新交易紀錄在內的所有交易紀錄
    """
    path = DirConfig()
    if use_old:
        current_transaction = pd.read_pickle(
            path.to_output_file("transaction_all.pkl"))
        return current_transaction
    else:
        product_info = pd.read_pickle(
            path.to_output_file("product_information.pkl"))
        # print(set(product_info['性別']))
        transaction_data = pd.read_excel(path.to_input_file('NB業績資料.xlsx'))
        # 預處理交易資料
        transaction_data.dropna(axis=1, inplace=True)  # 刪除有空值的資料
        # 刪除銷售數量為負的資料
        transaction_data = transaction_data[transaction_data['數量'] >= 0]
        transaction_data = transaction_data.iloc[:, 4:]  # 刪除原始資料中不需要的欄位
        transaction_data['單價'] = transaction_data['金額'] / \
            transaction_data['數量']  # 計算商品單價
        # 篩選出鞋類商品
        transaction_data = transaction_data[transaction_data['產品名稱'].str.contains(
            '鞋')]
        transaction_data['單據日期'] = pd.to_datetime(
            transaction_data['單據日期'], format='%Y/%m/%d')
        transaction_data.rename(columns={'原廠料號': '貨號'}, inplace=True)

        # 整合交易資料與商品資料
        all_data = pd.merge(transaction_data, product_info,
                            left_on='貨號', right_on='貨號', how='inner')
        # all_data.drop(columns='貨號', inplace=True)

        us, cm = split_size(all_data.pop('尺寸'))
        all_data['size_US'] = us
        all_data['size_cm'] = cm

        # 蝦皮 要改成 SHOPEE
        all_data.loc[(all_data['客戶名稱'].str.find('蝦皮') != -1),
                     '客戶名稱'] = 'SHOPEE'
        all_data['客戶名稱'] = clean_cust(all_data['客戶名稱'],
                                      to_clean=('SHOPEE', '官網', '商城', '樂天', 'YAHOO', 'MOMO', 'PCHOME'))
        all_data.drop(['金額'], axis=1, inplace=True)

        all_data['產品名稱'] = all_data['產品名稱'].str.split(
            '_').str[0].str.replace('【New Balance】', '')

        # 將新資料加回舊資料中
        current_transaction = pd.read_pickle(
            path.to_output_file("transaction_all.pkl"))
        current_transaction = current_transaction.append(all_data)
        current_transaction.dropna(axis=0, inplace=True)
        current_transaction.reset_index(drop=True, inplace=True)
        if update:
            current_transaction.to_pickle(
                path.to_output_file("transaction_all.pkl"))

        return current_transaction



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
