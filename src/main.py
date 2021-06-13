import numpy as np
from datetime import datetime
import time
from scipy.stats import poisson
import math

from .DP import DynamicProgramming, find_best_quantity
from dir_config import DirConfig
from .data_preprocessing import *
from .demand_estimation import *

path = DirConfig()


def do_cluster(num=3):
    # read data
    flow_dic, trans_dic, trans_data = read_data()
    trans_data['建議售價'] = trans_data['建議售價'].apply(str)
    trans_data['group_name'] = trans_data[[
        '品牌', '性別', '建議售價']].agg('-'.join, axis=1)

    # estimate the probability of each scenario
    trans_with_cluster = cluster(trans_data, num)
    return trans_with_cluster, [flow_dic, trans_dic, trans_data]


def predict_demand_dist(trans_cluster_v, trans_with_cluster, data_list, start_date, end_date, discount_rate=np.arange(0.5, 1, 0.1), origin_price=2880, min_sold=0, debug_mode=True):
    scenario_probability = trans_with_cluster['cluster_kind'].value_counts(
        normalize=True)
    if debug_mode:
        print(trans_cluster_v)
        # print(trans_cluster_v.pivot_table(index='cluster_kind', aggfunc={
        #     ('數量', 'count'): np.sum, ('數量', 'sum'): np.sum, ('建議售價', 'mean'): np.mean, ('販售時間', ''): np.mean}))
        print(scenario_probability)

    # do demand estimation for each scenario
    demand_prob = {}
    flow_dic, trans_dic, trans_data = data_list
    for i in trans_with_cluster['cluster_kind'].unique():
        demand_prob[i] = {}
        print("----- predicting cluster", i, "-----")
        max_sold = int(
            trans_cluster_v.loc[trans_cluster_v['cluster_kind'] == i, '數量'].quantile(0.95))
        # slice data and get transaction records that belong to cluster i
        groups_in_cluster = trans_cluster_v.loc[trans_cluster_v['cluster_kind'] == i]['貨號'].unique(
        )
        print("there are", len(groups_in_cluster), "groups")
        trans = trans_data.loc[trans_data['貨號'].isin(
            groups_in_cluster)]

        agg_lambda = pd.DataFrame(index=pd.date_range(start_date, end_date))
        for d in discount_rate:
            agg_lambda[d] = 0
        for g in trans['客戶名稱'].unique():
            # aggregate data and train/test split
            channel_trans = trans.loc[trans['客戶名稱'] == g]
            channel_trans = fill_daily_na(agg_daily_data(channel_trans))
            channel_trans, _ = agg_weekly_data(channel_trans)
            channel_trans = pd.merge(
                channel_trans, flow_dic[g], left_on='單據日期', right_on='單據日期')
            channel_trans.drop('Unnamed: 0', axis=1, inplace=True)

            training, testing = train_test_split(channel_trans)
            training.set_index('單據日期', inplace=True)
            testing.set_index('單據日期', inplace=True)

            # build time series model
            arima = TimeSeries(training['數量'], False)
            arima.fit(price=np.array(training['折數']).reshape(-1, 1),
                      p_range=range(3), q_range=range(3))

            # build linear regression model for every possible discount rate
            lm = LinearModel(
                training[['數量', '折數', '瀏覽數', 'seq_no', 'week_day']])
            traffic_pred = lm.fit_predict_traffic(
                testing['瀏覽數'], start_date, end_date)
            lm.fit_sales_daily()

            for d in discount_rate:  # for each discount rate
                print("________ discount rate:", d)
                arima_pred = arima.predict(
                    period=(end_date-start_date).days+1, price=d)
                arima_mse = arima.MSE(arima_pred[:len(testing)], testing['數量'])
                lm_pred, lm_mse = lm.predict_sales_daily(
                    testing['數量'], d, traffic_pred, start_date, end_date)
                pred = weighted_average(np.array([lm_mse, arima_mse]), np.array(
                    [lm_pred['數量_pred'], arima_pred]))

                agg_lambda[d] += pred / len(groups_in_cluster)

            if debug_mode:
                print("AIC =", arima.aic, end=", ")
                print("Best (p, d, q) =", arima.best_param_set, end=", ")
                print("model is good (not lack of fit)?",
                      arima.box_pierce_test(), end=", ")
                print("MSE =", arima_mse)

        # 在折扣為 d 時的，一個組合平均未來每一天的 lambda
        # for d in discount_rate:
        #     agg_lambda[d] = agg_lambda[d] / len(groups_in_cluster)

        # trasform the predicted lambda into the demand distribution
        agg_lambda['單據日期'] = agg_lambda.index
        agg_lambda, _ = agg_weekly_data(agg_lambda, False)
        agg_lambda.drop(['單據日期', 'week_day', 'year'], axis=1, inplace=True)
        agg_lambda_pivot = agg_lambda.pivot_table(index='week', aggfunc=np.sum)
        if debug_mode:
            print(agg_lambda)
            print(agg_lambda_pivot)
        agg_lambda_pivot.reset_index(drop=True, inplace=True)
        transpose_lambda = agg_lambda_pivot.T
        transpose_lambda.drop(0, axis=1, inplace=True)
        if debug_mode:
            print(transpose_lambda)
        for t in transpose_lambda.columns:
            demand_prob[i][t] = {}
            for d in transpose_lambda.index:
                demand_prob[i][t][d] = dict(zip(range(min_sold, max_sold), [poisson.pmf(
                    x, max(0, transpose_lambda.loc[d, t])) for x in range(min_sold, max_sold)]))
                demand_prob[i][t][d][max_sold] = max(
                    0, 1 - poisson.cdf(max_sold-1, max(0, transpose_lambda.loc[d, t])))
    return demand_prob


def main():
    # parameters setting
    origin_price = 800
    discount_rate = np.arange(0.4, 1.1, 0.1)
    start_time = time.time()
    max_sold = 50
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 2, 8)
    period_num = math.floor((end_date-start_date).days/7)
    buy_cost = 400

    trans_cluster, data_list = do_cluster(3)
    # print(trans_cluster[['group_name', 'cluster_kind']])
    cluster_group_dic = {v['group_name'].values[0]: v['cluster_kind'].values[0]
                         for i, v in trans_cluster[['group_name', 'cluster_kind']].iterrows()}
    # print(cluster_group_dic)
    data_list[2]['cluster_kind'] = data_list[2]['group_name'].map(
        cluster_group_dic)
    # print(data_list[2])
    cluster_id_dic = {v['貨號']: v['cluster_kind'] for i, v in data_list[2][[
        'cluster_kind', '貨號']].drop_duplicates().iterrows()}
    # print(cluster_id_dic)
    trans_cluster_v = data_list[2][['數量', '貨號']].groupby(
        '貨號').agg({'數量': sum}).reset_index()
    trans_cluster_v['cluster_kind'] = trans_cluster_v['貨號'].map(cluster_id_dic)
    # print(trans_cluster)
    # for k in trans_cluster['cluster_kind'].unique():
    #     max_q = int(trans_cluster.loc[trans_cluster['cluster_kind'] == k, '數量'].quantile(0.95))
    #     min_q = int(trans_cluster.loc[trans_cluster['cluster_kind'] == k, '數量'].quantile(0.05))
    #     print(max_q, min_q)
    for k, v in predict_demand_dist(trans_cluster_v, trans_cluster, data_list, start_date=datetime(2021, 1, 1), end_date=datetime(2021, 3, 8), discount_rate=discount_rate, origin_price=origin_price).items():
        print("demand distribution of cluster", k)
        # print(v)
        # print(v.keys())
        print("-")
        max_q = int(
            trans_cluster_v.loc[trans_cluster_v['cluster_kind'] == k, '數量'].quantile(0.95))
        min_q = int(
            trans_cluster_v.loc[trans_cluster_v['cluster_kind'] == k, '數量'].quantile(0.05))
        buy_num, model = find_best_quantity(
            v, max_sold, origin_price, period_num, buy_cost, max_q=max_q, min_q=min_q, interval=10)
        print(buy_num, model.total_reward)
        model.export_result('best_policy_'+str(k))

    print("time: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    main()
