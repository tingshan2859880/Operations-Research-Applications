import numpy as np
from datetime import datetime
import time
from scipy.stats import poisson

from .DP import DynamicProgramming
from dir_config import DirConfig
from .data_preprocessing import *
from .demand_estimation import *

path = DirConfig()


def predict_demand_dist(start_date, end_date, discount_rate=np.arange(0.5, 1, 0.1), origin_price=2880, min_sold=0, max_sold=10, debug_mode=True):
    # read data
    flow_dic, trans_dic, trans_data = read_data()
    trans_data['建議售價'] = trans_data['建議售價'].apply(str)
    trans_data['group_name'] = trans_data[[
        '品牌', '性別', '建議售價']].agg('-'.join, axis=1)

    # estimate the probability of each scenario
    trans_with_cluster = cluster(trans_data)

    scenario_probability = trans_with_cluster['cluster_kind'].value_counts(
        normalize=True)
    if debug_mode:
        print(trans_with_cluster)
        print(trans_with_cluster.pivot_table(index='cluster_kind', aggfunc={
            ('數量', 'count'): np.sum, ('數量', 'sum'): np.sum, ('建議售價', 'mean'): np.mean, ('販售時間', ''): np.mean}))
        print(scenario_probability)

    # do demand estimation for each scenario
    demand_prob = {}

    for i in trans_with_cluster['cluster_kind'].unique():
        demand_prob[i] = {}
        print("----- predicting cluster", i, "-----")

        # slice data and get transaction records that belong to cluster i
        groups_in_cluster = trans_with_cluster.loc[trans_with_cluster['cluster_kind'] == i]['group_name'].unique(
        )
        print("there are", len(groups_in_cluster), "groups")
        trans = trans_data.loc[trans_data['group_name'].isin(
            groups_in_cluster)]

        agg_lambda = pd.DataFrame(index=pd.date_range(start_date, end_date))
        for d in discount_rate:  # for each discount rate
            print("________ discount rate:", d)
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
                arima.fit(p_range=range(3), q_range=range(3))
                arima_pred = arima.predict(period=(end_date-start_date).days+1)
                arima_mse = arima.MSE(arima_pred[:len(testing)], testing['數量'])
                if debug_mode:
                    print("AIC =", arima.aic, end=", ")
                    print("Best (p, d, q) =", arima.best_param_set, end=", ")
                    print("model is good (not lack of fit)?",
                          arima.box_pierce_test(), end=", ")
                    print("MSE =", arima_mse)

                # build linear regression model for every possible discount rate
                lm = LinearModel(
                    training[['數量', '單價', '瀏覽數', 'seq_no', 'week_day']])
                traffic_pred = lm.fit_predict_traffic(
                    testing['瀏覽數'], start_date, end_date)
                lm.fit_sales_daily()
                lm_pred, lm_mse = lm.predict_sales_daily(
                    testing['數量'], d*origin_price, traffic_pred, start_date, end_date)
                pred = weighted_average(np.array([lm_mse, arima_mse]), np.array(
                    [lm_pred['數量_pred'], arima_pred]))

                agg_lambda[d] += pred

            # 在折扣為 d 時的，一個組合平均未來每一天的 lambda
            agg_lambda[d] = agg_lambda[d] / len(groups_in_cluster)

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
    start_time = time.time()
    for k, v in predict_demand_dist(start_date=datetime(2021, 1, 1), end_date=datetime(2021, 3, 8), max_sold=50).items():
        print("demand distribution of cluster", k)
        print(v)
        print("-")
    print("time: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    main()
