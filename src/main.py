import numpy as np
from datetime import datetime
import time
from scipy.stats import poisson
import math
import operator
import pickle

from .DP import DynamicProgramming, find_best_quantity
from dir_config import DirConfig
from .data_preprocessing import *
from .demand_estimation import *
from .plot import *

path = DirConfig()


def do_cluster(num=3) ->(pd.DataFrame, list):
    '''
    讀取資料和分群
    '''
    # read data
    flow_dic, trans_dic, trans_data = read_data()
    trans_data['建議售價'] = trans_data['建議售價'].apply(str)
    trans_data['group_name'] = trans_data[[
        '品牌', '性別', '建議售價']].agg('-'.join, axis=1)

    # estimate the probability of each scenario
    trans_with_cluster = cluster(trans_data, num)
    return trans_with_cluster, [flow_dic, trans_dic, trans_data]


def predict_demand_dist(trans_with_cluster_item, trans_with_cluster_group, max_solds, data_list, start_date, end_date, discount_rate=np.arange(0.5, 1, 0.1), origin_price=2880, min_sold=0, rerun=False, debug_mode=True) -> (dict, dict):
    '''
    estimate each sceanrio demand distribution in each period
    Args:
        trans_with_cluster_item:  index is item id
        trans_with_cluster_group:  index is group id
        data_list: dict() including pageview, origin transaction data, origin channel transaction data
        start_date: estimation state date
        end_date: estimation end date
        discount_rate: list() all possible discount rate
        origin_price: product price
        min_sold: min number for one period
        rerun: retrain the model
        debug_mode: print detail information
    Output:
        demand_prob: each scenario demand distribution
        demand_exp:
    '''
    if not rerun:
        return pd.read_pickle(path.to_new_output_file('demand_prob.pkl')), None

    scenario_probability = trans_with_cluster_item['cluster_kind'].value_counts(
        normalize=True)
    if debug_mode:
        print(trans_with_cluster_item)
        print(trans_with_cluster_group)
        print(scenario_probability)

    # do demand estimation for each scenario
    demand_prob = {}
    demand_exp = {}
    flow_dic, trans_dic, trans_data = data_list
    for i in trans_with_cluster_group['cluster_kind'].unique():
        demand_prob[i] = {}
        print("----- predicting cluster", i, "-----")
        # max_sold = int(
        #     trans_with_cluster_item.loc[trans_with_cluster_item['cluster_kind'] == i, '數量'].quantile(0.95))
        max_sold = max_solds[i]
        # slice data and get transaction records that belong to cluster i
        trans = find_prediction_group(
            trans_data, trans_with_cluster_group.loc[trans_with_cluster_group['cluster_kind'] == i], 'best-selling')
        # print(trans)
        groups_in_cluster = trans['貨號'].unique()

        agg_lambda = pd.DataFrame(index=pd.date_range(start_date, end_date))
        arima_lambda = pd.DataFrame(index=pd.date_range(start_date, end_date))
        lm_lambda = pd.DataFrame(index=pd.date_range(start_date, end_date))
        for d in discount_rate:
            agg_lambda[d] = 0
            arima_lambda[d] = 0
            lm_lambda[d] = 0
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
            arima = TimeSeries(training['數量'], True)
            arima.fit(price=np.array(training[['折數', '建議售價']]),
                      p_range=range(3), q_range=range(3))

            # build linear regression model for every possible discount rate
            lm = LinearModel(
                training[['數量', '折數', '建議售價', '瀏覽數', 'seq_no', 'week_day']])
            traffic_pred = lm.fit_predict_traffic(
                testing['瀏覽數'], start_date, end_date)
            lm.fit_sales_daily()

            for d in discount_rate:  # for each discount rate
                # print("________ discount rate:", d)
                periods = (end_date-start_date).days+1
                arima_pred = arima.predict(
                    period=periods, regressors=np.array([d, origin_price]*periods).reshape(-1, 2))
                arima_mse = arima.MSE(arima_pred[:len(testing)], testing['數量'])
                arima_lambda[d] += np.array(arima_pred) / \
                    len(groups_in_cluster)
                # print(np.array(arima_pred))
                lm_pred, lm_mse = lm.predict_sales_daily(
                    testing['數量'], d, origin_price, traffic_pred, start_date, end_date)
                # print(lm_pred['數量_pred'])
                lm_lambda[d] += np.array(lm_pred['數量_pred']) / \
                    len(groups_in_cluster)

                pred = weighted_average(np.array([lm_mse, arima_mse]), np.array(
                    [lm_pred['數量_pred'], arima_pred]))

                agg_lambda[d] += pred / len(groups_in_cluster)

            if debug_mode:
                print("AIC =", arima.aic, end=", ")
                print("Best (p, d, q) =", arima.best_param_set, end=", ")
                print("model is good (not lack of fit)?",
                      arima.box_pierce_test(), end=", ")
                print("MSE =", arima_mse)

        # plot
        trans_all = fill_daily_na(agg_daily_data(trans))
        training, testing = train_test_split(trans_all)
        training['數量'] = remove_outlier(training['數量'])
        training.set_index('單據日期', inplace=True)
        testing.set_index('單據日期', inplace=True)
        for d in discount_rate:
            prediction = pd.DataFrame(
                {'ARIMA': arima_lambda[d], 'LM': lm_lambda[d]})
            plot_demand(str(i)+'_'+str(d),
                        training['數量'], testing['數量'], prediction)

        # trasform the predicted lambda into the demand distribution
        agg_lambda['單據日期'] = agg_lambda.index
        agg_lambda, _ = agg_weekly_data(agg_lambda, False)
        agg_lambda.drop(['單據日期', 'week_day', 'year'], axis=1, inplace=True)
        agg_lambda_pivot = agg_lambda.pivot_table(
            index='week', aggfunc=np.sum)
        if debug_mode:
            print(agg_lambda)
            print(agg_lambda_pivot)
        agg_lambda_pivot.reset_index(drop=True, inplace=True)
        transpose_lambda = agg_lambda_pivot.T
        transpose_lambda.drop(0, axis=1, inplace=True)
        if debug_mode:
            print(transpose_lambda)
        demand_exp[i] = transpose_lambda
        for t in transpose_lambda.columns:
            demand_prob[i][t] = {}
            for d in transpose_lambda.index:
                demand_prob[i][t][d] = dict(zip(range(min_sold, max_sold), [poisson.pmf(
                    x, max(0, transpose_lambda.loc[d, t])) for x in range(min_sold, max_sold)]))
                demand_prob[i][t][d][max_sold] = max(
                    0, 1 - poisson.cdf(max_sold-1, max(0, transpose_lambda.loc[d, t])))

    with open(path.to_new_output_file('demand_prob.pkl'), 'wb') as handle:
        pickle.dump(demand_prob, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return demand_prob, demand_exp

def caculate_expected_revenue(cluster_kinds, buy_num, probs, demand_dic, origin_price, period_num, prom_rate_list, buy_cost) -> int:
    '''
    calculate the expected revenure considering all scenario probs
    '''
    buy_rev_ev = 0
    for k in cluster_kinds:
        model = DynamicProgramming(demand_dic[k], prom_rate_list, buy_num, min(100, buy_num), origin_price, period_num=period_num)
        rev = model.run() - buy_cost * buy_num
        buy_rev_ev += rev*probs[k]
    return buy_rev_ev

def main(clus_num=3, mode='EV')->dict:
    print('mode:', mode)
    '''
    Args:
        mode: the two-stage model（EV / SA / DEP(recourse)）
        clus_num: cluster number
    output:
        total_buy_name: all result about buy number and expected profit
    '''
    # parameters setting
    origin_price = 3000
    discount_rate = np.arange(0.4, 1.1, 0.1)
    start_time = time.time()
    max_sold = 50
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 3, 8)
    period_num = math.floor((end_date-start_date).days/7)
    buy_cost = 900

    # use historical data
    trans_with_cluster_group, data_list = do_cluster(clus_num)
    cluster_group_dic = {v['group_name'].values[0]: v['cluster_kind'].values[0]
                         for i, v in trans_with_cluster_group[['group_name', 'cluster_kind']].iterrows()}
    data_list[2]['cluster_kind'] = data_list[2]['group_name'].map(
        cluster_group_dic)
    cluster_id_dic = {v['貨號']: v['cluster_kind'] for i, v in data_list[2][[
        'cluster_kind', '貨號']].drop_duplicates().iterrows()}
    trans_with_cluster_item = data_list[2][['數量', '貨號']].groupby(
        '貨號').agg({'數量': sum}).reset_index()
    trans_with_cluster_item['cluster_kind'] = trans_with_cluster_item['貨號'].map(
        cluster_id_dic)
    
    # order quantity range setting
    max_percent = 0.95
    min_percent = 0.05
    probs = trans_with_cluster_item[['cluster_kind', '貨號']].groupby(
        'cluster_kind').count() / trans_with_cluster_item.shape[0]
    probs = probs.to_dict()['貨號']
    cluster_kinds = trans_with_cluster_item['cluster_kind'].unique()
    if mode == 'EV' or mode == 'DEP':
        # 機率以 ev 方式加總
        max_q = int(trans_with_cluster_item['數量'].quantile(max_percent))
        max_solds = {k: max_q for k in cluster_kinds}
        min_q = int(trans_with_cluster_item['數量'].quantile(min_percent))
        min_solds = {k: min_q for k in cluster_kinds}
    else:
        # 各個 cluster 單獨算
        max_solds = {}
        min_solds = {}
        for k in cluster_kinds:
            max_solds[k] = int(
                trans_with_cluster_item.loc[trans_with_cluster_item['cluster_kind'] == k, '數量'].quantile(max_percent))
            min_solds[k] = int(
                trans_with_cluster_item.loc[trans_with_cluster_item['cluster_kind'] == k, '數量'].quantile(min_percent))
    
    # demand estimation
    demand_dic, _ = predict_demand_dist(trans_with_cluster_item, trans_with_cluster_group, max_solds, data_list, start_date=datetime(
        2021, 1, 1), end_date=datetime(2021, 3, 8), discount_rate=discount_rate, origin_price=origin_price)
    prom_rate_list = list(demand_dic[0][1].keys())

    # different mode have different demand distribution
    # two-stage stochastic dynamic model
    if mode == 'EV':  # expected value
        ev_demand_dic = {}
        for t in demand_dic[0].keys():
            ev_demand_dic[t] = {}
            for p in demand_dic[0][1].keys():
                ev_demand_dic[t][p] = {}
                for q in range(0, max_q+1):
                    ev_demand_dic[t][p][q] = 0
                    for k in cluster_kinds:
                        ev_demand_dic[t][p][q] += demand_dic[k][t][p][q] * probs[k]
        buy_num, model = find_best_quantity(
            ev_demand_dic, max_sold, origin_price, period_num, buy_cost, max_q=max_solds[k], min_q=min_solds[k], interval=10)
        model.export_result('best_policy_EV')
        print('購買量：', buy_num)
        total_buy_name = {}
        total_buy_name['EV'] = [buy_num, caculate_expected_revenue(cluster_kinds, buy_num, probs, demand_dic, origin_price, period_num, prom_rate_list, buy_cost)]
    elif mode == 'DEP':  # recourse
        total_buy_name = {}
        buy_rev_ev = {}
        total_buy_name[-1] = [probs[i] for i in probs] + [1]
        for i in range(min_q, max_q+1):
            buy_rev_ev[i] = 0
            total_buy_name[i] = []

            for k in cluster_kinds:
                model = DynamicProgramming(demand_dic[k], prom_rate_list, i, min(100, i),
                                           origin_price, period_num=period_num)
                rev = model.run() - buy_cost * i
                total_buy_name[i].append(rev)
                buy_rev_ev[i] += rev*probs[k]
            total_buy_name[i].append(buy_rev_ev[i])

            # total_rev += rev*probs[k]
        df = pd.DataFrame.from_dict(
            total_buy_name, orient='index', columns=list(cluster_kinds)+['expected'])
        df.to_excel(path.to_new_output_file('DEP_order_quantity.xlsx'))
        buy_num = max(buy_rev_ev.items(), key=operator.itemgetter(1))[0]
        print('購買量：', buy_num)
        total_buy_name = {}
        total_buy_name['DEP'] = [buy_num, buy_rev_ev[buy_num]]
        revs = []
        for k in cluster_kinds:
            model = DynamicProgramming(demand_dic[k], prom_rate_list, buy_num, min(100, buy_num),
                                       origin_price, period_num=period_num)
            rev = model.run() - buy_cost * buy_num
            revs.append(rev)
            model.export_result('best_policy_DEP_'+str(k))
        pd.DataFrame([[buy_num]*len(cluster_kinds+1), [probs[i] for i in probs] + [1], revs+[buy_rev_ev[buy_num]]],
                     index=['訂購量', '機率', 'revenue'], columns=list(cluster_kinds)+['expected']).to_excel(path.to_new_output_file('DEP_best_ev.xlsx'))

    else:  # SA
        total_buy_name_ev = {}
        total_buy_name = {}
        for k, v in demand_dic.items():
            print("demand distribution of cluster", k)
            buy_num, model = find_best_quantity(
                v, max_sold, origin_price, period_num, buy_cost, max_q=max_solds[k], min_q=min_solds[k], interval=10)
            print(buy_num, model.total_reward)
            model.export_result('best_policy_'+str(k))

            total_buy_name_ev[k] = [buy_num, model.total_reward]
            total_buy_name[k] = [buy_num, caculate_expected_revenue(cluster_kinds, buy_num, probs, demand_dic, origin_price, period_num, prom_rate_list, buy_cost)]
        pd.DataFrame.from_dict(total_buy_name_ev, orient='index', columns=[
            'order quantity', 'expexted revenue']).to_excel(path.to_new_output_file('order_quantity.xlsx'))
            

    print("time: %.2f seconds" % (time.time() - start_time))
    return total_buy_name


if __name__ == '__main__':
    EV_dic = main(mode='EV')
    DEP_dic = main(mode='DEP')
    SA_dic = main(mode='SA')
    final = {**EV_dic, **DEP_dic, **SA_dic}
    pd.DataFrame.from_dict(final, orient='index').to_excel(path.to_new_output_file('SP_summary.xlsx'))
