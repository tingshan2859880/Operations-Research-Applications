import pandas as pd
import os

from data_preprocessing import *


class DynamicProgramming:
    def __init__(self, lambda_dict, prom_rate_list, inv_num, max_sold, price, week_dic={}, period_num=52):
        self.lambda_dict = lambda_dict
        self.prom_rate_list = prom_rate_list
        self.inv_num = inv_num
        self.max_sold = max_sold
        self.price = price
        self.week_dic = week_dic
        self.period_num = period_num
        return

    def run(self, save_result=False):
        """
        output:
            data: best action
            value_data: EV for best action
        excel 表：best policy
        """

        # print(lambda_dict)
        # max_sold = 10
        sold_unit = 1
        ttl_state = set(range(0, self.inv_num + sold_unit, sold_unit))

        # action
        # raw_price = 3000
        # if mode == 'discrete':
        #     # print('discrete')
        #     A = [-1] + list(lambda_dict.keys())
        # elif(mode == 'poisson'):
        #     # print('poisson')
        #     A = [-1] + list(lambda_dict[0].keys())
        A = [-1] + self.prom_rate_list
        print("total state:", sorted(ttl_state))
        print("total action:", A)

        # transition function and optimality equation
        V = {}
        V_record = {}
        best_action = {}
        V[0] = {}
        for s in ttl_state:
            V[0][s] = self.salvage_value * s  # 剩餘價值
        lambda_list = {}
        for t in range(1, self.period_num + 1):
            if t in self.week_dic:
                action_set = [-1, self.week_dic[t]]
            else:
                action_set = A
            V[t] = {}
            V_record[t] = {}
            best_action[t] = {}
            for s in ttl_state:
                V_record[t][s] = {}
                for a in action_set:
                    rev = 0
                    if a == -1:
                        V[t][0] = V[t - 1][0]
                    else:
                        lambda_list = self.lambda_dict[t-1][a]
                        for exp_demand, prob in lambda_list.items():
                            # for
                            s_s = max(0, s - exp_demand)  # 下一期剩下的量
                            rev += (V[t - 1][s_s] + min(exp_demand, s) * a * self.price) * prob - self.inv_cost * 7 * prob * (
                                s - min(exp_demand, s))
                            # print("V", V[t - 1][s_s])
                            # print("exp_demand", exp_demand)
                            # print(a)
                            # print("s", s)
                            # print("prob", prob)
                            # print("revv", rev)
                        rev = float(rev)
                        if s in V[t]:
                            # print("rev", rev)
                            if rev > V[t][s]:
                                if s <= 0:
                                    best_action[t][s] = -1
                                else:
                                    best_action[t][s] = a
                                V[t][s] = rev
                        else:
                            if s <= 0:
                                best_action[t][s] = -1
                            else:
                                best_action[t][s] = a
                            V[t][s] = rev
                    V_record[t][s][a] = rev

        # print(V_record)

        # DP result
        action_data = {}
        value_data = {}
        lambda_data = {}
        for i in ttl_state:
            action_data[i] = [0] * (self.period_num + 1)
            value_data[i] = [0] * (self.period_num + 1)
            lambda_data[i] = [0] * (self.period_num + 1)
        for t in range(self.period_num, 0, -1):
            for i in best_action[t].keys():
                action_data[i][t] = best_action[t][i]
                value_data[i][t] = V[t][i]

        data = pd.DataFrame.from_dict(action_data, orient='index').iloc[1:, 1:]
        value_data = pd.DataFrame.from_dict(value_data, orient='index')

        if save_result:
            if os.path.exists(os.path.join(os.abspath(__file__), "output")) == False:
                os.makedirs(os.path.join(os.abspath(__file__), "output"))

            with pd.ExcelWriter("output/best_policy.xlsx", engine='xlsxwriter') as writer:
                data.to_excel(writer, sheet_name="action")
                value_data.to_excel(writer, sheet_name="value")
                for t, plan in V_record.items():
                    period_data = pd.DataFrame.from_dict(
                        plan, orient='index', columns=sorted(A))
                    period_data['best_value'] = V[t].values()
                    period_data['best_action'] = best_action[t].values()
                    period_data.to_excel(writer, sheet_name="period_" + str(t))
        self.data = data
        self.value_data = value_data
        self.total_reward = V[self.period_num][max(ttl_state)]
        return self.total_reward
