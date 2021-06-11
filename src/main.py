from .DP import DynamicProgramming
from dir_config import DirConfig
from .data_preprocessing import *

path = DirConfig()


def main():
    # read data
    flow_dic, trans_dic, trans_data = read_data()
    trans_with_cluster = cluster(trans_data)
    scenario_probability = trans_with_cluster['cluster_kind'].value_counts(
        normalize=True)
    print(scenario_probability)

    for i in trans_with_cluster['cluster_kind'].unique():
        trans = trans_with_cluster.loc[trans_with_cluster['cluster_kind'] == i, ]
        print(trans)

    # train, test = train_test_split(trans_data)

    return


if __name__ == '__main__':
    main()
