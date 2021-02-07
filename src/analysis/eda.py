import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import chi2

def analyze_data(arg1, year, dynamic_cols, target, wait_th, cpu_th,nominal):
    """
    Perform exploratory data analysis, further clean the data, and output dataset
    for the following model
    """

    def time_tf(x):
        if ((x[:4]) == year):
            return True
        else: return False

    outpath = "src/analysis"

    #data frame within the 2020 interval
    dynamic, static = arg1
    tf_list = dynamic['batch_id'].apply(time_tf)
    dynamic = dynamic[tf_list]

    #examine where the outlier starts
    dynamic[target].value_counts(bins = 3000)
    #create a threshold for the outlier
    wait_threshold = wait_th

    #feature engineering on dynamic_cols
    dynamic[dynamic_cols] = dynamic[dynamic_cols].astype(float)
    dynamic = dynamic[dynamic.before_cpuutil_max < int(cpu_th)]
    dynamic = dynamic[dynamic.wait_msecs < int(wait_threshold)]
    #log tranform because their distributions are strongly skewed
    dynamic['before_harddpf_max'] = np.log(dynamic['before_harddpf_max'])
    dynamic['before_diskutil_max'] = np.log(dynamic['before_diskutil_max'])
    dynamic['before_networkutil_max'] = np.log(dynamic['before_networkutil_max'])
    dynamic[dynamic_cols] = dynamic[dynamic_cols].apply(lambda x: x.apply(lambda y: 0 if y < 0 else y))

    #select the first 1200 rows of data
    subset = dynamic.head(1200)

    sns.pairplot(subset)
    #drop before_networkutil_max because it shows no pattern
    dynamic = dynamic.drop("before_networkutil_max", axis =1)
    #we see outliers less than five
    dynamic = dynamic[dynamic['before_cpuutil_max'] > 5]
    dynamic = dynamic[dynamic['before_diskutil_max'] > 5]
    dynamic = dynamic[dynamic['before_harddpf_max'] > 5]

    subset = dynamic.head(1200)
    sns.pairplot(subset)
    plt.savefig(outpath+"/pairplot.png")

    plt.figure(figsize=(8, 4))
    sns.heatmap(subset.corr(), annot=True, linewidths=.5)
    plt.savefig(outpath+"/heatmap.png")

    #create the data target
    dynamic['target'] = pd.cut(dynamic.wait_msecs, bins=4, labels=[1,2,3,4])

    #feature engineering on static_cols
    def chi(table):
        stat, p, dof, expected = chi2_contingency(table)
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        print('probability=%.3f, critical=%.3f, stat=%.3f' %
              (prob, critical, stat))
        if abs(stat) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
        # interpret p-value
        alpha = 1.0 - prob
        print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')

    df = dynamic.merge(static, on = "guid", how = "left")
    for i in nominal:
        print(i)
        print("")
        data_crosstab = pd.crosstab(df[i],
                            df['target'],
                               margins = False)
        chi(data_crosstab)
        print("")
    #df.to_csv("/data/output/processed_data.csv")
    return df
