import pandas as pd
from scipy import stats
data = pd.read_csv()
for i in range(21):
    group1 = data.iloc[:, i]
    group2 = data.iloc[:, i+1]
def normality_test(data):
    stat, p_value = stats.shapiro(data)
    return p_value
p_value_group1 = normality_test(group1)
p_value_group2 = normality_test(group2)
levene_stat, levene_p = stats.levene(group1, group2)
alpha = 0.05
if p_value_group1 > alpha and p_value_group2 > alpha:
    if levene_p > alpha:
        t_stat, p_value = stats.ttest_ind(group1, group2)
        test_type = "t test（Homogeneity of variance）"
    else:
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        test_type = "t' test（Heterogeneity of variance）"
else:
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    test_type = "Mann-Whitney U test"
print(f'test_type: {test_type}')
print(f'p-value: {p_value:.4f}')
if p_value > alpha:
    print('no significance (fail to reject H0)')
else:
    print('significance (reject H0)')
