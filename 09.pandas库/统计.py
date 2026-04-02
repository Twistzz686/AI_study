import pandas as pd
import numpy as np
# df = pd.DataFrame({
#     'A':[1,2,np.nan,4],
#     'B':[5,np.nan,np.nan,8],
#     'C':['foo','bar','baz',np.nan]
# })
# count_per = df.count()
# print(count_per)

# count_per = df.count(axis=1)
# print(count_per)

# count_per = df.count(numeric_only=True)
# print(count_per)

# df = pd.DataFrame({
#     'A':[1,2,np.nan,4],
#     'B':[5,np.nan,np.nan,8],
#     'C':[12,np.nan,np.nan,np.nan]
# })

# sum_per = df.sum()
# print(sum_per)

# sum_per = df.sum(axis=1)
# print(sum_per)

# sum_per = df.sum(numeric_only=True)
# print(sum_per)

# sum_per = df.sum(min_count=2)
# print(sum_per)

# df = pd.DataFrame({
#     'A':[1,2,np.nan,4],
#     'B':[5,np.nan,np.nan,8],
#     'C':['foo','bar','baz',np.nan]
# })

# mean_per = df.mean(numeric_only=True)
# print(mean_per)

# mean_per = df.mean(axis=1,numeric_only=True)
# print(mean_per)

# df = pd.DataFrame({
#     'A':[1,2,np.nan,4],
#     'B':[5,np.nan,7,8],
#     'C':[12,33,1,6]
# })

# median_per =df.median()
# print(median_per)

# median_per = df.median(axis=1)
# print(median_per)

# median_per = df.median(numeric_only=True)
# print(median_per)

# df = pd.DataFrame({
#     'A':[1,2,np.nan,4],
#     'B':[5,np.nan,np.nan,8],
#     'C':['foo','bar','baz',np.nan]
# })

# min_per = df.min()
# print(min_per)

# min_per = df.min(axis=1,numeric_only=True)
# print(min_per)

# df = pd.DataFrame({
#     'A':[1,2,np.nan,4],
#     'B':[5,np.nan,np.nan,8],
#     'C':['foo','bar','baz',np.nan]
# })

# max_per = df.max()
# print(max_per)

# max_per = df.max(numeric_only=True)
# print(max_per)

# df = pd.DataFrame({
#     'A':[1,2,np.nan,4],
#     'B':[5,np.nan,np.nan,8],
#     'C':['foo','bar','baz',np.nan]
# })

# var_per = df.var(numeric_only=True)
# print(var_per)

# std_per =df.std(numeric_only=True,ddof=1)
# print(std_per)


# data = {'col1':[10,20,30,40,50],
#         'col2':[15,25,35,45,55],
#         'col3':[20,30,40,50,60]}
# df = pd.DataFrame(data)
# col_median = df.quantile(0.5)
# print(col_median)

df = pd.DataFrame({
    'A':[3,2,np.nan,4,1],
    'B':[5,np.nan,3,2,6],
    'C':['foo','bar','baz','qux','quux']
})

# cummin_per = df.cummin(skipna=False)
# print(cummin_per)

# cumsum_per = df.cumsum()
# print(cumsum_per)

cumprod_per = df.cumprod(numeric_only=True)
print(cumprod_per)