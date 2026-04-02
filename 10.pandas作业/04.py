import pandas as pd
import numpy as np

data = {
    '日期': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02',
            '2023-01-03', '2023-01-03', '2023-01-04', '2023-01-04'],
    '产品': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    '销售额': [1200, 800, 1500, np.nan, 1100, 950, np.nan, 1300],
    '地区': ['华北', '华东', '华北', '华东', '华北', '华东', '华北', '华东']
}

df = pd.DataFrame(data)
print(df)

missing = df.isnull()
print(missing)

count_per = df.count(numeric_only=True)
print(count_per)

mean_per = df.mean(numeric_only=True)
print(mean_per)
df1 =df.replace(to_replace=np.nan,value=mean_per)
print(df1)

df2 = df.dropna(how='any')
print(df2)

result = df[(df['销售额'] >= 1000) & (df['地区'] == '华北')]
print(result)

data2 = {'日期':['2023-01-05'],
         '产品':['A'],
         '销售额':[1100],
         '地区':['华南']}
df3 = pd.DataFrame(data2)
result2 = pd.concat([df,df3],axis=0)
print(result2)