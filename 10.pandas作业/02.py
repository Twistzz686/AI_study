import pandas as pd
data = {'学生成绩':['张三','李四','王五','赵六'],
        '语文成绩':[85,92,78,88],
        '数学成绩':[90,88,95,82],
        '英语成绩':[78,85,88,90]}
df = pd.DataFrame(data,index=(0,1,2,3))
print(df)
print(df.index)
print(df.columns)
print(df.dtypes)
print(df.head(n=2))
print(df.tail(n=3))
print(df.loc[2,'数学成绩'])
print(df.iloc[2,2])