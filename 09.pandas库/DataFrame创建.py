import pandas as pd
data_list = ['小明','小红','小刚']
columns = ['姓名']
df =pd.DataFrame(data_list,columns=columns)
print(df)

data_list2 = [['小明',20,95],['小红',18,90],['小刚',22,88]]
columns = ['姓名','年龄','成绩']
df2 = pd.DataFrame(data_list2,columns = columns)
print(df2)