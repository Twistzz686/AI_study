import pandas as pd
data = {
    '姓名':['张三','李四','王五'],
    '年龄':[28,34,29],
    '城市':['北京','上海','广州']
}
df = pd.DataFrame(data)
df.to_csv('人员信息.csv',index=False,encoding='utf-8')

data = pd.read_excel('./人员信息')
print(data)