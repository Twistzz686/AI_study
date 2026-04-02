import csv
import pandas as pd


# 定义数据
data = [
    ['日期', '产品', '销售额', '数量', '地区'],
    ['2023-01-01', 'A', '1200', '10', '华北'],
    ['2023-01-01', 'B', '800', '8', '华东'],
    ['2023-01-02', 'A', '1500', '12', '华北'],
    ['2023-01-02', 'B', '', '5', '华东'],  # 空值用空字符串
    ['2023-01-03', 'A', '1100', '9', '华北'],
    ['2023-01-03', 'B', '950', '10', '华东'],
    ['2023-01-04', 'A', '', '11', '华北'],
    ['2023-01-04', 'B', '1300', '13', '华东']
]

# 写入 CSV 文件
with open('sales_data.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV 文件已保存为 'sales_data.csv'")

df =pd.read_csv('sales_data.csv',nrows=6)
print(df)
print(df.count())
print(df.sum())
print(df.mean(numeric_only=True))
print(df.median(numeric_only=True))
print(df.min(numeric_only=True))
print(df.max(numeric_only=True))

df.to_excel('sales_analysis.xlsx')