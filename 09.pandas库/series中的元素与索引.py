import pandas as pd
series = pd.Series([10,20,30,40,50])
print(series)
print(series[0])
print(series[2])

series1 = pd.Series([10,20,30,40,50],index = ['a','b','c','d',0])
print(series1['a'])
print(series1['c'])
print(series1[0])
print(series1['b':'d'])