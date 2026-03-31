import pandas as pd
import numpy as np
# df = pd.DataFrame({'A':[1,2,3,4,5],'B':['a','b','a','b','a']})
# print(df)

# df_replace = df.replace(to_replace=1,value=100)
# print(df_replace)

# df_replace = df.replace(to_replace=[2,3,'a'],value='z')
# print(df_replace)

# df_replace = ({2:200,'b':'y'})
# df_replace = df.replace(to_replace=df_replace)
# print(df_replace)

# df = pd.DataFrame({'col1':['apple','banana','cherry','agerape','apricote'],'col2':['apple pie','banana split','cherry tart','grape juice','apricote jam']})
# df_replace = df.replace(to_replace=r'^a.*e$',value='fruit',regex=True)
# print(df_replace)

# df = pd.DataFrame({
#     'A':[1,2,3],'B':[4.5,5.5,6.5],'C':['7','8','9']
# })
# print(df)
# c = df['A'].astype(float)
# print(c)

# c=df.astype({'B':int,'C':int})
# print(c)

# df = pd.DataFrame({'col1':['A','A','B',np.nan,'D','C'],
#                    'col2':[2,1,9,8,7,4],
#                    'col3':[0,1,9,4,2,3],
#                    'col4':['a','B','c','D','e','F']})
# print(df)
# res1 = df.sort_values(by=['col1'])
# print(res1)

# arrays = [np.array(['qux','qux','foo','foo']),np.array(['two','one','two','one'])]
# df = pd.DataFrame({'A':[1,2,3,4],'B':[4,3,2,1]},index=arrays)
# print(df)
# df_sorted = df.sort_index(level=0)
# print(df_sorted)

# df_sorted = df.sort_index(level=1,ascending=False)
# print(df_sorted)

# df_sorted = df.sort_index(ascending=True)
# print(df_sorted)

df1 = pd.DataFrame({'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']},index=[0,1,2,3])
df2 = pd.DataFrame({'A':['A4','A5','A6','A7'],
                   'B':['B4','B5','B6','B7'],
                   'C':['C4','C5','C6','C7'],
                   'F':['F4','F5','F6','F7']},index=[4,5,6,7])
result = pd.concat([df1,df2])
print(result)