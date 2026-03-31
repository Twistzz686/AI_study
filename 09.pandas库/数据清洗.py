import pandas as pd
import numpy as np
# df =pd.DataFrame({'A':[1,2,np.nan],'B':[4,np.nan,6],'C':[7,8,9]})
# print(df)
# missing_values = df.isnull()
# print(missing_values)
#
# df_cleaned = df.dropna()
# print(df_cleaned)

# df = pd.DataFrame({'A':[1,2,np.nan],'B':[np.nan,np.nan,6],'C':[7,np.nan,9]})
# print(df)
# df_filled_value = df.fillna(value=0)
# print(df_filled_value)

# df_fill =df.ffill()
# print(df_fill)

# data = {'A':'a','B':'b','C':'c'}
# df_fill = df.fillna(value = data)
# print(df_fill)

# fill_limit = df.fillna(value=0,limit=1)
# print(fill_limit)

df = pd.DataFrame({'A':[1,1,2,2,3,3],'B':[1,1,2,2,3,3],'C':[1,1,2,2,3,3]})
print(df)
# df_dell = df.drop_duplicates()
# print(df_dell)

df_dell = df.drop_duplicates(subset=['A'])
print(df_dell)
