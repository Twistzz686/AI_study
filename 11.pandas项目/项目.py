import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('./source.xlsx')
df = df.fillna(0)
exam_data = df['exam'].values
attendance_data = df['attendance'].values
finally_data = np.round(exam_data * 0.7 + attendance_data * 0.3)
df['finally'] = finally_data
df['pass'] = df['finally'].apply(lambda x: 'yes' if x >=60 else 'no')
df.to_excel('./source1.xlsx')

# fig = plt.figure()
# pass_count = df['pass'].value_counts()
# plt.pie(pass_count,labels=pass_count.index,autopct='%1.1f%%')
# plt.show()

bins = np.arange(0,111,10)
hist,bin_edges = np.histogram(df['finally'],bins=bins)
print(hist)
print(bin_edges)

fig2 = plt.figure()
bar_width = (bin_edges[1] - bin_edges[0])
plt.bar(bin_edges[:-1],hist,width=bar_width,align='edge')

for i in range(len(hist)):
    if hist[i] > 0:
        plt.text(bin_edges[i] + bar_width / 2,hist[i] + 0.1,str(hist[i]),ha='center')

plt.title('finally')
plt.xlabel('score')
plt.ylabel('number')
plt.xticks(bin_edges[:-1])
plt.show()