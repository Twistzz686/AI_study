import matplotlib.pyplot as plt
labels = ['A','B','C','D','E']
values = [23,45,56,78,33]
plt.bar(labels, values,
        width = 0.3,
        bottom = 0)
help(plt.bar)
plt.show()