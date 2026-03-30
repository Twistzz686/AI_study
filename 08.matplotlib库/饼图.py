import matplotlib.pyplot as plt

sizes = [25,35,20,21]
labels = ['A','B','C','D']
colors = ['gold','yellowgreen','lightcoral','lightskyblue']
plt.pie(sizes,
        explode = [0.5,0,0,0],
        labels = labels,
        colors = colors,
        autopct = '%.1f%%',
        radius = 1)
plt.show()