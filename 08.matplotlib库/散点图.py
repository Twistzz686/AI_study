import matplotlib.pyplot as plt
import numpy as np
x= np.arange(0,3 * np.pi, 0.1)
y = np.sin(x)

colors = y

plt.scatter(x,y,
            s=10,
            c=colors,
            marker='o',
            cmap='viridis',
            vmin=-1,
            vmax=1,
            alpha=0.5,
            linewidths=0.5,
            edgecolors='w')
plt.colorbar()
plt.show()