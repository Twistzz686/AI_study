import pandas as pd
import numpy as np
data = np.array([1,2,3,4,5])
series = pd.Series(data,dtype=float,copy=False)
print(series)