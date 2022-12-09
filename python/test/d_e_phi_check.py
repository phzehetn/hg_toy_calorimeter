import numpy as np


z = 3.2
a = 0.319
sh = 0.006

print(np.arctan((a+sh)/z)-np.arctan((a)/z))


print(-np.log(np.arctan((a+sh)/z)/2)+np.log(np.arctan((a)/z)/2))