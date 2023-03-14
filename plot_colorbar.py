import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5, 3))
plt.imshow(np.array([[1]]), cmap='cividis')
cbar = plt.colorbar(aspect=2)
cbar.set_ticks([])
cbar.outline.set_visible(False)
plt.show()