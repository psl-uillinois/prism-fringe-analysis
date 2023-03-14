import numpy as np
import matplotlib.pyplot as plt

# Selectable by user
data_file = "output_index.csv"
x_axis = "Laser power (%)"
y_axis = "Refractive index"

prism_data = np.transpose(np.genfromtxt(data_file, delimiter=' '))

if np.shape(prism_data)[0] == 3:
    plt.errorbar(prism_data[0], prism_data[1], prism_data[2], linestyle='None', marker='o')
elif np.shape(prism_data)[0] == 2:
    plt.scatter(prism_data[0], prism_data[1])
else:
    print("Error: Prism data malformed")
    exit(1)


plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title("Prism data")

plt.show()
