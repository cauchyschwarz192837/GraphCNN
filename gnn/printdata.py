import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("/Users/admin/Downloads/GNN/cgcnn/test_results.csv", delimiter=',')
print(data)

plt.plot(data[:,1], data[:,1])
plt.xlabel('d33,f (pm/V)')
plt.ylabel('d33,f (pm/V)')
plt.scatter(data[:,1], data[:,2])
plt.show()

data = np.arange(0, 8+0.2, 0.2)
print(data)