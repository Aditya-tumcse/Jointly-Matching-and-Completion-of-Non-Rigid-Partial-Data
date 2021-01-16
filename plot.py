import plotly.graph_objects as go
import numpy as np
import model_data as md
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
t = np.linspace(0,10,50)
cp = np.zeros([50,3])
for i in range(50):
    x,y,z = np.cos(t[i]),np.sin(t[i]),t[i]
    cp[i] = [x,y,z]
    ax.scatter(x,y,z)
plt.show()