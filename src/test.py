import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.pyplot
import numpy as np

matplotlib.rcParams.update({'font.size': 12, 'font.family': 'sans'})

x = np.linspace(0, 5, 10)
y = x ** 2

fig, ax = plt.subplots(1, 1)

ax.plot(x, x ** 2, x, np.exp(x))
ax.set_yticks([0, 50, 100, 150])
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1, 1))
# ax.yaxis.set_major_formatter(formatter)
# ax.grid(True)
plt.show()
