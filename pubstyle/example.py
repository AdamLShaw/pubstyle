#%%
import numpy as np
import matplotlib.pyplot as plt
import pubstyle as ps
from matplotlib.colors import BoundaryNorm


rng = np.random.default_rng(123)
ps.set_publication_style()
# ps.reset_style()

fig, axs = plt.subplots(2, 2, figsize=ps.size('nature', 60), dpi=600)


# 1) Multi-color errorbars (uses current color cycle)
ax = axs[0, 0]
x = np.arange(10)
for k in range(2):
    y = 0.8*np.sin(0.6*x + k) + 0.2*k
    yerr = 0.15 + 0.1*rng.random(size=x.size)
    ax.errorbar(x, y, yerr=yerr, label=f"series {k+1}")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")

# 2) Log–log plot
ax = axs[0,1]
xx = np.logspace(0, 3, 60)
yy = 1.5 * xx**0.7 * np.exp(0.15*rng.standard_normal(xx.size))
ax.loglog(xx, yy, marker="o", linestyle="-", ms=4)
ax.set_xlabel("f (Hz)")
ax.set_ylabel("S(f)")

# 3) Heatmap with discrete levels + colorbar
ax = axs[1,0]
Nx, Ny = 40, 30
X, Y = np.meshgrid(np.linspace(-3, 3, Nx), np.linspace(-2, 2, Ny), indexing="xy")
Z = np.exp(-((X-0.8)**2 + (Y+0.3)**2)) - 0.6*np.exp(-((X+1.0)**2 + (Y-0.5)**2)) + 0.1*rng.standard_normal((Ny, Nx))
levels = np.linspace(-1.0, 1.0, 11)  
cmap = plt.get_cmap("RdPu_r", len(levels) - 1)
norm = BoundaryNorm(levels, cmap.N)
im = ax.imshow(Z, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], cmap=cmap, norm=norm, aspect="auto")
cbar = fig.colorbar(im, ax=ax, ticks=levels[::5])
cbar.set_label("value")
ax.set_xlabel("x")
ax.set_ylabel("y")


# 4) Histogram + CDF (counts)
ax = axs[1, 1]
data = 100 + 30*rng.standard_normal(500)
n, bins, _ = ax.hist(data, bins=20, color='g')
ax.set_xlabel("finesse")
ax.set_ylabel("count")
# CDF on right axis (counts, not %)
ax2 = ax.twinx()
sorted_d = np.sort(data)
cdf_counts = np.arange(1, sorted_d.size+1)
ax2.plot(sorted_d, cdf_counts, drawstyle="steps-post",lw=0.75,color='k')
ax2.set_ylim(0, cdf_counts.max())
ax2.set_ylabel("cumulative count")


ps.format(fig)

plt.show()