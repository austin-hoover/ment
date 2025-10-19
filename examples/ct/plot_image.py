import os
import sys

import matplotlib.pyplot as plt

from utils import gen_image


os.makedirs("outputs/plot_image", exist_ok=True)

name = "tree"
if len(sys.argv) > 1:
    name = sys.argv[1]

im = gen_image(name)

fig, ax = plt.subplots()
ax.pcolormesh(im.T)
plt.savefig(f"outputs/plot_image/fig_{name}.png", dpi=300)
