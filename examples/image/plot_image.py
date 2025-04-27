import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from utils import gen_image

name = None
if len(sys.argv) > 1:
    name = sys.argv[1]

im = gen_image(name)

fig, ax = plt.subplots()
ax.pcolormesh(im.T)
plt.show()
