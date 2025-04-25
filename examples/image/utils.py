import os
import numpy as np
import skimage


def get_grid_points(coords: list[np.ndarray]) -> np.ndarray:
    return np.vstack([c.ravel() for c in np.meshgrid(*coords, indexing="ij")]).T


def gen_image(key: str, res: int) -> None:
    images = None
    
    if key == "shepp":
        image = skimage.data.shepp_logan_phantom()
        image = image[::-1, :]
        # image = image.T
        
    else:
        filenames = {
            "leaf": "leaf.png",
            "tree": "tree.png",
        }
        assert key in filenames
        
        filename = filenames[key]
        filename = os.path.join("./images", filename)
    
        image = skimage.io.imread(filename, as_gray=True)
        image = 1.0 - image
        image = image[::-1, :]
        image = image.T

    shape = (res, res)
    image = skimage.transform.resize(image, shape, anti_aliasing=True)
    return image


def rec_sart(projections: np.ndarray, angles: np.ndarray, iterations: int = 1) -> np.ndarray:
    image = skimage.transform.iradon_sart(projections.T, theta=-angles)
    for _ in range(iterations - 1):
        image = skimage.transform.iradon_sart(projections.T, theta=-angles, image=image)
    image = image.T
    return image


def rec_fbp(projections: np.ndarray, angles: np.ndarray, iterations: int = 1) -> np.ndarray:
    image = skimage.transform.iradon(projections.T, theta=-angles)
    image = image.T
    return image

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for key in ["shepp", "leaf", "tree"]:
        image = gen_image(key, res=256)
    
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pcolormesh(image.T, cmap="Greys")
        plt.show()