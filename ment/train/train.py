import os
import time

from typing import Callable
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm as tqdm_nb
from tqdm import tqdm

from ..core import MENT
from ..utils import ListLogger


class Trainer:
    def __init__(
        self,
        model: MENT,
        plot_func: Callable = None,
        eval_func: Callable = None,
        output_dir: str = None,
        notebook: bool = False,
    ) -> None:

        self.model = model
        self.plot = plot_func
        self.eval = eval_func
        self.notebook = notebook

        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

            self.fig_dir = os.path.join(self.output_dir, f"figures")
            os.makedirs(self.fig_dir, exist_ok=True)

            self.checkpoint_dir = os.path.join(self.output_dir, f"checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_filename(self, filename: str, epoch: int, ext: str = None) -> str:
        filename = f"{filename}_{epoch:03.0f}"
        if ext is not None:
            filename = f"{filename}.{ext}"
        return filename

    def get_progress_bar(self, length):
        if self.notebook:
            return tqdm_nb(total=length)
        else:
            return tqdm(total=length)

    def plot_model(self, epoch: int, **savefig_kws) -> None:
        if self.plot is None:
            return

        ext = savefig_kws.pop("ext", "png")

        for index, fig in enumerate(self.plot(self.model)):
            if self.output_dir is not None:
                path = self.get_filename(f"fig_{index:02.0f}", epoch, ext=ext)
                path = os.path.join(self.fig_dir, path)

                print(f"Saving file {path}")
                fig.savefig(path, **savefig_kws)

            if self.notebook:
                plt.show()

            plt.close("all")

    def eval_model(self, epoch: int) -> None:
        if self.eval == False:
            return {}

        if self.output_dir is not None:
            path = self.get_filename("model", epoch, ext="pt")
            path = os.path.join(self.checkpoint_dir, path)

            print(f"Saving file {path}")
            self.model.save(path)

        if self.eval is not None:
            return self.eval(self.model)

    def train(self, epochs: int, savefig_kws: dict = None, **kws) -> None:
        """Run Gauss-Seidel relaxation algorithm."""

        if savefig_kws is None:
            savefig_kws = {}
        savefig_kws.setdefault("dpi", 300)

        path = None
        if self.output_dir is not None:
            path = os.path.join(self.output_dir, "history.pkl")
        logger = ListLogger(path=path)

        start_time = time.time()

        for epoch in range(epochs + 1):
            if epoch > 0:
                print("epoch = {}".format(epoch))
                self.model.gauss_seidel_step(**kws)

            # Log info.
            # (I think `eval_model` should return a dict with the data fit error and
            # the statistical distance from the true distribution. Then we can
            # print those numbers here. Same goes for `Trainer`.)
            info = dict()
            info["epoch"] = epoch
            info["time"] = time.time() - start_time
            info["D_norm"] = None
            logger.write(info)

            self.plot_model(epoch, **savefig_kws)

            result = self.eval_model(epoch)
            pprint(result)
