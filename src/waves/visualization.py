"""3D surface animation ("moving carpet") of the solution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)


def animate_solution(
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    frames: list[npt.NDArray[np.float64]],
    times: list[float],
    filename: str | Path = "wave_carpet.mp4",
    stride: int = 2,
    fps: int = 30,
) -> FuncAnimation:
    """
    3D surface animation with displacement-based colours and a slowly
    rotating camera. Falls back to a GIF (via Pillow) if FFmpeg is not
    available.
    """
    Xs, Ys = X[::stride, ::stride], Y[::stride, ::stride]
    zmax = max(0.6 * np.abs(frames[0]).max(), 1e-6)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    norm = plt.Normalize(-zmax, zmax)

    def draw(k: int) -> list:
        ax.clear()
        Z = frames[k][::stride, ::stride]
        ax.plot_surface(
            Xs, Ys, Z, cmap=cm.coolwarm, norm=norm,
            rstride=1, cstride=1, linewidth=0,
            antialiased=True, shade=True,
        )
        ax.set_zlim(-zmax, zmax)
        ax.set_xlabel("x", color="w")
        ax.set_ylabel("y", color="w")
        ax.set_title(f"2D wave equation   t = {times[k]:.3f}", color="w", fontsize=13)
        ax.tick_params(colors="w")
        # smooth camera: slow continuous azimuth sweep + gentle elevation
        ax.view_init(
            elev=42 + 8 * np.sin(2 * np.pi * k / len(frames)),
            azim=-60 + 0.35 * k,
        )
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)

    try:
        anim.save(str(filename), writer=FFMpegWriter(fps=fps, bitrate=3000))
        print(f"saved {filename}")
    except Exception as exc:  # ffmpeg unavailable
        print(f"ffmpeg failed ({exc}); falling back to GIF")
        gif = str(filename).rsplit(".", 1)[0] + ".gif"
        anim.save(gif, writer=PillowWriter(fps=fps))
        print(f"saved {gif}")
    plt.close(fig)
    return anim
