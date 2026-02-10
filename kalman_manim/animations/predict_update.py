"""Reusable predict-update cycle animation for Kalman filter scenes."""

from __future__ import annotations

from manim import *
import numpy as np

from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse
from kalman_manim.style import (
    COLOR_PREDICTION,
    COLOR_MEASUREMENT,
    COLOR_POSTERIOR,
    NORMAL_ANIM,
    FAST_ANIM,
    PAUSE_SHORT,
    MEASUREMENT_DOT_RADIUS,
)


def animate_predict_step(
    scene: Scene,
    ellipse: GaussianEllipse,
    new_mean: np.ndarray,
    new_cov: np.ndarray,
    run_time: float = NORMAL_ANIM,
):
    """Animate the prediction step: ellipse moves and grows.

    Parameters
    ----------
    scene : Scene
        The manim scene.
    ellipse : GaussianEllipse
        Current estimate ellipse (will be transformed in-place).
    new_mean, new_cov : np.ndarray
        Predicted mean and covariance after F @ x and F @ P @ F.T + Q.
    run_time : float
        Animation duration.
    """
    scene.play(ellipse.animate_to(new_mean, new_cov), run_time=run_time)


def animate_update_step(
    scene: Scene,
    predicted_ellipse: GaussianEllipse,
    measurement: np.ndarray,
    updated_mean: np.ndarray,
    updated_cov: np.ndarray,
    meas_cov: np.ndarray | None = None,
    axes=None,
    run_time: float = NORMAL_ANIM,
):
    """Animate the measurement update step.

    1. Show measurement dot appearing.
    2. Optionally show measurement uncertainty ellipse.
    3. Morph predicted ellipse into posterior.

    Returns
    -------
    Dot
        The measurement dot (caller can remove it later).
    """
    # Measurement dot
    if axes is not None:
        meas_point = axes.c2p(measurement[0], measurement[1])
    else:
        meas_point = np.array([measurement[0], measurement[1], 0])

    meas_dot = Dot(meas_point, radius=MEASUREMENT_DOT_RADIUS, color=COLOR_MEASUREMENT)
    scene.play(FadeIn(meas_dot, scale=1.5), run_time=FAST_ANIM)

    # Optional measurement uncertainty ellipse
    meas_ellipse = None
    if meas_cov is not None:
        meas_ellipse = GaussianEllipse(
            mean=measurement,
            cov=meas_cov,
            color=COLOR_MEASUREMENT,
            fill_opacity=0.15,
            show_center=False,
            axes=axes,
        )
        scene.play(FadeIn(meas_ellipse), run_time=FAST_ANIM)

    # Update: morph predicted → posterior
    scene.play(
        predicted_ellipse.animate_to(updated_mean, updated_cov),
        run_time=run_time,
    )

    # Clean up measurement ellipse
    if meas_ellipse is not None:
        scene.play(FadeOut(meas_ellipse), run_time=FAST_ANIM * 0.5)

    return meas_dot


def animate_full_cycle(
    scene: Scene,
    ellipse: GaussianEllipse,
    pred_mean: np.ndarray,
    pred_cov: np.ndarray,
    measurement: np.ndarray,
    upd_mean: np.ndarray,
    upd_cov: np.ndarray,
    meas_cov: np.ndarray | None = None,
    axes=None,
    predict_time: float = NORMAL_ANIM,
    update_time: float = NORMAL_ANIM,
):
    """Animate one complete predict → update cycle.

    Returns the measurement dot for optional cleanup.
    """
    animate_predict_step(scene, ellipse, pred_mean, pred_cov, run_time=predict_time)
    scene.wait(PAUSE_SHORT)
    meas_dot = animate_update_step(
        scene, ellipse, measurement, upd_mean, upd_cov,
        meas_cov=meas_cov, axes=axes, run_time=update_time,
    )
    return meas_dot
