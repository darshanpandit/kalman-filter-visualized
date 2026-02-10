"""Animation for visualizing the product of two Gaussians."""

from manim import *
import numpy as np

from kalman_manim.mobjects.gaussian_ellipse import GaussianEllipse
from kalman_manim.utils import gaussian_product_2d
from kalman_manim.style import COLOR_POSTERIOR, NORMAL_ANIM


def animate_gaussian_multiply(
    scene: Scene,
    g1: GaussianEllipse,
    g2: GaussianEllipse,
    result_color: str = COLOR_POSTERIOR,
    run_time: float = NORMAL_ANIM,
    axes=None,
):
    """Animate two GaussianEllipses merging into their product.

    1. Both ellipses pulse briefly to draw attention.
    2. A new posterior ellipse fades in at the product location.

    Parameters
    ----------
    scene : Scene
        The manim scene to play animations on.
    g1, g2 : GaussianEllipse
        The two Gaussian ellipses to multiply.
    result_color : str
        Color for the resulting posterior ellipse.
    run_time : float
        Duration of the merge animation.
    axes : Axes or None
        Coordinate system (passed through to the result ellipse).

    Returns
    -------
    GaussianEllipse
        The newly created posterior ellipse (already added to scene).
    """
    mu_new, cov_new = gaussian_product_2d(g1._mean, g1._cov, g2._mean, g2._cov)

    result = GaussianEllipse(
        mean=mu_new,
        cov=cov_new,
        color=result_color,
        fill_opacity=0.35,
        show_center=True,
        axes=axes,
    )

    # Animate: both inputs pulse, then result fades in
    scene.play(
        g1.animate.set_fill(opacity=g1._fill_opacity * 0.5),
        g2.animate.set_fill(opacity=g2._fill_opacity * 0.5),
        FadeIn(result, scale=0.8),
        run_time=run_time,
    )

    return result
