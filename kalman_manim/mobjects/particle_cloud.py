"""ParticleCloud — visualizes a particle filter's weighted particle set."""

from __future__ import annotations

from manim import *
import numpy as np

from kalman_manim.style import COLOR_PROCESS_NOISE, DOT_RADIUS_SMALL


class ParticleCloud(VGroup):
    """Renders a particle filter's weighted particle set as a swarm of dots.

    Particle opacity/size scales with weight to show importance.

    Parameters
    ----------
    particles : np.ndarray (N, 2)
        Particle positions [x, y].
    weights : np.ndarray (N,)
        Particle weights (will be normalized).
    color : str
        Base color for particles.
    axes : Axes or None
        Coordinate system for conversion.
    max_radius : float
        Maximum dot radius (for highest-weight particle).
    min_radius : float
        Minimum dot radius (for lowest-weight particle).
    max_opacity : float
        Maximum fill opacity.
    min_opacity : float
        Minimum fill opacity.
    max_particles_shown : int
        Cap on visible particles (for performance).
    """

    def __init__(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
        color: str = COLOR_PROCESS_NOISE,
        axes=None,
        max_radius: float = 0.06,
        min_radius: float = 0.02,
        max_opacity: float = 0.9,
        min_opacity: float = 0.2,
        max_particles_shown: int = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._axes = axes
        self._particles = np.array(particles, dtype=float)
        self._weights = np.array(weights, dtype=float)

        # Normalize weights
        w = self._weights / (self._weights.sum() + 1e-300)

        # Subsample if too many particles
        n = len(particles)
        if n > max_particles_shown:
            indices = np.random.choice(n, max_particles_shown, replace=False, p=w)
            particles = particles[indices]
            w = w[indices]
            w /= w.sum()

        # Scale weights to [0, 1] for visual mapping
        w_min, w_max = w.min(), w.max()
        if w_max > w_min:
            w_norm = (w - w_min) / (w_max - w_min)
        else:
            w_norm = np.ones_like(w)

        self.dots = VGroup()
        for i, p in enumerate(particles):
            pos = self._to_scene(p[:2])
            t = w_norm[i]
            radius = min_radius + t * (max_radius - min_radius)
            opacity = min_opacity + t * (max_opacity - min_opacity)
            dot = Dot(pos, radius=radius, color=color, fill_opacity=opacity)
            self.dots.add(dot)

        self.add(self.dots)

    def _to_scene(self, xy):
        if self._axes is not None:
            return self._axes.c2p(xy[0], xy[1])
        return np.array([xy[0], xy[1], 0])

    @staticmethod
    def create_updated(particles, weights, color=COLOR_PROCESS_NOISE,
                        axes=None, **kwargs):
        """Factory method to create a new cloud (for Transform animations)."""
        return ParticleCloud(particles, weights, color=color, axes=axes, **kwargs)
