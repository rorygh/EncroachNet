"""Conductor catenary fitting and vegetation-clearance computation.

Implements docs/architecture.md Stages 5-6. Both photogrammetric MVS and LiDAR
undersample thin wires, so we never trust the raw backprojected wire points
directly -- we cluster them per span/conductor and fit the known catenary model,
then measure clearance against the fitted curve rather than the raw points.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN


@dataclass
class Catenary:
    """z(s) = a * cosh(s / a) + c, in a local frame aligned with the span direction.
    `origin` and `direction` map local (s, z) back to world XYZ.
    """
    a: float
    c: float
    origin: np.ndarray      # (3,) world-space origin of the local frame
    direction: np.ndarray   # (3,) unit vector, span's principal horizontal direction
    s_range: tuple[float, float]

    def sample(self, num_points: int = 200) -> np.ndarray:
        s = np.linspace(self.s_range[0], self.s_range[1], num_points)
        z = self.a * np.cosh(s / self.a) + self.c
        # local frame: s along `direction` (horizontal), z along world-up
        pts = self.origin[None, :] + s[:, None] * self.direction[None, :]
        pts[:, 2] += z - self.origin[2]
        return pts


def cluster_conductor_spans(points: np.ndarray, eps: float = 1.0, min_samples: int = 20) -> list[np.ndarray]:
    """Groups raw powerline points into per-span, per-conductor clusters via DBSCAN
    in 3D. Multiple parallel conductors in the same span will typically separate
    along the axis perpendicular to the corridor direction if `eps` is tuned below
    the inter-conductor spacing (docs/architecture.md Stage 5, step 1).
    """
    if len(points) == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    clusters = [points[labels == lbl] for lbl in sorted(set(labels)) if lbl != -1]
    return clusters


def _fit_local_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal horizontal direction via PCA on the XY-projected points."""
    xy = points[:, :2]
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction_xy = vt[0]
    direction = np.array([direction_xy[0], direction_xy[1], 0.0])
    direction /= np.linalg.norm(direction)
    origin = np.array([centroid[0], centroid[1], points[:, 2].min()])
    return origin, direction


def fit_catenary(points: np.ndarray) -> Catenary:
    """Nonlinear least-squares catenary fit (docs/architecture.md Stage 5, step 3).

    Initializes `a` from the span's sag-to-length ratio (a first-order catenary
    approximation) rather than an arbitrary constant, since Newton-type least
    squares on cosh(s/a) is sensitive to initialization.
    """
    origin, direction = _fit_local_frame(points)
    s = (points[:, :2] - origin[:2]) @ direction[:2]
    z = points[:, 2]

    span_length = s.max() - s.min()
    sag = z.max() - z.min()
    a0 = max((span_length ** 2) / (8 * max(sag, 1e-3)), 1.0)
    c0 = z.min() - a0

    def residuals(params):
        a, c = params
        return z - (a * np.cosh(s / a) + c)

    result = least_squares(residuals, x0=[a0, c0], method="lm", max_nfev=2000)
    a_fit, c_fit = result.x
    return Catenary(a=a_fit, c=c_fit, origin=origin, direction=direction,
                     s_range=(s.min(), s.max()))


def fit_all_conductors(powerline_points: np.ndarray, eps: float = 1.0,
                        min_samples: int = 20) -> list[Catenary]:
    clusters = cluster_conductor_spans(powerline_points, eps, min_samples)
    catenaries = []
    for cluster in clusters:
        try:
            catenaries.append(fit_catenary(cluster))
        except Exception:
            continue  # degenerate cluster (e.g. near-collinear in XY); skip rather than crash the batch
    return catenaries


def clearance_distances(vegetation_points: np.ndarray, catenaries: list[Catenary],
                         samples_per_curve: int = 200) -> np.ndarray:
    """docs/architecture.md Stage 6: d_i = min_j min_s ||x_i - C_j(s)||.

    Approximates the continuous nearest-point-on-curve query with a dense sample
    of each fitted catenary -- accurate to within span_length / samples_per_curve,
    which is well under typical MVCD thresholds (meters) for reasonable span lengths.
    """
    if not catenaries or len(vegetation_points) == 0:
        return np.full(len(vegetation_points), np.inf)

    curve_points = np.concatenate([c.sample(samples_per_curve) for c in catenaries], axis=0)
    # brute-force nearest neighbor; swap for a KD-tree if corridor scale grows large
    diffs = vegetation_points[:, None, :] - curve_points[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return dists.min(axis=1)


def flag_risk_points(vegetation_points: np.ndarray, catenaries: list[Catenary],
                      mvcd_meters: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distances, risk_mask) where risk_mask[i] = distances[i] < mvcd_meters."""
    distances = clearance_distances(vegetation_points, catenaries)
    return distances, distances < mvcd_meters
