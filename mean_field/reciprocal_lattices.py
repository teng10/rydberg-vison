"""Reciprocal lattices and first Brillouin zone."""
#TODO(YT): could combine this with the lattice module.

import dataclasses
import functools
import itertools

import numpy as np
import shapely

from mean_field import lattices

def _point_line_distance(
  px: float, py: float, A: float, B: float, C: float
) -> float:
  """Compute the distance between a point and a line.
  
  The line is defined by the equation Ax + By + C = 0.
  Args:
    px, py: point coordinates
    A, B, C: line coefficients
  
  Returns:
    The distance between the point and the line.
  """
  return np.abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)


@dataclasses.dataclass
class ReciprocalDiceLattice:
  size_x: int
  size_y: int
  b1: np.ndarray = np.array([np.pi / np.sqrt(3), - np.pi / 3])
  b2: np.ndarray = np.array([0, 2 * np.pi / 3.])
  bz_center: np.ndarray = np.array([0, 0])
  p_gamma: np.ndarray = np.array([0., 0.]) # Gamma point in Broullin zone.
  p_m = np.array([0., np.pi / 3.])
  p_k = np.array([np.pi / (3. * np.sqrt(3.)), np.pi / 3.])
  
  
  def __post_init__(self):
    self.bz_vertices = np.array(
      [
      [*self.bz_center],
      [*self.b1],
      [*(self.b1+ self.b2)],
      [*self.b2],    
      ]
    )      
    # Generate a regular kpoints within the parallelogram defined by b1 and b2.
    grid_steps =list(
      itertools.product(np.arange(self.size_x), np.arange(self.size_y))
    ) # all possible combinations of steps along b1 and b2
    self.kpoints = np.stack(
      [step[0] * self.b1 / self.size_x  + step[1] * self.b2 / self.size_y 
      for step in grid_steps]
    )
    self.bz_polygon_hexagon = lattices.generate_shapely_hexagon(
        np.linalg.norm(self.b2) / np.sqrt(3.), *self.bz_center
    )
    self.bz_polygon = shapely.geometry.Polygon(self.bz_vertices)

  def find_points_on_line(
    self,
    endpoint1: np.ndarray,
    endpoint2: np.ndarray,
    tolerance: float = 1e-3
  ) -> np.ndarray:
    """Find kpoints on a line, sorted by distance to the first endpoint.
    
    Args:
      endpoint1: first endpoint of the line. 
      endpoint2: second endpoint of the line.
      tolerance: the maximum distance from the line.

    Returns:
      An array of points on the line.
    """
    x1, y1 = endpoint1
    x2, y2 = endpoint2
    points = self.kpoints
    # Calculate line coefficients A, B, C from the endpoints
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    # Calculate distances of all points to the line
    distances = np.array(
        [_point_line_distance(px, py, A, B, C) for px, py in points]
    )
    within_range = (
        (points[:, 0] >= min(x1, x2)) & (points[:, 0] <= max(x1, x2)) \
        & (points[:, 1] >= min(y1, y2)) & (points[:, 1] <= max(y1, y2))
    )
    # Find points where the distance is less than or equal to the tolerance
    on_line = points[(distances <= tolerance) & within_range]
    # Sort based on distance to the first endpoint
    return on_line[np.argsort(np.linalg.norm(on_line - endpoint1, axis=1))]

  def kpoint_to_index(
      self,
      kpoint: np.ndarray,
      expansion: int = 1, 
      tolerance: float = 1e-3
  ) -> int:
    """Convert a kpoint to an index in the reciprocal lattice."""
    expansion_vectors = [
        i * self.b1 + j * self.b2
        for i in range(-expansion, expansion)
        for j in range(-expansion, expansion)
    ]
    expanded_kpoints = np.concatenate(
        [self.translated_kpoints(vector) for vector in expansion_vectors]
    )
    index = np.argmin(np.linalg.norm(expanded_kpoints - kpoint, axis=1))
    norm = np.linalg.norm(expanded_kpoints[index] - kpoint)
    if norm > tolerance:
      raise ValueError(
          f"The {kpoint=} with {norm=} is not in the expanded lattice."
      )
    return index % len(self.kpoints)

  def kpoints_to_indices(
      self,
      kpoints: np.ndarray,
      expansion: int = 1,
      tolerance: float = 1e-3
  ) -> np.ndarray:
    """Convert multiple kpoints to indices in the reciprocal lattice."""
    return np.vectorize(
        functools.partial(
            self.kpoint_to_index, expansion=expansion, tolerance=tolerance
        ), 
        signature='(x)->()'
    )(kpoints)
    
  def index_to_kpoint(self, index: int) -> np.ndarray:
    """Convert an index in the reciprocal lattice to a kpoint."""
    return self.kpoints[index]
  
  def translated_kpoints(self, translation: np.ndarray) -> np.ndarray:
    """Translate all kpoints by a given vector."""
    return self.kpoints + translation
