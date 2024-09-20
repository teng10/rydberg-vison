"""Lattice class and lattice helper functions."""
from __future__ import annotations
import math
import dataclasses
import itertools
from typing import Mapping, Tuple

import numpy as np
import shapely

from mean_field import func_utils


vectorized_method = func_utils.vectorized_method


@dataclasses.dataclass
class Lattice:
  """Lattice class for storing lattice points and performing operations."""
  points: np.ndarray
  decimal_precision: int = 3
  loc_to_idx: Mapping[Tuple[str, ...], int] = dataclasses.field(
      init=False, repr=False)
  n_sites: int = dataclasses.field(init=False)
  ndim: int = dataclasses.field(init=False)

  def __post_init__(self):
    self.n_sites, self.ndim = self.points.shape
    loc_to_idx = {}
    for idx in range(self.n_sites):
      loc_to_idx[self.point_to_key(self.get_point(idx))] = idx
    self.loc_to_idx = loc_to_idx

  @vectorized_method(signature='(),(x)->()')
  def get_idx(self, point: np.ndarray) -> int:
    """Returns lattice index at the `location`."""
    return self.loc_to_idx[self.point_to_key(point.astype(float))]

  def get_point(self, idx: int) -> np.ndarray:
    """Returns lattice coordinates at the `idx`-th site."""
    return self.points[idx]

  def point_to_key(self, r: np.ndarray):
    """Returns a hashable key for the lattice point `r`."""
    return str(tuple(np.round(r, self.decimal_precision)))

  def merge(self, other: Lattice, raise_on_overlap: bool = False) -> Lattice:
    """Returns a new lattice with points from `self` and `other`."""
    def _unique_ordered(arr):
        """Returns unique vectors in `arr` in order of appearance.
        Note: this is important for DMRG in 2 dimension.
        """
        _, unique_indices = np.unique(
            arr, axis=0, return_index=True
        )
        sorted_indices = np.argsort(unique_indices)
        return unique_indices[sorted_indices]

    common_precision = min(self.decimal_precision, other.decimal_precision)
    combined_points = np.concatenate([self.points, other.points])
    combined_points_rounded = np.round(combined_points, common_precision)
    unique_indices = _unique_ordered(combined_points_rounded)
    unique_combined = combined_points[unique_indices]
    if raise_on_overlap and unique_combined.shape != combined_points.shape:
      raise ValueError('Attempting to merge lattice with overlapp.')
    return Lattice(unique_combined, common_precision)

  def shift(self, vector: np.ndarray) -> Lattice:
    """Returns a new lattice shifted by `vector`."""
    new_points = self.points + vector
    return Lattice(new_points, self.decimal_precision)

  def get_expanded_lattice(
      self, 
      Lx: int, 
      Ly: int,
      a1: np.ndarray,
      a2: np.ndarray,
  ) -> Lattice:
    """Constructs expanded lattice (Lx, Ly) given unit cell and lattice vectors.
    Args:
      unit cell: unit cell of the lattice.
      Lx: number of unit cells in x direction.
      Ly: number of unit cells in y direction.
      a1: lattice vector 1.
      a2: lattice vector 2.
    
    Returns: Expanded lattice. 
    """
    unit_cell = Lattice(self.points, self.decimal_precision)
    expanded_lattice = sum(
        unit_cell.shift(a1 * i + a2 * j)
        for i, j in itertools.product(range(Lx), range(Ly))
    )
    return expanded_lattice  
  
  def __add__(self, other):
    return self.merge(other)

  def __radd__(self, other):
    if isinstance(other, int) and other == 0:
      return self
    raise NotImplementedError(f'__radd__ not implemented for {type(other)=}')

  def __eq__(self, other):
    """Returns True if `self` and `other` are equal."""
    return (isinstance(other, Lattice)
            and np.array_equal(self.points, other.points))


class KagomeLattice(Lattice):
  def __init__(
      self,
      a: float = 1.0,
  ):
    """Kagome lattice of lattice constant `a`."""
    unit_cell_points = a * np.array([
        [0.25, 0], [-0.25, 0.], [0., np.sqrt(3) / 4.]
        ]
    ) 
    self.unit_cell = Lattice(unit_cell_points)
    self.a1 = a * np.array([1.0, 0.0])
    self.a2 = a * np.array([1. / 2., np.sqrt(3.0) / 2.])
    self.height = np.sqrt(3) / 4.  # height of the unit cell
    self.points = unit_cell_points

  def get_expanded_lattice(
      self,
      size_x: int,
      size_y: int,        
  ) -> Lattice:
    """Returns a lattice of size `size_x` x `size_y`."""
    expanded_lattice = sum(
        self.unit_cell.shift(self.a1 * i + self.a2 * j)
        for i, j in itertools.product(range(size_x), range(size_y))
    )
    return expanded_lattice
  

class DiceLattice(Lattice):
  def __init__(
      self,
      a: float = 1.0,
  ):
    """Kagome lattice of lattice constant `a`."""
    self.delta1 = a * np.array([np.sqrt(3) / 2., 1. / 2.]) 
    self.delta2 = a * np.array([-np.sqrt(3) / 2., 1. / 2.])
    self.delta3 = a * np.array([0.0, -1.0])
    unit_cell_points = np.array([
        np.array([0.0, 0.0]),
        self.delta3, 
        self.delta1 + self.delta3,
    ])    
    self.a1 = a * np.array([np.sqrt(3), 0.0])
    self.a2 = a * np.array([np.sqrt(3) / 2., 3. / 2.])    
    self.unit_cell = Lattice(unit_cell_points)
    self.points = unit_cell_points

  def get_expanded_lattice(
      self,
      size_x: int,
      size_y: int,        
  ) -> Lattice:
    """Returns a lattice of size `size_x` x `size_y`."""
    expanded_lattice = sum(
        self.unit_cell.shift(self.a1 * i + self.a2 * j)
        for i, j in itertools.product(range(size_x), range(size_y))
    )
    return expanded_lattice
  

class EnlargedDiceLattice(Lattice):
  def __init__(
      self,
      a: float = 1.0,
  ):
    """Kagome lattice of lattice constant `a`."""
    self.delta1 = a * np.array([np.sqrt(3) / 2., 1. / 2.])
    self.delta2 = a * np.array([-np.sqrt(3) / 2., 1. / 2.])
    self.delta3 = a * np.array([0.0, -1.0])
    unit_cell_points = np.array([
        np.array([0.0, 0.0]),
        self.delta1, 
        self.delta3, 
        2. * self.delta1,
        self.delta1 + self.delta3,
        self.delta1 - self.delta2,
        self.delta3 - self.delta2,
        2. * self.delta1 - self.delta2,
        self.delta1 + self.delta3 - self.delta2, 
        2. * self.delta1 + self.delta3 - self.delta2, 
        self.delta1 + 2. * self.delta3 - self.delta2, 
        self.delta1 + self.delta3 - 2. * self.delta2, 
    ])
    self.sublattice = len(unit_cell_points)
    self.unit_cell = Lattice(unit_cell_points)
    self.a1 = a * np.array([2 * np.sqrt(3), 0.0])
    self.a2 = a * np.array([np.sqrt(3), 3.0])
    # self.height = np.sqrt(3) / 4.  # height of the unit cell
    self.points = unit_cell_points

  def get_expanded_lattice(
      self,
      size_x: int,
      size_y: int,        
  ) -> Lattice:
    """Returns a lattice of size `size_x` x `size_y`."""
    expanded_lattice = sum(
        self.unit_cell.shift(self.a1 * i + self.a2 * j)
        for i, j in itertools.product(range(size_x), range(size_y))
    )
    return expanded_lattice

  def eta(self):
    """Definition of sublattice frustrated bond values."""
    eta_mat = np.zeros((self.sublattice, self.sublattice))
    # alpha = 2
    eta_mat[1, 0] = 1.
    eta_mat[1, 3] = 1.
    eta_mat[1, 4] = 1.
    eta_mat[1, 5] = -1.
    eta_mat[1, 10] = -1.
    eta_mat[1, 11] = 1.
    # alpha = 3
    eta_mat[2, 0] = 1.
    eta_mat[2, 3] = 1.
    eta_mat[2, 4] = -1.
    eta_mat[2, 6] = 1.
    eta_mat[2, 9] = 1.
    eta_mat[2, 11] = -1.
    # alpha = 8
    eta_mat[7, 0] = 1.
    eta_mat[7, 3] = 1.
    eta_mat[7, 5] = 1.
    eta_mat[7, 6] = -1.
    eta_mat[7, 9] = -1.
    eta_mat[7, 10] = 1.
    # alpha = 9
    eta_mat[8, 4] = 1.
    eta_mat[8, 5] = 1.
    eta_mat[8, 6] = 1.
    eta_mat[8, 9] = 1.
    eta_mat[8, 10] = 1.
    eta_mat[8, 11] = 1.
    return eta_mat + np.transpose(eta_mat)

  def delta(self):
    """Definition of sublattice displacement tensor."""
    delta_tensor = np.zeros((self.sublattice, self.sublattice, 2))
    # alpha = 2
    delta_tensor[1, 0] = - self.delta1
    delta_tensor[1, 3] = self.delta1
    delta_tensor[1, 4] = self.delta3
    delta_tensor[1, 5] = - self.delta2
    delta_tensor[1, 10] = self.delta2
    delta_tensor[1, 11] = - self.delta3
    # alpha = 3
    delta_tensor[2, 0] = - self.delta3
    delta_tensor[2, 3] = self.delta3
    delta_tensor[2, 4] = self.delta1
    delta_tensor[2, 6] = - self.delta2
    delta_tensor[2, 9] = self.delta2
    delta_tensor[2, 11] = - self.delta1
    # alpha = 8
    delta_tensor[7, 0] = - self.delta2
    delta_tensor[7, 3] = self.delta2
    delta_tensor[7, 5] = -self.delta1
    delta_tensor[7, 6] = - self.delta3
    delta_tensor[7, 9] = self.delta3
    delta_tensor[7, 10] = self.delta1
    # alpha = 9
    delta_tensor[8, 4] = self.delta2
    delta_tensor[8, 5] = - self.delta3
    delta_tensor[8, 6] = - self.delta1
    delta_tensor[8, 9] = self.delta1
    delta_tensor[8, 10] = self.delta3
    delta_tensor[8, 11] = - self.delta2
    return delta_tensor - np.transpose(delta_tensor, (1, 0, 2))


def get_restricted(
    lattice: Lattice,
    polygon: shapely.geometry.Polygon,
)-> Lattice:
  """Returns `lattice` with points restricted to be inside of a`polygon`.

  Args:
    lattice: input lattice.
    polygon: a shapely polygon specifying restricted boundaries.

  Returns:
    A part of `lattice` that is contains only points within the `polygon`.
  """
  new_points = []
  for point in lattice.points:
    shapely_point = shapely.geometry.Point(point)
    if polygon.contains(shapely_point):
      new_points.append(point)
  return Lattice(np.stack(new_points), lattice.decimal_precision)


def generate_shapely_hexagon(length:float, x: float, y: float,
) -> shapely.geometry.Polygon:
  """Generates hexagon centered on (x, y) using shapely.

  Args:
    length: length of the hexagon's edge.
    x: x-coordinate of the hexagon's center.
    y: y-coordinate of the hexagon's center.

  Returns:
    The polygon containing the hexagon's coordinates.
  """
  vertices = [
      [x + math.cos(math.radians(angle)) * length,
       y + math.sin(math.radians(angle)) * length]
       for angle in range(0, 360, 60)
  ]
  return shapely.geometry.Polygon(vertices)


def generate_shapely_rectangle(width:float, height:float, x: float, y: float,
) -> shapely.geometry.Polygon:
  """Generates rectangle centered on (x, y) using shapely.

  Args:
    width: width of the rectangle's edge.
    height: height of the rectangle's edge.
    x: x-coordinate of the rectangle's center.
    y: y-coordinate of the rectangle's center.

  Returns:
    The polygon containing the rectangle's coordinates.
  """
  vertices = [
      [x + width / 2., y + height / 2.],
      [x + width / 2., y - height / 2.],
      [x - width / 2., y - height / 2.],
      [x - width / 2., y + height / 2.],
  ]
  return shapely.geometry.Polygon(vertices)
