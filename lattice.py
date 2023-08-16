#Create a Lattice class?

import dataclasses
import numpy as np
import einops
import math
import shapely
from shapely.geometry.polygon import Polygon

from typing import Any, Callable, Tuple, List
Array = np.ndarray

@dataclasses.dataclass
class Lattice:
  """Holds specification of the lattice.

  Attributes:
    a1: first principal lattice vector.
    a2: second principal lattice vector. 
    lattice_type: lattice geometry.
    polygon: cut for the kagome lattice.
    unit_cell_bases: locations of sites in the unit cell.
  """
  a1: np.ndarray
  a2: np.ndarray
  lattice_type: str
  polygon: Any
  unit_cell_bases: List[np.ndarray]

def create_grid_mesh(params):
  """
  Create a mesh grid of the regtangular grid (nx, ny) (in units of lattice vectors v = nx a_x + ny a_y)
  """
  kx1 = params['x1']
  kx2 = params['x2']
  ky1 = params['y1']
  ky2 = params['y2']
  Nx = params['Nx']
  Ny = params['Ny']
  origin_shift = params['O']

  spacing_x = (kx2 - kx1) / (Nx)
  spacing_y = (ky2 - ky1) / (Ny)
  kx_grid = np.linspace(kx1, kx2 - spacing_x, Nx) + origin_shift[0]   #x grid with shift of ny (not y!)
  ky_grid = np.linspace(ky1, ky2 - spacing_y, Ny) + origin_shift[1]

  #create a mesh
  X, Y = np.meshgrid(kx_grid, ky_grid)

  return X, Y

def create_grid(params: dict) -> Array:
  """Computes x, y coordinates of lattice specified by `params`.
  
  Args:
    params: dictionary specifying lattice details.
  
  Returns:
    [N, 2] array specifying x, y corodinates of N lattice points.
  """
  #create a mesh
  my_x, my_y = create_grid_mesh(params)
  a1 = params['a1']
  a2 = params['a2']
  bases = params['unit_cell_bases']
  #stack the mesh to (Nx, Ny, d=2) dim
  # print(my_x.shape)
  # my_grid = np.stack([my_x * a1 + my_y * a2], axis=-1)
  my_grid = np.stack([my_x, my_y], axis=-1)
  # print(my_grid.shape)

  my_grid = einops.rearrange(my_grid, 'x y d -> (x y) d') #integer grid
  BZ_grid = np.tensordot(my_grid[:, 0], a1, 0) + np.tensordot(my_grid[:, 1], a2, 0) # N points of 2d lattice vector
  all_pts_grid = []
  for basis in bases:
    all_pts_grid.append(BZ_grid + basis)
  return np.concatenate(all_pts_grid, 0)


def convert_to_XY(
    pts: Array)-> Tuple[Array]:
  return zip((pts[:, 0], pts[:, 1]))  

def get_contained_pts_poly(
    pts: Array, 
    poly: Any)-> Array:
  '''
  Util function getting points contained in `polygon` from an array of points `pts`.

  Args:
    pts: an array of points.
    poly: a polygon from shapely. 
  
  Returns:
    An array of points that are contained in `poly`.
  '''
  pts_cut = []
  # check if points live in the polygon region
  for i, pt in enumerate(list(pts)):
    # print(pt.shape)
    poly_pt = shapely.Point(pt)
    if poly.contains(poly_pt):
      pts_cut.append(pt)
  return np.stack(pts_cut, 0)

def is_eb_contained(
    eb_vs: Array, 
    poly: Any)-> bool:
  '''
  Util function to determine if an elementary loop `eb_vs` is contained in polygon `poly`.

  Args:
    eb_vs: [N, 2, 2] array of N loops, each with two vertices indexed in the 1st dimension [:, i, :].
    poly: a polygon from shapely. 
  
  Returns:
    A boolean constant for whether the `eb` are contained in `poly`.
  '''
  
  # check if points live in the polygon region
  for i, bond in enumerate(list(eb_vs)):  # loops through all the bonds in the `eb_vs`
    for pt in list(bond):
      poly_pt = shapely.Point(pt)
      if not poly.contains(poly_pt):
        return False
  return True

def get_contained_els_poly(
    els: List[dict], 
    poly: Any, 
    key = "bd_vs")-> List[dict]:
  '''
  Util function getting all  contained in `polygon` from an array of points `pts`.

  Args:
    els: a list of elementary loops.
    poly: a polygon from shapely. 
    key: key for accessing the vertices [array] of the `els`. 
  
  Returns:
    A list of loops that are contained in `poly`.
  '''
  els_contained = []
  # check if the loops live in the polygon region
  for i, el in enumerate(list(els)):
    eb_vs = el[key]
    if is_eb_contained(eb_vs, poly):
      els_contained.append(el)
  return els_contained

def create_hexagon(l, x, y):
    """
    Create a hexagon centered on (x, y)
    :param l: length of the hexagon's edge
    :param x: x-coordinate of the hexagon's center
    :param y: y-coordinate of the hexagon's center
    :return: The polygon containing the hexagon's coordinates
    """
    c = [[x + math.cos(math.radians(angle)) * l, y + math.sin(math.radians(angle)) * l] for angle in range(0, 360, 60)]
    return Polygon(c)