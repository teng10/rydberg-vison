import dataclasses
import numpy as np
import einops
import utils
import lattice
from shapely import affinity
from shapely.geometry.polygon import Polygon

from typing import Any, Callable, Tuple, List
Array = np.ndarray


@dataclasses.dataclass
class Visons:
  """Holds specification of the vison excitations.

  Attributes:
    visonBonds_vs: list of [2, 2] array for bonds on the branch cut.
    v_pt: array specifying vison location. 
  """
  visonBonds_vs: List[np.ndarray]
  v_pt: np.ndarray

@dataclasses.dataclass
class Bonds:
  """Holds specification of the lattice.

  Attributes:
    a1: first principal lattice vector.
    a2: second principal lattice vector. 
    lattice_type: lattice geometry.
    polygon: cut for the kagome lattice.
    polygon_eb: list of polygons for small loops definition.
    unit_cell_bases: locations of sites in the unit cell.
    unit_cell_bases_ebs: locations of sites for the small loop.
  """
  polygon_ebs: List[Any]
  unit_cell_bases_ebs: List[np.ndarray]

def _find_all_bonds(pts, max_d=1.1, min_d=0.0):
  '''
  Find all bonds given all points `pts` within a cut-off distance `max_d`.
  Args:
    pts: array of shape [N, 2] for all points.
    max_d: float, cut-off distance.
  Returns:
    bonds: array of shape [M, k, 2] which indicate M sets of bonds.
  '''
  # utility function to find all bonds given all points `pts`
  my_points = utils.close_pairs_pdist(pts, max_d, min_d) # indices of all bonds
  bonds = pts[my_points]
  return np.around(bonds, 2)

def _get_ryd_bonds(
  bonds: np.ndarray, 
  distances: List):
  """
  Utility function to find all bonds for the Rydberg blockade constraint within a cut-off distance `max_d`.
  Args:
    bonds: array of shape [N, 2] for all points.
    distances: list of k-pair distances for the Rydberg blockade constraint.
  Returns:
    bonds: array of shape [k, M] which indicate indices for M sets of bonds.
  """
  # first compute the middle point of each bond
  mids = np.mean(bonds, axis=1)
  # build blockades for each bond
  blockades = []
  for _, (d_max, d_min) in enumerate(distances):
    blockades.append(utils.close_pairs_pdist(mids, d_max, d_min)) # indices of all bonds within the cut-off distance
  # return np.stack(blockades, axis=0) # [k, M]
  return blockades # list of [M] arrays
  
def elemetntaryLoopBowTie(type, anchor, lattice_specs=None):
  """
  Generate coordinates for a pair of triangles (t1, t2), boundary bonds coordinates, and middle bond coordinates.
  `type` is the type of the elementry loop, `anchor` is an array, the coordinate of a chosen vertex
  """
  anchor = np.array(anchor)
  if lattice_specs != None: 
    if lattice_specs.lattice_type == 'kagome':
      height = np.sqrt(3)/2
      if type == 0:
        # anchor at the top of the triangle
        t1 = anchor + np.array([0., -2 / 3 * np.sqrt(3)/2])
        t2 = anchor + np.array([1., -4 / 3 * np.sqrt(3)/2])
        # h = anchor + np.array([0., - 2 * np.sqrt(3)/2])
        bd_vs = anchor + np.array([
            [[0, 0], [1/2, - height]],
            [[1/2, - height], [1, -2 * height]],
            [[1, -2 * height], [3/2, - height]], 
            [[3/2, - height], [1/2, - height]],
            [[1/2, - height], [-1/2, - height]],
            [[-1/2, - height],  [0, 0]],
        ])      # boundary vertices
        mid_vs = anchor + np.array([
            [[-1/2, - height], [1/2, -height]]
        ])
        bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
        bd_signs = np.array([
            -1, -1, 1, -1, -1, 1 # signs of the bond given the orientation
            ]
        )
      if type == 1:
        # anchor is at the tip of the triangle
        t1 = anchor + np.array([-1./2., -1 / 3 * height])
        t2 = anchor + np.array([-1./2., -5 / 3 * height])
        # h = anchor + np.array([-1.5, - height])
        bd_vs = anchor + np.array([
            [[0, 0], [-1/2, - height]],
            [[-1/2, - height], [-1, -2 * height]],
            [[-1, - 2 * height], [0, -2 * height]],
            [[0, -2 * height], [-0.5, - height]], 
            [[-0.5, - height], [-1, 0]],
            [[-1, 0], [0., 0.]]
        ])      # boundary vertices
        mid_vs = anchor + np.array([
            [[-1., 0.], [-1/2, -height]]
        ])
        bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
        bd_signs = np.array([
            -1, -1, 1, 1, 1, 1 
            ]
        )       
      if type == 2:
        # anchor is at the tip of the triangle
        t1 = anchor + np.array([-0.5, 1 / 3 * height])
        t2 = anchor + np.array([-1.5, -1 / 3 * height])
        bd_vs = anchor + np.array([
            [[0, 0], [-1., 0.]],
            [[-1., 0.], [-2, 0.]],
            [[-2., 0.], [-1.5, - height]], 
            [[-1.5, - height], [-1., 0.]],
            [[-1., 0.], [-0.5, height]],
            [[-0.5, height], [0., 0.]],
        ])      # boundary vertices
        mid_vs = anchor + np.array([
            [[-1., 0.], [-0.5, height]]
        ])
        bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
        bd_signs = np.array([
            -1, -1, -1, 1, 1, -1 
            ]
        )      
      if type == 3:
        # type 3 is a hexagon geometry
        # anchor is at the upper right corner of the hexagon
        t1 = anchor + np.array([-0.5, - height])
        t2 = anchor + np.array([-0.5, - height])
        bd_vs = anchor + np.array([
            [[0, 0], [0.5, - height]],
            [[0.5, - height], [0., -2 * height]],
            [[0., -2 * height], [-1., - 2 * height]], 
            [[-1., - 2 * height], [-1.5, - height]],
            [[-1.5, - height], [-1., 0.]],
            [[-1., 0.], [0., 0.]],
        ])      # boundary vertices
        mid_vs = anchor + np.array([
            [[-1., 0.], [-0.5, height]]
        ])
        bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
        bd_signs = np.array([
            -1, -1, -1, 1, 1, 1 
            ]
        )      
      return {"triangles":np.array([t1, t2]), "bd_vs": np.around(bd_vs, 2), "mid_vs": np.around(mid_vs, 2), "bs_ps":np.around(bd_ps, 2), "bd_signs":bd_signs}
  
def get_elementaryLoops(n_lattice_size, lattice_type, lattice_specs=None, ebSpecs=None, elemetntaryLoopfn=elemetntaryLoopBowTie):
  nx, ny = n_lattice_size

  if lattice_specs != None:
    if lattice_specs.lattice_type == 'kagome':
      #create all elementry bonds
      # Redefine the polygon for boundary conditions
      height = np.sqrt(3)/2
      epsilon = 0.01 # to shift the boundary of the polygon slightly so that the region counts correct points within the boundary
      
      grid_params_eb0 = {'x1':0, 'x2':nx, 'y1':0, 'y2':ny, 'Nx':nx, 'Ny':ny, 
      'a1':lattice_specs.a1, 'a2':lattice_specs.a2, 
      'O':(0., 0.), # shifting the whole position of the origin so that the loop does not start below 0-axis
      'unit_cell_bases':[np.array([0, height])]} # at the top of top triangle
      ListEB_comb = []
      num_bases = len(ebSpecs.unit_cell_bases_ebs)
      for i in range(num_bases):
        grid_params_eb0['unit_cell_bases'] = [ebSpecs.unit_cell_bases_ebs[i]] # reset the basis vector for each
        polygon = ebSpecs.polygon_ebs[i]
        points_eb0 = list(lattice.create_grid(grid_params_eb0))
        points_eb0 = lattice.get_contained_pts_poly(points_eb0, polygon)
        ListEBs0 = [elemetntaryLoopfn(i, p, lattice_specs) for p in points_eb0]
        ListEBs0_poly = lattice.get_contained_els_poly(ListEBs0, lattice_specs.polygon) # select the bonds that are fully in the \
        ListEB_comb.append(ListEBs0_poly)
      return ListEB_comb

  elif lattice_type == 'triangular':
    #create all elementry bonds
    # Redefine the polygon for boundary conditions
    height = np.sqrt(3)/2
    epsilon = 0.01 # to shift the boundary of the polygon slightly so that the region counts correct points within the boundary
    polygon = Polygon([(int((nx-1)/2) - epsilon, 0 - epsilon), 
                      ((nx-1) + epsilon, 0 - epsilon), 
                      (int((nx-1)/2)/2 + (nx-1) - 1 - epsilon, int((nx-1)/2) * height - epsilon), 
                      ((nx-1)/2 + int((nx-1)/2) - 1 - epsilon, (nx-1) * height - epsilon), 
                      ((nx-1)/2 - epsilon, (nx-1) * height + epsilon), 
                      (int((nx-1)/2)/2  - epsilon, int((nx-1)/2) * height + epsilon)])

    grid_params_eb1 = {'x1':0, 'x2':nx-1, 'y1':0, 'y2':ny-1, 'Nx':nx-1, 'Ny':ny-1, 
    'a1':np.array([1,0]), 'a2':np.array([1/2, np.sqrt(3)/2]), 'O':(0., 0.),
    'unit_cell_bases':[np.array([0,0])]}
    points_eb1 = list(lattice.create_grid(grid_params_eb1))
    points_eb1 = lattice.get_contained_pts_poly(points_eb1, polygon)

    # Redefine the polygon for boundary conditions
    polygon = Polygon([(int((nx-1)/2) - epsilon, 0 - epsilon), 
                      ((nx-1) + epsilon, 0 - epsilon), 
                      (int((nx-1)/2)/2 + (nx-1) - epsilon, int((nx-1)/2) * height - epsilon), 
                      ((nx-1)/2 + int((nx-1)/2) - epsilon, (nx-1) * height - epsilon), 
                      ((nx-1)/2 - epsilon, (nx-1) * height + epsilon), 
                      (int((nx-1)/2)/2  - epsilon, int((nx-1)/2) * height + epsilon)])


    grid_params_eb0 = {'x1':0, 'x2':nx-2, 'y1':0, 'y2':ny-1, 'Nx':nx-2, 'Ny':ny-1, 
    'a1':np.array([1,0]), 'a2':np.array([1/2, np.sqrt(3)/2]), 'O':(1., 0.),
    'unit_cell_bases':[np.array([0,0])]}
    points_eb0 = list(lattice.create_grid(grid_params_eb0))
    points_eb0 = lattice.get_contained_pts_poly(points_eb0, polygon)

    grid_params_eb2 = {'x1':0, 'x2':nx-1, 'y1':0, 'y2':ny-2, 'Nx':nx-1, 'Ny':ny-2, 
    'a1':np.array([1,0]), 'a2':np.array([1/2, np.sqrt(3)/2]), 'O':(0., 1.),
    'unit_cell_bases':[np.array([0,0])]}
    points_eb2 = list(lattice.create_grid(grid_params_eb2))
    points_eb2 = lattice.get_contained_pts_poly(points_eb2, polygon)
    ListEBs0 = [elemetntaryLoop(0, p) for p in points_eb0]
    ListEBs1 = [elemetntaryLoop(1, p) for p in points_eb1]
    ListEBs2 = [elemetntaryLoop(2, p) for p in points_eb2]
    ListEB_comb = [ListEBs0, ListEBs1, ListEBs2]
    return ListEB_comb
    # return sum(ListEB_comb, [])


def elemetntaryLoop(type, anchor, lattice_specs=None, loop_length=8):
  """
  Generate coordinates for a pair of triangles (t1, t2), boundary bonds coordinates, and middle bond coordinates.
  `type` is the type of the elementry loop, `anchor` is an array, the coordinate of a chosen vertex
  """
  anchor = np.array(anchor)
  if lattice_specs != None: 
    if lattice_specs.lattice_type == 'kagome':
      height = np.sqrt(3)/2
      if loop_length == 8:
        if type == 0:
          # anchor at the top of the triangle
          t1 = anchor + np.array([0., -2 / 3 * np.sqrt(3)/2])
          h = anchor + np.array([0., - 2 * np.sqrt(3)/2])
          bd_vs = anchor + np.array([
              [[0, 0], [1/2, - height]],
              [[1/2, - height], [3/2, - height]],
              [[3/2, - height], [1, -2 * height]],
              [[1, -2 * height], [1/2, - 3 * height]], 
              [[1/2, - 3 * height], [-1/2, - 3 * height]],
              [[-1/2, - 3 * height], [-1, - 2 * height]],
              [[-1, - 2 * height], [-1/2, - height]],
              [[-1/2, - height], [0, 0]],
          ])      # boundary vertices
          mid_vs = anchor + np.array([
              [[-1/2, - height], [1/2, -height]]
          ])
          bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
          bd_signs = np.array([
              -1, 1, -1, -1, -1, 1, 1, 1 # signs of the bond given the orientation
              ]
          )
        if type == 1:
          # anchor is at the tip of the triangle
          t1 = anchor + np.array([-1./2., -1 / 3 * height])
          h = anchor + np.array([-1.5, - height])
          bd_vs = anchor + np.array([
              [[0, 0], [-1/2, - height]],
              [[-1/2, - height], [0, -2 * height]],
              [[0, - 2 * height], [-1, -2 * height]],
              [[-1, -2 * height], [-2, - 2 * height]], 
              [[-2, - 2 * height], [-2.5, - 1 * height]],
              [[-2.5, - 1 * height], [-2., 0.]],
              [[-2., 0.], [-1., 0.]],
              [[-1., 0.], [0., 0.]],
          ])      # boundary vertices
          mid_vs = anchor + np.array([
              [[-1., 0.], [-1/2, -height]]
          ])
          bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
          bd_signs = np.array([
              -1, -1, -1, -1, 1, 1, 1, 1 
              ]
          )       
        if type == 2:
          # anchor is at the tip of the triangle
          t1 = anchor + np.array([-0.5, 1 / 3 * height])
          h = anchor + np.array([-1.5,  height])
          bd_vs = anchor + np.array([
              [[0, 0], [-1., 0.]],
              [[-1., 0.], [-3/2, - height]],
              [[-3/2, - height], [-2, 0.]],
              [[-2., 0.], [-2.5, height]], 
              [[-2.5, height], [-2., 2 * height]],
              [[-2., 2 * height], [-1., 2 * height]],
              [[-1., 2 * height], [-0.5, height]],
              [[-0.5, height], [0., 0.]],
          ])      # boundary vertices
          mid_vs = anchor + np.array([
              [[-1., 0.], [-0.5, height]]
          ])
          bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
          bd_signs = np.array([
              -1, -1, 1, 1, 1, 1, -1, -1 
              ]
          )             
        if type == 3:
          # anchor is at the tip of the triangle
          t1 = anchor + np.array([0., 2 / 3 * height])
          h = anchor + np.array([0., 2 * height])
          bd_vs = anchor + np.array([
              [[0, 0], [-0.5, height]],
              [[-0.5, height], [-1.5, height]],
              [[-1.5, height], [-1., 2 * height]],
              [[-1., 2 * height], [-0.5, 3 * height]], 
              [[-0.5, 3 * height], [0.5, 3 * height]],
              [[0.5, 3 * height], [1., 2 * height]],
              [[1., 2 * height], [0.5, height]],
              [[0.5, height], [0., 0.]],
          ])      # boundary vertices
          mid_vs = anchor + np.array([
              [[-0.5, height], [0.5, height]]
          ])
          bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
          bd_signs = np.array([
              1, -1, 1, 1, 1, -1, -1, -1 
              ]
          )        
        if type == 4:
          # anchor is at the tip of the triangle
          t1 = anchor + np.array([0.5, 1 / 3 * height])
          h = anchor + np.array([1.5,  height])
          bd_vs = anchor + np.array([
              [[0, 0], [0.5, height]],
              [[0.5, height], [0., 2 * height]],
              [[0., 2 * height], [1., 2 * height]],
              [[1., 2 * height], [2., 2 * height]], 
              [[2., 2 * height], [2.5,  height]],
              [[2.5,  height], [2., 0.]],
              [[2., 0.], [1., 0.]],
              [[1., 0.], [0., 0.]],
          ])      # boundary vertices
          mid_vs = anchor + np.array([
              [[0.5, height], [1., 0.]]
          ])
          bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
          bd_signs = np.array([
              1, 1, 1, 1, -1, -1, -1, -1 
              ]
          )                            
        if type == 5:
          # anchor is at the tip of the triangle
          t1 = anchor + np.array([0.5, -1 / 3 * height])
          h = anchor + np.array([1.5,  -height])
          bd_vs = anchor + np.array([
              [[0, 0], [1., 0.]],
              [[1., 0.], [1.5, height]],
              [[1.5, height], [2., 0.]],
              [[2., 0.], [2.5, - height]], 
              [[2.5, - height], [2., -2 * height]],
              [[2., -2 * height], [1., -2 * height]],
              [[1., -2 * height], [0.5, - height]],
              [[0.5, - height], [0., 0.]],
          ])      # boundary vertices
          mid_vs = anchor + np.array([
              [[0.5, - height], [1., 0.]]
          ])
          bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
          bd_signs = np.array([
              1, 1, -1, -1, -1, -1, 1, 1 
              ]
          )                 
        return {"triangles":np.array([t1, h]), "bd_vs": np.around(bd_vs, 2), "mid_vs": np.around(mid_vs, 2), "bs_ps":np.around(bd_ps, 2), "bd_signs":bd_signs}
  
  height = np.sqrt(3)/2
  if type == 0:
    t1 = anchor + np.array([0, 2 / 3 * np.sqrt(3)/2])   # triangles
    t2 = anchor + np.array([1 / 2, 1 / 3 * np.sqrt(3)/2])
    bd_vs = anchor + np.array([
        [[0, 0], [1, 0]],
        [[1, 0], [1/2, height]],
        [[1/2, height], [-1/2, height]],
        [[-1/2, height], [0, 0]]
        ])
    mid_vs = anchor + np.array([
        [[0, 0], [1/2, height]]
    ])
    bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
    bd_signs = np.array([
        1, 1, -1, -1
        # [1], [1], [-1], [-1]
        ])
  if type == 1:
    t1 = anchor + np.array([1 / 2, 1 / 3 * np.sqrt(3)/2])
    t2 = anchor + np.array([1, 2 / 3 * np.sqrt(3)/2])
    bd_vs = anchor + np.array([
        [[0, 0], [1, 0]],
        [[0, 0], [1/2, height]],
        [[1, 0], [3/2, height]], 
        [[1/2, height], [3/2, height]]
    ])      # boundary vertices
    mid_vs = anchor + np.array([
        [[1, 0], [1/2, height]]
    ])
    bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
    bd_signs = np.array([
        1, -1, 1, -1
        # [1], [1], [-1], [-1]
        ]
    )
  if type == 2:
    t1 = anchor + np.array([1 / 2, 1 / 3 * np.sqrt(3)/2])
    t2 = anchor + np.array([1 / 2, -1 / 3 * np.sqrt(3)/2])
    bd_vs = anchor + np.array([
        [[0., 0.], [1./2., - height]],
        [[1./2., - height], [1., 0.]],
        [[1., 0.], [1./2., height]], 
        [[1./2., height], [0., 0.]]
    ])      # boundary vertices
    mid_vs = anchor + np.array([
        [[0, 0], [1, 0]]
    ])
    bd_ps = np.mean(bd_vs, axis=1, keepdims=False)
    bd_signs = np.array([
        -1, 1, 1, -1
        # [1], [1], [-1], [-1]
        ]
    ) 
  return {"triangles":np.array([t1, t2]), "bd_vs": np.around(bd_vs, 2), "mid_vs": np.around(mid_vs, 2), "bs_ps":np.around(bd_ps, 2), "bd_signs":bd_signs}


def check_merge(loop, eb):
  """
  This function checks if we should merge the two ojects L and eb. 
  `loop` is a general loop.
  `eb` is an elementary bond, where there are four boundaries. 
  """
  def _check_common_coord(a, b):
    # Takes two arrays a_ij and b_kj each 2d array with last common dimension j (=2 for our case).
    # Return the absolute of the sum of the difference for each pairs (ik)
    dim_a = a.shape[0]
    dim_b = b.shape[0]
    a = np.around(a, 2)
    b = np.around(b, 2)
    a2 = einops.repeat(a, 'a b -> a new_axis b ', new_axis=dim_b)
    b2 = einops.repeat(b, 'a b -> new_axis a b', new_axis=dim_a)
    return np.sum(np.abs(einops.rearrange(a2 - b2, 'a b c -> (a b)c')), axis=1)
  
  def _merge_loops(loop, eb):
    """
    Merge loop and eb: remove repeated bd_v and add that to `mid_vs`.
    """
    loop_merge = jax.tree_util.tree_map(lambda x, y: np.concatenate([x, y], axis=0), loop, eb)
    bd_vs = loop_merge["bd_vs"]
    mid_vs = loop_merge["mid_vs"]
    bd_signs = loop_merge["bd_signs"]
    bd = np.mean(bd_vs, axis=1, keepdims=False)
    # middle = np.mean(mid_vs, axis=1, keepdims=False)
    unq, count = np.unique(bd, axis=0, return_counts=True)
    # print(unq, count)
    repeated_ars = unq[count>1]
    repeated_indices = [np.argwhere(np.all(bd == repeated_ar, axis=1)) for repeated_ar in repeated_ars]
    # for repeated_ar in repeated_ars:
    #   repeated_idx = np.argwhere(np.all(bd == repeated_ar, axis=1))   # Find which the index of the pairs that is repeated
    #   repeated_indices.append()
    repeated_idx_flatten = np.ndarray.flatten(np.array(repeated_indices))
    repeated_idx_sorted = sorted(repeated_idx_flatten, reverse=True) 
    for repeated_idx in repeated_idx_sorted:
      bd_vs = np.delete(bd_vs, repeated_idx, axis=0)    # new boundary vertices
      bd_ps = np.mean(bd_vs, axis=1, keepdims=False)    # new boundary points
      bd_signs = np.delete(bd_signs, repeated_idx, axis=0)    # new boundary bd_signs
      # repeated_idx = einops.repeat(repeated_idx, 'a b -> a (repeat b)', repeat=2)
      # repeated_idx = einops.repeat(repeated_idx, 'a b -> a b c', c=2)
      # repeated_bd_v = np.take_along_axis(bd_vs, repeated_idx, axis=0)
      # repeated_vs_unq = np.unique(repeated_bd_v, axis=0, return_counts=False)


      # print(repeated_bd_v.shape)
      # print(mid_vs.shape)
      # repeated_bd_v = bd_vs[repeated_idx]
      # repeated_ar = einops.rearrange(repeated_bd_v, 'b c-> 1 b c')
      # print(repeated_bd_v)
      # print(f"the repeated vs are {repeated_bd_v}")
      
      # print(f"the unique repeated vs are {repeated_vs_unq}")
      # print(repeated_vs_unq)
      # print(repeated_ar.shape)
      # mid_vs = np.concatenate([mid_vs, repeated_vs_unq], axis=0)    #todo: finish marging mid_vs
    return {"bd_vs": bd_vs, "mid_vs": mid_vs, "triangles": loop_merge["triangles"], "bs_ps":bd_ps, "bd_signs":bd_signs}    

  t_l = loop["triangles"]
  t_eb = eb["triangles"]
  bd_vs_l = loop["bd_vs"]
  bd_vs_eb = eb["bd_vs"]
  boundaries_l = np.mean(bd_vs_l, axis=1, keepdims=False)
  boundaries_eb = np.mean(bd_vs_eb, axis=1, keepdims=False)

  t_diff = _check_common_coord(t_l, t_eb) # sum abs difference in the pairs of coordinates 
  if t_diff.shape[0] > np.count_nonzero(t_diff):  # if there are zeroes (overlapping triangles)
    return False, {}
  else:
    boundaries_diff = _check_common_coord(boundaries_l, boundaries_eb)
    if boundaries_diff.shape[0] == np.count_nonzero(boundaries_diff):    # if there are no zeroes (overlapping boundaries)
      return False, {}
    else:
      # if (boundaries_diff.shape[0] - np.count_nonzero(boundaries_diff)) % 2 == 0:
      merge_loop = _merge_loops(loop, eb)
      return True, merge_loop
      # else:
      #   return False, {}

