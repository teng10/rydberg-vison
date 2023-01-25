def elemetntaryLoop(type, anchor):
  """
  Generate coordinates for a pair of triangles (t1, t2), boundary bonds coordinates, and middle bond coordinates.
  `type` is the type of the elementry loop, `anchor` is an array, the coordinate of a chosen vertex
  """
  anchor = np.array(anchor)
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

