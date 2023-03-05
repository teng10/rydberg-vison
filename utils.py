import numpy as np
from scipy.spatial.distance import pdist

def condensed_to_pair_indices(n,k):   #TODO: replace with scipy.spatial.distance.squareform
    x = n-(4.*n**2-4*n-8*k+1)**.5/2-.5
    i = x.astype(int)
    j = k+i*(i+3-2*n)/2+1
    return i.astype(int),j.astype(int)
def close_pairs_pdist(X, max_d, min_d):
    d = pdist(X)  # build condensed distance matrix
    # find indices of close pairs 
    #  (within max_d and not within min_d)
    k = ((d<max_d) & (min_d<d)).nonzero()[0] 
    # return condensed_to_pair_indices(X.shape[0],k)  
    return np.stack(condensed_to_pair_indices(X.shape[0],k)).T  # return pair indices (N, 2)

def build_dict_bonds(bonds):
  bonds = np.around(bonds, 2)
  bonds2 = np.flip(bonds, 1)
  length = bonds.shape[0]
  keys = list(bonds)
  keys2 = list(bonds2)
  dict_bonds = {str(keys[i]): i for i in np.arange(length).astype(int)}
  dict_bonds2 = {str(keys2[i]): i for i in np.arange(length).astype(int)}
  return {**dict_bonds, **dict_bonds2}

def get_energy(couplings, Q, Q0, loops, bonds_dict):
  alpha = couplings["alpha"]
  beta = couplings["beta"]
  K = couplings["K"]
  Qs = Q0 * Q
  Qs_loops_vals = []
  for l in loops:
    bd_vs = l["bd_vs"]
    bd_signs = l["bd_signs"]
    bd_vs = np.around(bd_vs, 2)
    
    indices = np.array([bonds_dict[str(bd_v)] for bd_v in bd_vs])
    Qs_loop = np.take_along_axis(Qs, indices, 0)
    # signs_loop = np.take_along_axis(bd_signs, indices, 0)
    Qs_loops_vals.append(np.prod(Qs_loop * bd_signs))
  Q_quad = np.sum(np.abs(Qs)**2)
  Q_quarc = np.sum(np.abs(Qs)**4)
  return - alpha * Q_quad - beta / 2 * Q_quarc + K * sum(Qs_loops_vals)

def get_energy_rydberg(couplings, Q, Q0, loops, ryd_bonds, bonds_dict):
  alpha = couplings["alpha"]
  beta = couplings["beta"]
  K = couplings["K"]
  Qs = Q0 * Q
  Qs_loops_vals = []
  for l in loops:
    bd_vs = l["bd_vs"]
    bd_signs = l["bd_signs"]
    bd_vs = np.around(bd_vs, 2)
    
    indices = np.array([bonds_dict[str(bd_v)] for bd_v in bd_vs])
    Qs_loop = np.take_along_axis(Qs, indices, 0)
    # signs_loop = np.take_along_axis(bd_signs, indices, 0)
    Qs_loops_vals.append(np.prod(Qs_loop * bd_signs))
    
  # add rydberg bonds contribution to the energy
  # form is $R_{ij} |Q_i|^2 |Q_j|^2$ for different k-nn bonds (ij) 
  # below we will use a constant coupling `beta` for all bonds
  ryd_bonds_vals = []
  for _, k_bonds_all in enumerate(ryd_bonds): # loop over k-nn rydberg bonds
    # k_bonds is an array of k-nn bonds (M, 2, 2)
    ryd_k_vals = [] # list of rydberg bonds vals for k-nn
    k_bonds_all = np.around(k_bonds_all, 2) # round bonds to 2 decimals
    # loop over bonds in k-nn
    for k_bonds in k_bonds_all:
      # k_bonds is a pair bond (2, 2, 2)
      indices = np.array([bonds_dict[str(bd_v)] for bd_v in k_bonds])
      Qs_ryd_bonds = np.take_along_axis(Qs, indices, 0)
      # TODO make this a test case
      # assert Qs_ryd_bonds.shape == (2, ), "Qs_ryd_bonds shape is not (2, ) for a pair of Rydberg bonds" 
      ryd_k_vals.append(np.prod(np.abs(Qs_ryd_bonds)**2))  # take product of |Q_i|^2 |Q_j|^2
    ryd_bonds_vals.append(np.sum(ryd_k_vals)) # sum over bonds in k-nn and append to list

  # for bd_vs in list(ryd_bonds):
  #   bd_vs = np.around(bd_vs, 2)
    
  #   indices = np.array([bonds_dict[str(bd_v)] for bd_v in bd_vs])
  #   Qs_ryd_bonds = np.take_along_axis(Qs, indices, 0)
    
  #   ryd_bonds_vals.append(np.prod(np.abs(Qs_ryd_bonds)**2))

  Q_quad = np.sum(np.abs(Qs)**2)
  # Q_quarc = np.sum(np.abs(Qs)**4)
  return - alpha * Q_quad - beta / 2 * sum(ryd_bonds_vals) + K * sum(Qs_loops_vals)