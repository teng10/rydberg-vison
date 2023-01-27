from scipy.spatial.distance import pdist
import numpy as np
# from scipy.spatial import ckdtree

def condensed_to_pair_indices(n,k):
    x = n-(4.*n**2-4*n-8*k+1)**.5/2-.5
    i = x.astype(int)
    j = k+i*(i+3-2*n)/2+1
    return i.astype(int),j.astype(int)
def close_pairs_pdist(X, max_d):
    d = pdist(X)
    k = (d<max_d).nonzero()[0]
    # return k
    return condensed_to_pair_indices(X.shape[0],k)    

def build_dict_bonds(bonds):
  bonds = np.around(bonds, 2)
  bonds2 = np.flip(bonds, 1)
  length = bonds.shape[0]
  keys = list(bonds)
  keys2 = list(bonds2)
  dict_bonds = {(keys[i]).tobytes(): i for i in np.arange(length).astype(int)}
  dict_bonds2 = {(keys2[i]).tobytes(): i for i in np.arange(length).astype(int)}
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
    
    indices = np.array([bonds_dict[bd_v.tobytes()] for bd_v in bd_vs])
    Qs_loop = np.take_along_axis(Qs, indices, 0)
    # signs_loop = np.take_along_axis(bd_signs, indices, 0)
    Qs_loops_vals.append(np.prod(Qs_loop * bd_signs))
  Q_quad = np.sum(np.abs(Qs)**2)
  Q_quarc = np.sum(np.abs(Qs)**4)
  return - alpha * Q_quad - beta / 2 * Q_quarc + K * sum(Qs_loops_vals)
