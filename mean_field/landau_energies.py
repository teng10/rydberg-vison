"""Landau energy functionals for gauge fields."""
import itertools

import numpy as np

from mean_field import node_collections

def compute_energy_loop(
    couplings: dict,
    Qs: np.ndarray,
    Qs_signs: np.ndarray,
    all_loops_nodes: node_collections.NodesCollection,
    nodes_nn: node_collections.NodesCollection,
) -> float:
  """"Laundau energy functional for Z2 gauge fields ('Qs').
 
      For instance, the energy functional for the Z2 gauge fields is given by:
      E = - alpha * sum(Qs**2) - beta/2 * sum(Qs**4) + k * sum(Qs_loops_vals)
      where Qs_loops_vals are the product of the gauge fields around a loop.
      
      Args:
        couplings: dictionary of couplings in the Landau functional.
        Qs: array of gauge fields. Default is all real and positive.
        Qs_signs: signs of the gauge fields.
        all_loops_nodes: loops contributions to the energy functional.
        nodes_nn: nearest neighbors of the lattice.
    
      Returns:
        Energy functional of the gauge fields ('Qs').
  """
  alpha = couplings["alpha"]
  beta = couplings["beta"]
  k = couplings["K"]
  Qs_loops_vals = []
  edges = node_collections.extract_edge_sequence(all_loops_nodes)
  for edge_l in edges:
    indices = nodes_nn.idx_from_sites(edge_l)  # Extract the indices of edge
    indices = indices % len(Qs)  # Qs are the same for real field for both directions
    Q_loop = np.take_along_axis(Qs, indices, 0)
    Q_signs = np.take_along_axis(Qs_signs, indices, 0)
    Qs_loops_vals.append(np.prod(Q_loop * Q_signs))
  Q_quad = np.sum(np.abs(Qs)**2)
  Q_quarc = np.sum(np.abs(Qs)**4)
  return - alpha * Q_quad - beta / 2 * Q_quarc + k * sum(Qs_loops_vals)


def compute_energy_loop_rydberg(
    couplings: dict,
    Qs: np.ndarray,
    Qs_signs: np.ndarray,
    all_loops_nodes,
    nodes_nn,
    nodes_nn_both_directions
):
  """Laundau energy functional for Z2 gauge fields with Rydberg blockade.
  
  Args:
    couplings: dictionary of couplings.
    Qs: array of gauge fields. Default is all real and positive.
    Qs_signs: array of signs of the gauge fields.
    all_loops_nodes: loops contributions to the energy functional.
    nodes_nn: nearest neighbors of the lattice.
    nodes_nn_both_directions: nearest neighbors for both directions.

  """
  delta = couplings['delta']
  v0 = couplings['v0']
  k = couplings['K']
  Qs_loops_vals = []
  edges_loops = node_collections.extract_edge_sequence(all_loops_nodes)

  for edge_l in edges_loops:
    indices = nodes_nn_both_directions.idx_from_sites(edge_l)  # Extract the indices of edge
    indices = indices % len(Qs)  # Qs are the same for real field for both directions
    Q_loop = np.take_along_axis(Qs, indices, 0)
    Q_signs = np.take_along_axis(Qs_signs, indices, 0)
    Qs_loops_vals.append(np.prod(Q_loop * Q_signs))

  # build B field pairs from Rydberg blockade
  blockade_energy = 0
  for site in range(nodes_nn.lattice.n_sites):
    edges_indices = np.where(nodes_nn.nodes == site)[0]
    edges_rydberg = nodes_nn.nodes[edges_indices]
    b_bond_indices = nodes_nn_both_directions.idx_from_sites(edges_rydberg)  # Extract the indices of edge
    b_bond_indices_pair = list(itertools.combinations(b_bond_indices, 2))
    for b_bond_pair in b_bond_indices_pair:
      Q_pair = Qs[(np.array(b_bond_pair) % len(Qs))]
      blockade_energy += np.prod(Q_pair ** 2)    
  Q_quad = np.sum(np.abs(Qs)**2)

  return - delta * Q_quad + v0 / 2. * blockade_energy + k * sum(Qs_loops_vals)
