"""Tests for hamiltonian.py."""
import itertools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from mean_field import hamiltonians
from mean_field import lattices
from mean_field import reciprocal_lattices
from mean_field import structure_factor


class StructureFactorTests(parameterized.TestCase):
  """Tests for structure factor computations."""

  def setUp(self):
    """Use Ising Hamiltonian for testing."""
    my_params = {
        't': 0.5, 'm_tri': 2.5, 'm_hex_ratio': 1.,
    }
    visonham = hamiltonians.IsingHuhHamiltonian(my_params) # Hamiltonian
    bz_lattice_size = 100
    bz_dice = reciprocal_lattices.ReciprocalDiceLattice(
        bz_lattice_size, bz_lattice_size
    )
    DiceLattice = lattices.EnlargedDiceLattice() # Create Dice lattice
    q_path_name = 'gamma_m_k_gamma'
    q_steps = 100
    self.points_q = reciprocal_lattices.BZ_PATH_REGISTRY[q_path_name](
        bz_dice, q_steps
    )
    self.sf_cls = structure_factor.DynamicalStructureFactor(
        DiceLattice, bz_dice, visonham
    )
    PHYSICAL_PROPERTIES_REGISTRY = structure_factor.PHYSICAL_PROPERTIES_REGISTRY
    self.sf_fn = PHYSICAL_PROPERTIES_REGISTRY['static_structure_factor']

  def test_structure_factor(self):
    """Tests for structure factor."""
    sf_results = self.sf_fn(self.sf_cls, self.points_q)
    

#TODO(YT): 
# test relabeling of the Hamiltonian spectrum.
# Define hamiltonian parameters
# my_params = {
#     't0': 1, 't': 1., 'm_tri': -10., 'm_hex': -5.,
#     'u': np.array([np.sqrt(3.), 0.]), 'v': np.array([np.sqrt(3.) / 2., -3. / 2.])
# }
# bz_dice = reciprocal_lattices.ReciprocalDiceLattice(100, 100)
# visonham = hamiltonian.IsingHuhHamiltonian(my_params)
# spectrum = visonham.get_eignenvalue_spectra(bz_dice.kpoints)

# spectrum_k = visonham.get_eignenvalue_spectra(bz_dice.kpoints)
# spectrum_k.evecs[new_indices[0]] == spectrum_k.evecs[new_indices][0]
