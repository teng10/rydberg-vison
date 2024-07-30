"""Tests for hamiltonian.py."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from mean_field import hamiltonian


class HamiltonianTests(parameterized.TestCase):
  """Tests for hamiltonians and their diagonalizations."""

  def setUp(self):
    """Use Ising Hamiltonian for testing."""
    my_params = {
        't0': 1, 't': 0.1, 'm_tri': 0., 'm_hex': 0.,
        'u': np.array([np.sqrt(3.), 0.]),
        'v': np.array([np.sqrt(3.) / 2., -3. / 2.])
    }
    self.visonham = hamiltonian.IsingHuhHamiltonian(my_params) # Hamiltonian

  def test_hermicity(self):
    """Tests for hermicity of the hamiltonian."""
    q_vec = np.random.rand(2)
    ham_mat = self.visonham.momentum_matrix(*q_vec)
    np.testing.assert_allclose(ham_mat, np.conj(np.transpose(ham_mat)))
  
  def test_eigvals(self):
    """Tests for eigenvalues of the hamiltonian."""
    q_vec = np.array([0., 1.])
    actual_eigvals = self.visonham.get_eignenvalue_spectra([q_vec]).evals
    expected_eigvals = np.array([
        -2.46874, -2.46874, -2.46874, -2.46874, 0.0346146, 0.0346146, \
        0.0346146, 0.0346146, 2.43412, 2.43412, 2.43412, 2.43412
    ])
    np.testing.assert_allclose(
        actual_eigvals[0, :], expected_eigvals, atol=1e-5
    )

    q_vec = np.array([1., 1.3])
    actual_eigvals = self.visonham.get_eignenvalue_spectra([q_vec]).evals
    expected_eigvals = np.array([
        -2.47837, -2.47837, -2.47837, -2.47837, 0.0521981, 0.0521981, \
        0.0521981, 0.0521981, 2.42618, 2.42618, 2.42618, 2.42618
    ])
    np.testing.assert_allclose(
        actual_eigvals[0, :], expected_eigvals, atol=1e-5
    )    

  def test_mk_spectra(self):
    """Tests for getting -k spectra from k spectra using symmetry.
    
    The symmetry of the hamiltonian is H_{-k} = H_{k}^*.
    """
    q_vec = np.array([1., 1.3])
    ham_mat = self.visonham.momentum_matrix(*q_vec)
    ham_mat_mk = self.visonham.momentum_matrix(*(-q_vec))
    np.testing.assert_allclose(ham_mat_mk, np.conj(ham_mat))
    spectrum = self.visonham.get_eignenvalue_spectra(q_vec[None, ...])
    actual_spectrum_mk = self.visonham.get_eignenvalue_spectra(
        -q_vec[None, ...]
    )
    expected_spectrum_mk = hamiltonian._get_mk_spectra_symmetry(spectrum)
    np.testing.assert_allclose(
        actual_spectrum_mk.evals, expected_spectrum_mk.evals
    )
    np.testing.assert_allclose(
        actual_spectrum_mk.evecs, expected_spectrum_mk.evecs
    )

if __name__ == '__main__':
  absltest.main()    
