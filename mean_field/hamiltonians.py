"""Mean field Hamiltonian utilities."""
from __future__ import annotations

import abc
from abc import abstractmethod
import dataclasses

import numpy as np
import scipy as sp


def _get_mk_spectra_symmetry(k_spectra: MeanFieldSpectrum)-> MeanFieldSpectrum:
  """Get the spectra for -k points for a Hamiltonian with symmetry."""
  return MeanFieldSpectrum(
    k_spectra.evals, np.conjugate(k_spectra.evecs), -k_spectra.kpoints
  )


@dataclasses.dataclass
class MeanFieldSpectrum:
  evals: np.ndarray
  evecs: np.ndarray
  kpoints: np.ndarray


class Hamiltonian(abc.ABC):
  """Abstract class for constructing and diagonalizing mean-field Hamiltonians.
  """
  @property
  def brilloin_zone(self): #TODO(YT): remove this property
    return self._brillouin_zone

  @abstractmethod
  def momentum_matrix(self, kx: float, ky: float):
    ...

  @abstractmethod
  def diagonalize(self, kx: float, ky: float):
    ...

  def get_eignenvalue_spectra(
      self,
      kpoints: np.ndarray
  ) -> MeanFieldSpectrum:
    """Compute the eigenvalues and eigenvectors for k points.

    Args:
      kpoints: k points to compute spectra (n_points, 2).

    Returns:
      MeanFieldSpectrum: eigenvalues, eigenvectors, and k points.
    """
    if self.__class__.__bases__[0].__name__ == 'BosonHamiltonian':
      permutation_matrix = np.block([
          [
              np.zeros((self.sublattice, self.sublattice)),
              np.eye(self.sublattice)[::-1]
          ],
          [
              np.eye(self.sublattice),
              np.zeros((self.sublattice, self.sublattice))
          ]
      ])
      eigenvalues = []
      eigenvectors = []
      for kx, ky in kpoints:  # grid points suggest real space, maybe rename?
        eig_vals, eig_vecs = self.diagonalize(kx, ky)
        eig_vals = eig_vals @ permutation_matrix
        eig_vecs = eig_vecs @ permutation_matrix
        eigenvalues.append(eig_vals)
        eigenvectors.append(eig_vecs)
    elif self.__class__.__bases__[0].__name__ == 'ClassicalHamiltonian':
      eigenvalues = []
      eigenvectors = []
      for kx, ky in kpoints:
        eig_vals, eig_vecs = self.diagonalize(kx, ky)
        eigenvalues.append(eig_vals)
        eigenvectors.append(eig_vecs)
    else:
      raise ValueError(
          f'Hamiltonian class {self.__class__.__bases__[0].__name__} \
          not recognized.'
      )
    return MeanFieldSpectrum(
        np.array(eigenvalues), np.array(eigenvectors), kpoints
    )

class BosonHamiltonian(Hamiltonian):
  """Hamiltonian for bosons."""

  def diagonalize(self, kx, ky):
    ham_mat = self.momentum_matrix(kx, ky)
    sign_matrix = np.diag(np.concatenate(
        [np.ones(self.sublattice), -np.ones(self.sublattice)]
    ))  # rho matrix for diagonalizing a bosonic Hamiltonian.
    return sp.linalg.eig(sign_matrix @ ham_mat)


class ClassicalHamiltonian(Hamiltonian):
  """Hamiltonian for classical variables."""

  def diagonalize(self, kx, ky):
    ham_mat = self.momentum_matrix(kx, ky)
    return sp.linalg.eigh(ham_mat)


class BosonHuhHamiltonian(BosonHamiltonian):
  """Vison mean-field Hamiltonian from Huh https://arxiv.org/abs/1106.3330.

  This Hamiltonian is defined in boston basis.
  """

  def __init__(
      self,
      params: dict,
      brillouin_zone = None,
      sublattice: int = 12,
  ):
    self.params = params
    # Check that necessary parameters are provided.
    # TODO(YT):
    assert 't' in params, 't is not provided.'
    assert 't0' in params, 't0 is not provided.'
    assert 'h_tri' in params, 'h_tri is not provided.'
    assert 'h_hex' in params, 'h_hex is not provided.'
    assert 'u' in params, 'u is not provided.'
    assert 'v' in params, 'v is not provided.'

    self._brillouin_zone = brillouin_zone
    self.sublattice = sublattice

  def momentum_matrix(self, kx, ky):
    t = self.params['t']
    t0 = self.params['t0']
    h_tri = self.params['h_tri']
    h_hex = self.params['h_hex']
    u = self.params['u']
    v = self.params['v']
    k = np.array([kx, ky])
    nnn_1 = t * (1. + np.exp(2.j * k @ u) + np.exp(2.j * k @ (u - v)))
    nnn_2 = t * (1. + np.exp(2.j * k @ u) + np.exp(2.j * k @ v))
    nnn_3 = t * (1. + np.exp(2.j * k @ v) + np.exp(2.j * k @ (v - u)))
    nn_1 = t0 * np.exp(2.j * k @ u)
    nn_2 = t0 * np.exp(2.j * k @ v)
    # Define the coupling matrix.
    coupling_mat = np.zeros(
        (self.sublattice, self.sublattice),
        dtype=np.complex128
    )
    coupling_mat[1, 0] = 1.
    coupling_mat[2, 0] = 1.
    coupling_mat[3, 0] = nnn_1
    coupling_mat[7, 0] = nn_1
    coupling_mat[3, 1] = 1.
    coupling_mat[4, 1] = 1.
    coupling_mat[5, 1] = -1.
    coupling_mat[10, 1] = np.conjugate(nn_2)
    coupling_mat[11, 1] = nn_2
    coupling_mat[3, 2] = nn_1 * np.conjugate(nn_2)
    coupling_mat[4, 2] = -1.
    coupling_mat[6, 2] = 1.
    coupling_mat[9, 2] = nn_1
    coupling_mat[11, 2] = np.conjugate(nn_1)
    coupling_mat[7, 3] = 1.
    coupling_mat[8, 4] = 1.
    coupling_mat[11, 4] = nnn_2
    coupling_mat[7, 5] = 1.
    coupling_mat[8, 5] = 1.
    coupling_mat[10, 5] = nnn_3
    coupling_mat[7, 6] = -1. * nn_1 * np.conjugate(nn_2)
    coupling_mat[8, 6] = 1.
    coupling_mat[9, 6] = nnn_1
    coupling_mat[9, 7] = -1.
    coupling_mat[10, 7] = np.conjugate(nn_1) * nn_2
    coupling_mat[9, 8] = 1.
    coupling_mat[10, 8] = 1.
    coupling_mat[11, 8] = 1.
    # Fill in complex conjugate.
    coupling_mat = coupling_mat + np.conjugate(coupling_mat.T)
    # Arrange in bogoliubov block form.  #TODO(YT): rewrite using np.block.
    ham_mat = np.block(
      [[coupling_mat, coupling_mat], [coupling_mat, coupling_mat]]
    )
    # add onsite term.
    field_onsite = np.array(
        [
        h_tri, h_hex, h_hex, h_tri, h_tri, h_tri,
        h_tri, h_hex, h_hex, h_tri, h_tri, h_tri
        ]
    )
    ham_mat += np.diag(np.concatenate([field_onsite, field_onsite]))
    return -1. * ham_mat


class IsingHuhHamiltonian(ClassicalHamiltonian):
  """Vison mean-field hamiltonian from Huh https://arxiv.org/abs/1106.3330.

  This Hamiltonia is defined in the Ising basis for phi4 field theory.

  Note default coupling is ferromagnetic H = - J \sum_{<i, j>} H_ij."""

  def __init__(
      self,
      params: dict,
      brillouin_zone = None,
      sublattice: int = 12,
      t0: float = 1. # nn coupling
  ):
    self.params = params
    self.t0 = t0
    # Check that necessary parameters are provided.
    # TODO(YT):
    assert 't' in params, 't is not provided.'
    assert 'm_tri' in params, 'm_tri is not provided.'
    assert 'm_hex_ratio' in params, 'm_hex/m_tri ratio is not provided.'
    self._brillouin_zone = brillouin_zone
    self.sublattice = sublattice

  def momentum_matrix(self, kx, ky):
    t = self.params['t']
    mass_tri = self.params['m_tri']
    mass_hex = self.params['m_hex_ratio'] * mass_tri
    u = np.array([np.sqrt(3.), 0.])
    v = np.array([np.sqrt(3.) / 2., -3. / 2.])
    k = np.array([kx, ky])
    nnn_1 = t * (1. + np.exp(2.j * k @ u) + np.exp(2.j * k @ (u - v)))
    nnn_2 = t * (1. + np.exp(2.j * k @ u) + np.exp(2.j * k @ v))
    nnn_3 = t * (1. + np.exp(2.j * k @ v) + np.exp(2.j * k @ (v - u)))
    nn_1 = self.t0 * np.exp(2.j * k @ u)
    nn_2 = self.t0 * np.exp(2.j * k @ v)
    # Define the coupling matrix.
    coupling_mat = np.zeros(
        (self.sublattice, self.sublattice),
        dtype=np.complex128
    )
    coupling_mat[1, 0] = 1.
    coupling_mat[2, 0] = 1.
    coupling_mat[3, 0] = nnn_1
    coupling_mat[7, 0] = nn_1
    coupling_mat[3, 1] = 1.
    coupling_mat[4, 1] = 1.
    coupling_mat[5, 1] = -1.
    coupling_mat[10, 1] = -nn_2
    coupling_mat[11, 1] = nn_2
    coupling_mat[3, 2] = nn_1 * np.conjugate(nn_2)
    coupling_mat[4, 2] = -1.
    coupling_mat[6, 2] = 1.
    coupling_mat[9, 2] = nn_1
    coupling_mat[11, 2] = -nn_1
    coupling_mat[7, 3] = 1.
    coupling_mat[8, 4] = 1.
    coupling_mat[11, 4] = nnn_2
    coupling_mat[7, 5] = 1.
    coupling_mat[8, 5] = 1.
    coupling_mat[10, 5] = nnn_3
    coupling_mat[7, 6] = -1. * nn_1 * np.conjugate(nn_2)
    coupling_mat[8, 6] = 1.
    coupling_mat[9, 6] = nnn_1
    coupling_mat[9, 7] = -1.
    coupling_mat[10, 7] = np.conjugate(nn_1) * nn_2
    coupling_mat[9, 8] = 1.
    coupling_mat[10, 8] = 1.
    coupling_mat[11, 8] = 1.
    # Fill in complex conjugate. Minus sign for - J \sum_{<i, j>} H_ij.
    ham_mat =  - (coupling_mat + np.conjugate(coupling_mat.T))
    # add onsite term.
    field_onsite = np.array(
      [
      mass_tri, mass_hex, mass_hex, mass_tri, mass_tri, mass_tri,
      mass_tri, mass_hex, mass_hex, mass_tri, mass_tri, mass_tri
      ]
    ) ** 2
    ham_mat += np.diag(field_onsite)
    return ham_mat
