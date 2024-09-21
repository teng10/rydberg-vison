"""Mean field Hamiltonian utilities."""
from __future__ import annotations

import abc
from abc import abstractmethod
import dataclasses
import functools

import numpy as np
import jax.numpy as jnp
import scipy as sp


HAMILTONIAN_REGISTRY = {}

def _register_ham_fn(get_ham_fn, name: str):
  """Registers `get_ham_fn` in global `HAMILTONIAN_REGISTRY`."""
  registered_fn = HAMILTONIAN_REGISTRY.get(name, None)
  if registered_fn is None:
    HAMILTONIAN_REGISTRY[name] = get_ham_fn
  else:
    if registered_fn != get_ham_fn:
      raise ValueError(f'{name} is already registerd {registered_fn}.')


register_ham_fn = lambda name: functools.partial(
    _register_ham_fn, name=name
)

@register_ham_fn('ising_pi_flux')
def get_ising_pi_flux(
    ham_params: dict
) -> ClassicalHamiltonian:
  """Generate Ising Hamiltonian with pi flux gauge field configurations.
  
  Args:
    ham_params: Hamiltonian parameters.
  """
  return IsingHuhHamiltonian(ham_params)

@register_ham_fn('ising_zero_flux')
def get_ising_pi_flux(
    ham_params: dict
) -> ClassicalHamiltonian:
  """Generate Ising Hamiltonian with zero flux gauge field configurations.
  
  Args:
    ham_params: Hamiltonian parameters.
  """
  return IsingZeroFluxHamiltonian(ham_params)

def _get_mk_spectra_symmetry(k_spectra: MeanFieldSpectrum)-> MeanFieldSpectrum:
  """Get the spectra for -k points for a Hamiltonian with symmetry."""
  return MeanFieldSpectrum(
    k_spectra.evals, jnp.conjugate(k_spectra.evecs), -k_spectra.kpoints
  )


@dataclasses.dataclass
class MeanFieldSpectrum:
  evals: jnp.ndarray
  evecs: jnp.ndarray
  kpoints: jnp.ndarray


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
        jnp.array(eigenvalues), jnp.array(eigenvectors), kpoints
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
    return jnp.linalg.eigh(ham_mat)


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
    # Yejin's convention
    # u = np.array([3. / 2., np.sqrt(3.) / 2.])
    # v = np.array([0., - np.sqrt(3)])    
    # Our convention      
    u = np.array([np.sqrt(3.), 0.])
    v = np.array([np.sqrt(3.) / 2., -3. / 2.])
    k = jnp.array([kx, ky])
    nnn_1 = t * (1. + jnp.exp(2.j * k @ u) + jnp.exp(2.j * k @ (u - v)))
    nnn_2 = t * (1. + jnp.exp(2.j * k @ u) + jnp.exp(2.j * k @ v))
    nnn_3 = t * (1. + jnp.exp(2.j * k @ v) + jnp.exp(2.j * k @ (v - u)))
    nn_1 = self.t0 * jnp.exp(2.j * k @ u)
    nn_2 = self.t0 * jnp.exp(2.j * k @ v)
    # Define the coupling matrix.
    coupling_mat = jnp.zeros(
        (self.sublattice, self.sublattice),
        dtype=jnp.complex64
    )
    coupling_mat = coupling_mat.at[1, 0].set(1.)
    coupling_mat = coupling_mat.at[2, 0].set(1.)
    coupling_mat = coupling_mat.at[3, 0].set(nnn_1)
    coupling_mat = coupling_mat.at[7, 0].set(nn_1)
    coupling_mat = coupling_mat.at[3, 1].set(1.)
    coupling_mat = coupling_mat.at[4, 1].set(1.)
    coupling_mat = coupling_mat.at[5, 1].set(-1.)
    coupling_mat = coupling_mat.at[10, 1].set(-nn_2)
    coupling_mat = coupling_mat.at[11, 1].set(nn_2)
    coupling_mat = coupling_mat.at[3, 2].set(nn_1 * jnp.conjugate(nn_2))
    coupling_mat = coupling_mat.at[4, 2].set(-1.)
    coupling_mat = coupling_mat.at[6, 2].set(1.)
    coupling_mat = coupling_mat.at[9, 2].set(nn_1)
    coupling_mat = coupling_mat.at[11, 2].set(-nn_1)
    coupling_mat = coupling_mat.at[7, 3].set(1.)
    coupling_mat = coupling_mat.at[8, 4].set(1.)
    coupling_mat = coupling_mat.at[11, 4].set(nnn_2)
    coupling_mat = coupling_mat.at[7, 5].set(1.)
    coupling_mat = coupling_mat.at[8, 5].set(1.)
    coupling_mat = coupling_mat.at[10, 5].set(nnn_3)
    coupling_mat = coupling_mat.at[7, 6].set(-1. * nn_1 * jnp.conjugate(nn_2))
    coupling_mat = coupling_mat.at[8, 6].set(1.)
    coupling_mat = coupling_mat.at[9, 6].set(nnn_1)
    coupling_mat = coupling_mat.at[9, 7].set(-1.)
    coupling_mat = coupling_mat.at[10, 7].set(jnp.conjugate(nn_1) * nn_2)
    coupling_mat = coupling_mat.at[9, 8].set(1.)
    coupling_mat = coupling_mat.at[10, 8].set(1.)
    coupling_mat = coupling_mat.at[11, 8].set(1.)
    # Fill in complex conjugate. Minus sign for - J \sum_{<i, j>} H_ij.
    ham_mat =  - (coupling_mat + jnp.conjugate(jnp.transpose(coupling_mat)))
    # add onsite term.
    field_onsite = jnp.array(
      [
      mass_tri, mass_hex, mass_hex, mass_tri, mass_tri, mass_tri,
      mass_tri, mass_hex, mass_hex, mass_tri, mass_tri, mass_tri
      ]
    ) ** 2
    ham_mat += jnp.diag(field_onsite)
    return ham_mat
  
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


class IsingZeroFluxHamiltonian(ClassicalHamiltonian):
  """Vison mean-field hamiltonian with zero gauge field flux.

  This Hamiltonia is defined in the Ising basis for phi4 field theory.

  Note default coupling is ferromagnetic H = - J \sum_{<i, j>} H_ij."""

  def __init__(
      self,
      params: dict,
      brillouin_zone = None,
      sublattice: int = 3,
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
    # Yejin's convention
    # u = np.array([3. / 2., np.sqrt(3.) / 2.])
    # v = np.array([0., - np.sqrt(3)])    
    # Our convention      
    u = np.array([np.sqrt(3.), 0.])
    v = np.array([np.sqrt(3.) / 2., -3. / 2.])
    k = jnp.array([kx, ky])
    nnn_1 = t * jnp.exp(1.j * k @ (u - v))
    nn_1 = self.t0 * jnp.exp(1.j * k @ u)
    nn_2 = self.t0 * jnp.exp(1.j * k @ v)
    # Define the coupling matrix.
    coupling_mat = jnp.zeros(
        (self.sublattice, self.sublattice),
        dtype=jnp.complex64
    )
    coupling_mat = coupling_mat.at[0, 1].set(jnp.conjugate(nn_1))
    coupling_mat = coupling_mat.at[0, 2].set(3. * nnn_1)
    coupling_mat = coupling_mat.at[1, 2].set(jnp.conjugate(nn_2))
    # Fill in complex conjugate. Minus sign for - J \sum_{<i, j>} H_ij.
    ham_mat =  - (coupling_mat + jnp.conjugate(jnp.transpose(coupling_mat)))
    # add onsite term.
    field_onsite = jnp.array([mass_tri, mass_hex, mass_tri]) ** 2
    ham_mat += jnp.diag(field_onsite)
    return ham_mat
  
  def eta(self):
    """Definition of sublattice frustrated bond values."""
    eta_mat = np.zeros((self.sublattice, self.sublattice))
    eta_mat[0, 1] = 1.
    eta_mat[0, 2] = 1.
    eta_mat[1, 2] = 1.
    return eta_mat + np.transpose(eta_mat)  
