"""Computations of the dynamical structure factor."""
from __future__ import annotations

import abc
import numpy as np
import jax.numpy as jnp


class DynamicalStructureFactor(abc.ABC):
  """Class for computing the dynamical structure factor."""

  def __init__(
      self,
      lattice: Lattice,
      bz_lattice: ReciprocalDiceLattice,
      hamiltonian: Hamiltonian,
  ):
    self.delta = lattice.delta()
    self.eta = lattice.eta()
    self.hamiltonian = hamiltonian
    self.sublattice = hamiltonian.sublattice
    self.kpoints = bz_lattice.kpoints
    self.decimal_precision = 5

    self.spectrum_k = self.hamiltonian.get_eignenvalue_spectra(self.kpoints)
    self.spectrum_mk = self.hamiltonian.get_eignenvalue_spectra( -self.kpoints)
    self.qmk_spectra = {}
    self.kmq_spectra = {}


  def _compute_eigenvector_matrix_elements(self, eigvecs_k1, eigvecs_k2):
    """Compute the eigenvector matrix elements.

    We want to multiply the components of the same k indices and same l indices,
    e.g. V_{k1}^{\alpha l}V_{k2}^{\beta l}
    arriving at a tensor of shape (k, \alpha, \beta, l) without summing over l.
    """
    return np.expand_dims(eigvecs_k1, axis=2) * np.expand_dims(eigvecs_k1, axis=1)

  def _lorenzian(self, omega: float, omegap: float, gamma:float=0.1) -> float:
    """Lorenzian function to approximate delta function over frequencies.

    The width of the lorenzian is given by gamma.

    Args:
      omega: The frequency.
      omegap: The peak frequency.
      gamma: The width of the lorenzian.

    Returns:
      The lorenzian function.
    """
    return gamma / (2. * jnp.pi * ((omega - omegap)**2 + (gamma / 2.)**2))

  def _frequency_factor(self, omega, eigvals_k, eigvals_qmk):
    """Computes the additional frequency dependent factor.

    (1/\sqrt{\epsilon_l} + 1/\sqrt{\epsilon_l'}) \n
    \times (\omega / (\omega^2 + (\sqrt{\epsilon_l} + \sqrt{\epsilon_l'})^2))

    Args:
      omega: The frequency.
      q_vector: The momentum transfer.

    Returns:
      Frequency dependent tensor of shape (k, l, l').
    """
    eigvals_k = np.expand_dims(eigvals_k, axis=2)
    eigvals_qmk = np.expand_dims(eigvals_qmk, axis=1)
    return jnp.pi**2 * (
      1. / jnp.sqrt(eigvals_k) + 1. / jnp.sqrt(eigvals_qmk)
    ) * (
        omega / (omega**2 + (jnp.sqrt(eigvals_k) + jnp.sqrt(eigvals_qmk))**2)
    ) * self._lorenzian(omega, jnp.sqrt(eigvals_k) + jnp.sqrt(eigvals_qmk))


  def momentum_factor_matrix(
      self,
      q_vector: np.ndarray,
      sign: int=1,
  ) -> np.ndarray:
    """Computes the momentum exponential factor matrix (momentum, alpha, beta).

    e^{i * sign (q / 2 - k) \cdot \delta_{\alpha \beta}}
    
    Args:
      q_vector: The momentum transfer.
      sign: The sign of the momentum transfer.
    
    Returns:
      The momentum factor matrix.
    """
    return np.exp(1j * sign * (
        np.einsum('ij, klj -> ikl', q_vector / 2. - self.kpoints, self.delta)
    ))

  def _vector_hashkey(self, vector: np.ndarray):
    return str(tuple(np.round(vector, self.decimal_precision)))


  def compute_structure_factor(
      self,
      omega: float,
      q_vectors: np.ndarray,
  ) -> jnp.ndarray:
    """Computes the dynamical structure factor.

    Args:
      omega: The frequency.
      q_vectors: The momentum transfers.

    Returns:
      The dynamical structure factor.
    """
    # Compute all the spectra for the different momentum transfers q
    # TODO(YT): consider using jax for this computation?
    structure_factor_qs = []
    for q_vector in q_vectors:
      if (self._vector_hashkey(q_vector) in self.qmk_spectra
          and self._vector_hashkey(q_vector) in self.kmq_spectra
      ):
        spectrum_qmk = self.qmk_spectra[self._vector_hashkey(q_vector)]
        spectrum_kmq = self.kmq_spectra[self._vector_hashkey(q_vector)]
      else:
        spectrum_qmk = self.hamiltonian.get_eignenvalue_spectra(
            -(self.kpoints - q_vector)
        )
        spectrum_kmq = self.hamiltonian.get_eignenvalue_spectra(
            (self.kpoints - q_vector)
        )
        self.qmk_spectra[self._vector_hashkey(q_vector)] = spectrum_qmk
        self.kmq_spectra[self._vector_hashkey(q_vector)] = spectrum_kmq

      eigvals_k = self.spectrum_k.evals
      eigvals_qmk = spectrum_qmk.evals

      eigvecs_k = self.spectrum_k.evecs
      eigvecs_qmk = spectrum_qmk.evecs
      eigvecs_kmq = spectrum_kmq.evecs
      eigvecs_mk = self.spectrum_mk.evecs

      momentum_q = self.momentum_factor_matrix(q_vector, 1)
      momentum_mq = self.momentum_factor_matrix(q_vector, -1)

      eigvec_k = self._compute_eigenvector_matrix_elements(
          eigvecs_k, eigvecs_mk
      )
      eigvec_qk = self._compute_eigenvector_matrix_elements(
          eigvecs_qmk, eigvecs_kmq
      )

      freq_mat = self._frequency_factor(omega, eigvals_k, eigvals_qmk)

      first_term = 1. / len(self.kpoints) * jnp.einsum(
          'ab, gd, pab, pgd, pagm, pbdn, pmn -> ',
          self.eta, self.eta,
          momentum_q, momentum_mq, eigvec_k, eigvec_qk, freq_mat,
          optimize=True,
      )
      second_term = 1. / len(self.kpoints) * jnp.einsum(
          'ab, gd, pab, pgd, padm, pbgn, pmn -> ',
          self.eta, self.eta,
          momentum_q, momentum_mq, eigvec_k, eigvec_qk, freq_mat,
          optimize=True,
      )
      structure_factor_qs.append(first_term + second_term)
    return jnp.array(structure_factor_qs)
