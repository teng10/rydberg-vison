"""Computations of the dynamical structure factor."""
from __future__ import annotations
import functools
import tqdm
from typing import Tuple

import abc
import numpy as np
import jax
import jax.numpy as jnp

from mean_field import hamiltonians
from mean_field.lattices import Lattice
from mean_field.reciprocal_lattices import ReciprocalDiceLattice
from mean_field.hamiltonians import Hamiltonian

  
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
    self.decimal_precision = 8

    self.spectrum_k = self.hamiltonian.get_eignenvalue_spectra(self.kpoints)
    #TODO(YT): verify the symmetry of the Hamiltonian
    self.spectrum_mk = hamiltonians._get_mk_spectra_symmetry(self.spectrum_k)
    self.qmk_spectra = {}
    self.kmq_spectra = {}


  def _compute_eigenvector_matrix_elements(self, eigvecs_k1, eigvecs_k2):
    """Compute the eigenvector matrix elements.

    We want to multiply the components of the same k indices and same l indices,
    e.g. V_{k1}^{\alpha l}V_{k2}^{\beta l}
    arriving at a tensor of shape (k, \alpha, \beta, l) without summing over l.
    """
    return np.expand_dims(eigvecs_k1, axis=2) * np.expand_dims(eigvecs_k2, axis=1)

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
    eigvals_k = jnp.expand_dims(eigvals_k, axis=2)
    eigvals_qmk = jnp.expand_dims(eigvals_qmk, axis=1)
    return jnp.pi**2 * (
      1. / jnp.sqrt(eigvals_k) + 1. / jnp.sqrt(eigvals_qmk)
    ) * (
        1 / (2 * (jnp.sqrt(eigvals_k) + jnp.sqrt(eigvals_qmk)))
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
          momentum_q, momentum_q, eigvec_k, eigvec_qk, freq_mat,
          optimize=True,
      )
      structure_factor_qs.append(first_term + second_term)
    return jnp.array(structure_factor_qs)


  def get_ingredients(self, q_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the ingredients for the structure factor.
    
    Ingredients are the partial contractions of the structure factor,
    and the eigenvalues of the Hamiltonian for the k and q - k points.

    Args:
      q_vector: The momentum transfer.
    
    Returns:
      The ingredients for the structure factor.
    """
    spectrum_qmk = self.hamiltonian.get_eignenvalue_spectra(
        -(self.kpoints - q_vector)
    )
    spectrum_kmq = hamiltonians._get_mk_spectra_symmetry(spectrum_qmk)
    eigvals_k = self.spectrum_k.evals
    eigvals_qmk = spectrum_qmk.evals

    eigvecs_k = self.spectrum_k.evecs
    eigvecs_qmk = spectrum_qmk.evecs
    eigvecs_kmq = spectrum_kmq.evecs
    eigvecs_mk = self.spectrum_mk.evecs

    momentum_q = self.momentum_factor_matrix(q_vector, 1)
    momentum_mq = self.momentum_factor_matrix(q_vector, -1)

    eigvec_k = self._compute_eigenvector_matrix_elements(eigvecs_k, eigvecs_mk)
    eigvec_qk = self._compute_eigenvector_matrix_elements(eigvecs_qmk, eigvecs_kmq)

    first_partial_contraction = np.einsum(
        'ab, gd, pab, pgd, pagm, pbdn -> pmn',
        self.eta, self.eta,
        momentum_q, momentum_mq, eigvec_k, eigvec_qk, 
        optimize=True,
    )
    second_partial_contraction = np.einsum(
        'ab, gd, pab, pgd, padm, pbgn -> pmn',
        self.eta, self.eta,
        momentum_q, momentum_q, eigvec_k, eigvec_qk, 
        optimize=True,
    )
    # print(f"the zero eigenvalues for k are {np.where(eigvals_k == 0)}")
    # print(f"the zero eigenvalues for qmk are {np.where(eigvals_qmk == 0)}")
    return (
        first_partial_contraction, second_partial_contraction,
        eigvals_k, eigvals_qmk
    )
    
  def calculate_from_ingredients(
      self,
      omega: float,
      ingredients: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
  )-> float:
    """Computes the structure factor from precomputed partial contractions.
    
    Args:
      omega: The frequency.
      ingredients: The ingredients for the structure factor.

    Returns:
      The structure factor.  
    """
    first_ingredient, second_ingredient, eigvals_k, eigvals_qmk = ingredients
    freq_mat = self._frequency_factor(omega, eigvals_k, eigvals_qmk)
    return 1. / len(first_ingredient) * jnp.einsum(
        'pmn, pmn -> ',
        first_ingredient + second_ingredient, freq_mat
    )
      
  def structure_factor_partially_contracted(
        self, 
        q_vectors: np.ndarray, 
        omegas: np.ndarray
    )-> jnp.ndarray:
    """Compute structure factor for a single q and all omegas."""

    sf_all_qs = []
    for q_vector in tqdm.tqdm(q_vectors):
      sf_q_partial_contraction_fn = functools.partial(
          self.calculate_from_ingredients,
          ingredients=self.get_ingredients(q_vector)
      )
      sf_q_vmap = jax.vmap(sf_q_partial_contraction_fn)
      sf_q_vmap_jitted = jax.jit(sf_q_vmap)
      sf_all_qs.append(sf_q_vmap_jitted(omegas))
    return jnp.stack(sf_all_qs) 
