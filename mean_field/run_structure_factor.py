"""Main file for running computations of structure factor."""
  # run this script with the following command in the terminal:
# python -m mean_field.run_structure_factor \
# --ham_config=mean_field/ham_configs/ising_config_test.py \
# --ham_config.job_id=1 \
# --ham_config.task_id=0
from absl import app
from absl import flags
from datetime import datetime
import os

from ml_collections import config_flags
import numpy as np
import jax.numpy as jnp
import xarray as xr

from mean_field import data_utils
from mean_field import hamiltonians
from mean_field import lattices
from mean_field import reciprocal_lattices
from mean_field import structure_factor

config_flags.DEFINE_config_file('ham_config')
FLAGS = flags.FLAGS

import jax
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


PHYSICAL_PROPERTIES_REGISTRY = structure_factor.PHYSICAL_PROPERTIES_REGISTRY

def _compute_structure_factor(
    ham_params: dict,
    bz_lattice_size: int,
    omegas: np.ndarray,
    q_path_name: str,
    q_steps: int,
    sf_type: str,
):
  bz_dice = reciprocal_lattices.ReciprocalDiceLattice(
      bz_lattice_size, bz_lattice_size
  ) # Brillouin zone
  visonham = hamiltonians.IsingHuhHamiltonian(ham_params) # Hamiltonian
  DiceLattice = lattices.EnlargedDiceLattice() # Create Dice lattice
  # Define q points on the full Brillouin zone
  points_q = reciprocal_lattices.BZ_PATH_REGISTRY[q_path_name](bz_dice, q_steps)
  sf_cls = structure_factor.DynamicalStructureFactor(
      DiceLattice, bz_dice, visonham
  )
  # results are (q_index, omega) array
  # time code
  start_time = datetime.now()
  if sf_type == 'static_structure_factor':
    sf_fn = sf_cls.compute_static_structure_factor
    sf_fn_jit = jax.jit(sf_fn)
    # sf_fn_vmap = jax.vmap(sf_fn, in_axes=(0,))
    # sf_fn_jit = jax.jit(sf_fn_vmap)
    # Batch computations to avoid memory issues.
    sf_results = jax.lax.map(sf_fn_jit, points_q, batch_size=500)[..., jnp.newaxis]
    omegas = np.array([0.])

    # sf_results = sf_cls.static_structure_factor_partially_contracted(points_q)
  elif sf_type == 'dynamic_structure_factor':
    sf_results = sf_cls.structure_factor_partially_contracted(points_q, omegas)
  else:
    raise ValueError(f'{sf_type} not recgonized.')
  end_time = datetime.now()
  print(f"Time taken: {end_time - start_time}")
  # sf_fn = PHYSICAL_PROPERTIES_REGISTRY['dynamic_structure_factor']
  # sf_results = sf_fn(points_q, omegas)
  sf_coords = {
      'q_index': np.arange(len(points_q)),
      'omega': omegas, 
      'qxy': np.arange(2)
  }
  sf_results_expanded = sf_results
  for _ in enumerate(ham_params):
    sf_results_expanded = sf_results_expanded[..., np.newaxis]
  return xr.Dataset(
      {
          'structure_factor': (
              ['q_index', 'omega'] + list(ham_params.keys()),
              sf_results_expanded
          ), 
          'q_points': (['q_index', 'qxy'], points_q)
      },
      coords={
          **sf_coords, 
          **{key: np.array([val]) for key, val in ham_params.items()}
      }
  )

def run_computation(config):
  if config.sweep_name in config.sweep_fn_registry:
    sweep_param = config.sweep_fn_registry[config.sweep_name]
    config.update_from_flattened_dict(sweep_param[config.task_id])
  elif config.sweep_name is None:
    pass
  else:
    raise ValueError(f'{config.sweep_name} not in sweep_fn_registry.')  
  ham_params = config.task.kwargs
  print(f"ham_params: {ham_params}")
  bz_lattice_size = config.task.bz_lattice_size
  omegas = config.sf.omegas
  q_path_name = config.sf.q_path_name
  q_steps = config.sf.q_steps
  CURRENT_DATE = datetime.now().strftime('%m%d')
  ds = _compute_structure_factor(
      ham_params, bz_lattice_size, omegas, q_path_name, q_steps, 
      sf_type=config.sf.sf_type
  )
  print(ds.structure_factor)
  ds = data_utils.convert_to_real_ds(ds)
  if config.output.save_data:
    data_dir = config.output.data_dir.replace('%CURRENT_DATE', CURRENT_DATE)
    filename = config.output.filename.replace('%JOB_ID', str(config.job_id))
    filename = filename.replace('%TASK_ID', str(config.task_id))
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    filepath = os.path.join(data_dir, filename)
    ds.to_netcdf(filepath + '.nc')
  return ds


def main(argv):
  config = FLAGS.ham_config
  return run_computation(config)


if __name__ == '__main__':
  app.run(main)