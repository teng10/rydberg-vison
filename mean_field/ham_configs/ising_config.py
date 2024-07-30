"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'vison_ising'


def sweep_param_fn(
    t: float,
    m_tri: float,
    m_hex_ratio: float,
) -> dict:
  """Helper function for constructing dmrg sweep parameters."""
  return {
      'task.kwargs.t': t,
      'task.kwargs.m_tri': m_tri,
      'task.kwargs.m_hex_ratio': m_hex_ratio,
      'output.filename':  '_'.join(['%JOB_ID', '%TASK_ID', DEFAULT_TASK_NAME,
          f'{t=}', f'{m_tri=}', f'{m_hex_ratio=}'])
  }


def ham_params_sweep_fn(
    ts: float,
    m_tris: float,
    m_hex_ratios: float,
):
  """Sweep over hamiltonian parameter configs."""
  for t in ts:
    for m_tri in m_tris:
      for m_hex_ratio in m_hex_ratios:
        yield sweep_param_fn(t=t, m_tri=m_tri, m_hex_ratio=m_hex_ratio)


SWEEP_FN_REGISTRY = {
    'sweep_t_m': list(ham_params_sweep_fn(
        ts=[0.1, 0.5],
        m_tris=[-5, -10, -15],
        m_hex_ratios=[0.5, 1., 2.]
    )),
}


def get_config():
  """config using surface code as an example."""
  config = config_dict.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.task_id = config_dict.placeholder(int)
  # Task configuration.
  config.task = config_dict.ConfigDict()
  config.task.name = DEFAULT_TASK_NAME #TODO(YT): add config option
  config.task.bz_lattice_size = 50
  config.task.kwargs = {
      't': 0.5, 'm_tri': -20, 'm_hex_ratio': -2.5
  }
  # Structure factor computation.
  config.sf = config_dict.ConfigDict()
  config.sf.omegas = np.linspace(0.1, 3., 3)
  config.sf.q_path_name = 'gamma_m_k_gamma'
  config.sf.q_steps = 2
  # sweep parameters.
  config.sweep_name = 'sweep_t_m'  # Could change this in slurm script
  config.sweep_fn_registry = SWEEP_FN_REGISTRY
  # Save options.
  config.output = config_dict.ConfigDict()
  config.output.save_data = True
  config.output.data_dir = f'{home}/vison/Data/Tests/%CURRENT_DATE/'
  # by default we use date/job-id for file name.
  # need to keep this line for default value and jobs not on cluster.
  config.output.filename = '%JOB_ID'
  return config
