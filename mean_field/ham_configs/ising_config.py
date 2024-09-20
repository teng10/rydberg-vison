"""config file for data generation."""
import os

import numpy as np
from ml_collections import config_dict

home = os.path.expanduser('~')
DEFAULT_TASK_NAME = 'ising_pi_flux'


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
        m_tris=[2.5, 3., 3.5],
        m_hex_ratios=[0.5, 1., 2.]
    )),
    'sweep_large_triangular': list(ham_params_sweep_fn(
        ts=[0.1, 0.5],
        m_tris=[5., ],
        m_hex_ratios=[0.1, 0.5, 1.]
    )),
    'sweep_larger_omega': list(ham_params_sweep_fn(
        ts=[0.1, 0.5],
        m_tris=[2.5, 3., 3.5, 5.],
        m_hex_ratios=[0.25, 1.]
    )),
    'sweep_mass': list(ham_params_sweep_fn(
            ts=[0.5, ],
            m_tris=[2.5, 3., 3.5],
            m_hex_ratios=[1., 2.]
    )),
    'sweep_ferro': list(ham_params_sweep_fn(
            ts=[-0.5, ],
            m_tris=[2.5, 3., 3.5],
            m_hex_ratios=[0.5, 1., 2.]
    )),
    'sweep_af': list(ham_params_sweep_fn(
        ts=[0.5, ],
        m_tris=[2.5, 3., 3.5],
        m_hex_ratios=[0.5, 1., 2.]
    )),
    'sweep_af_ferro': list(ham_params_sweep_fn(
        ts=[0.5, -0.5],
        m_tris=[2.5, ],
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
  config.task.name = DEFAULT_TASK_NAME # 'ising_pi_flux' #'ising_zero_flux'
  config.task.bz_lattice_size = 50
  config.task.kwargs = {
      't': 0.5, 'm_tri': 25., 'm_hex_ratio': 2.5
  }
  # Structure factor computation.
  config.sf = config_dict.ConfigDict()
  config.sf.sf_type = 'dynamic_structure_factor' #'static_structure_factor'
  config.sf.omegas = np.linspace(.1, 7., 800)
  config.sf.q_path_name = 'ebz_gamma_m_k_gamma'# 'square_full' #
  config.sf.q_steps = 100
  config.sf.batch_size = 30
  # sweep parameters.
  config.sweep_name = 'sweep_af_ferro'  # Could change this in slurm script
  config.sweep_fn_registry = SWEEP_FN_REGISTRY
  # Save options.
  config.output = config_dict.ConfigDict()
  config.output.save_data = True
  config.output.data_dir = f'{home}/vison/Data/Tests/%CURRENT_DATE/'
  # by default we use date/job-id for file name.
  # need to keep this line for default value and jobs not on cluster.
  config.output.filename = '%JOB_ID'
  return config
