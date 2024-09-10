"""Results or data utilities."""

import numpy as np
import xarray as xr


def split_complex_ds(ds: xr.Dataset) -> xr.Dataset:
  """Split complex dataset variables into real and imaginary parts."""
  for var in ds.data_vars:
    if np.iscomplexobj(ds[var].values):
      ds[var + '_real'] = ds[var].real
      ds[var + '_imag'] = ds[var].imag
      ds = ds.drop_vars(var)
  return ds


def combine_complex_ds(ds: xr.Dataset) -> xr.Dataset:
  """Combine real and imaginary parts of complex dataset variables."""
  for var in ds.data_vars:
    if var.endswith('real'):
      var_real = var
      var_imag = var[:-len('_real')] + '_imag'
      ds[var[:-5]] = ds[var_real] + 1.j * ds[var_imag]
      ds = ds.drop_vars([var_real, var_imag])
  return ds

def convert_to_real_ds(ds: xr.Dataset, tol: float=1e-4) -> xr.Dataset:
  """Check and converts complex dataset to real dataset."""
  for var_name in ds.data_vars:
    if np.all(np.abs(ds[var_name].imag) < tol):
      ds[var_name] = ds[var_name].real
    else:
      raise ValueError(f'Imaginary part of {var_name} has a max component \
          {np.max(np.abs(ds[var_name].imag))} great than {tol=}.')
  return ds
