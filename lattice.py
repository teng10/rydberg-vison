#Create a Lattice class?

def create_grid_mesh(params):
  kx1 = params['x1']
  kx2 = params['x2']
  ky1 = params['y1']
  ky2 = params['y2']
  Nx = params['Nx']
  Ny = params['Ny']

  spacing_x = (kx2 - kx1) / (Nx)
  spacing_y = (ky2 - ky1) / (Ny)
  kx_grid = np.linspace(kx1, kx2 - spacing_x, Nx)
  ky_grid = np.linspace(ky1, ky2 - spacing_y, Ny)

  #create a mesh
  X, Y = np.meshgrid(kx_grid, ky_grid)

  return X, Y

def create_grid(params):

  #create a mesh
  my_x, my_y = create_grid_mesh(params)
  a1 = params['a1']
  a2 = params['a2']
  #stack the mesh to (Nx, Ny, d=2) dim
  # print(my_x.shape)
  # my_grid = np.stack([my_x * a1 + my_y * a2], axis=-1)
  my_grid = np.stack([my_x, my_y], axis=-1)
  # print(my_grid.shape)

  my_grid = einops.rearrange(my_grid, 'x y d -> (x y) d') #integer grid

  return np.tensordot(my_grid[:, 0], a1, 0) + np.tensordot(my_grid[:, 1], a2, 0)