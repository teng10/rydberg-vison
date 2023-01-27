import numpy as np
def _plot_v_bonds(ax, v_pairs, c='b'):
  # helper function for plotting pairs of bonds
  for i, v_pair in enumerate(v_pairs):
    if isinstance(c, np.ndarray):
      ax.plot(v_pair[:, 0], v_pair[:, 1], color=c[i])
    else:
      ax.plot(v_pair[:, 0], v_pair[:, 1], color=c)

def _plot_visons(ax, v_pairs, Q_gs, Q_vison, c='b', p=1.):
  # helper function for plotting pairs of bonds
  shift_x = 0.05
  for i, v_pair in enumerate(v_pairs):
    Q0 = Q_gs[i]
    Qv = Q_vison[i]
    diff = np.around(Qv - Q0, 2)
    thickness = np.sqrt(np.abs(Qv-Q0))
    v_mid = np.around(np.mean(v_pair, axis=0), 3)
    if isinstance(c, np.ndarray):
      ax.plot(v_pair[:, 0], v_pair[:, 1], color=c[i], linewidth=p * thickness)
    else:
      ax.plot(v_pair[:, 0], v_pair[:, 1], color=c, linewidth=p * thickness)
      ax.annotate(diff, (v_mid[0]+ shift_x, v_mid[1]), color='r')
  