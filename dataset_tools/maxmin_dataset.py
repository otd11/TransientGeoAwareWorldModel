

# Third party.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy import ndimage
import torch
import re
import h5py
from concurrent.futures import ProcessPoolExecutor
import os




# ==================================================================================== #
#                                      Parameters                                      #
# ==================================================================================== #

directory_path = "/net/milz/datasets/mathplus_physics/raw_data/100variations/hdf5/"


# dataset max and min


# values from the 1st 19 simulations
# [T_min, T_max]:  [299.994873046875, 313.7619323730469]
# [u_min, u_max]:  [-0.19720515608787537, 0.20272187888622284]
# [v_min, v_max]:  [-0.15918399393558502, 0.22270657122135162]
# [p_min, p_max]:  [-6.264317035675049, 6.542110443115234]




T_min = 299.994873046875
T_max = 313.7619323730469
u_min = -0.19720515608787537
u_max = 0.20272187888622284
v_min = -0.15918399393558502
v_max = 0.22270657122135162 
p_min = -6.264317035675049
p_max = 6.542110443115234


