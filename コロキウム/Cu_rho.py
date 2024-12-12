# coding: utf-8
#
# Source file  : <Will be updated during commit>
# Last modified: <Will be updated during commit>
# Committed by : <Will be updated during commit>

#coding: utf-8

studies_path = "../"

import sys
sys.path.append(studies_path)

#import argparse
import logging
import numpy as np
import os

from ase  import Atom, Atoms
from msspec.utils import *
from msspec.misc import set_log_level
from ase.build import bulk, fcc111
#from ase.lattice.tetragonal import SimpleTetragonalFactory
from ase.visualize import view
from ase.io import write
from ase.io import read
from matplotlib import pyplot as plt

from msspec.calculator import MSSPEC
from msspec.iodata import Data
#from msspec.looper import Sweep, Looper
from msspec.utils import hemispherical_cluster, get_atom_index, cut_cylinder

## functions for renormalization studies
from studies.load_from_file import load_from_stream
from studies.MsMatrix import MsMatrix
from studies.find_optimal_spectral_radius import find_optimal_spectral_radius

#logging.basicConfig(level=logging.INFO)
set_log_level("debug")

do_Cu = True
##########

atom = 'Cu'
level = '2p3/2'

guess = [0.8, 0.1]
all_ke = [95,95,1]
#absorber = 0
#off_cone_events = 0

##################### 100 direction ##########################
#Create the Ni atomic chain
if do_Cu:
   savefile = 'spectral_radius.hdf5'
#   savefile = 'spectral_radius_{}.hdf5'.format('Cu')

   a0 = 3.6 # The lattice parameter in angstroms

# Create the copper cubic cell
#   copper = bulk('Cu', a=a0, cubic=True)
#   cluster = hemispherical_cluster(copper, planes=4, emitter_plane=3)   
#   cluster = cut_cylinder(cluster, radius=1.75*nn)

# Set the absorber (the deepest atom centered in the xy-plane)
#   cluster.absorber = get_atom_index(cluster, 0, 0, 0)
   a0  = 3.61                             # the lattice parameter of the cubic unit cell
   d   = a0/np.sqrt(3)                    # interplanar distance in (111) direction
   nn  = a0/np.sqrt(2)                    # nearest neighbors distance in [001] plane
   Cu=fcc111('Cu', a=a0, size=(2,2,3))    # Copper unit cell grown along (111) axis
   Cu.cell[-1,-1] = 3*d                   # update cell to the size of unit cell

# create a huge hemispherical cluster based on this unit cell
   cluster = hemispherical_cluster(Cu, emitter_tag=1, diameter=25, emitter_plane=3, planes=4)

# cut a cylinder inside this clsuter
   cluster = cut_cylinder(cluster, radius=1.75*nn)

# Set the absorber (the deepest atom centered in the xy-plane)
   cluster.absorber = get_atom_index(cluster, 0, 0, 0)
#   view(cluster)
#   exit(0)

# Create a calculator for the spectral radius
   calc = MSSPEC(spectroscopy='EIG', algorithm='inversion')

# Set the cluster to use for the calculation
   calc.set_atoms(cluster)

   calc.calculation_parameters.RA_cutoff = 2

   calc.phagen_parameters.potype = 'hedin'
#   calc.phagen_parameters.potype = 'xalph'

   data = calc.get_eigen_values(kinetic_energy=all_ke)

   # Save results to hdf5 file
   data.save(savefile)

### Find optimal renormalization parameter for this chain and kinetic energy ####

   efile = 'calc/output/eigenvalues.dat'
   data = load_from_stream(efile)
   file_path = "/home/MsSpec/Thory_Group/Colloquium/Cu_eigenvalues/calc/output/eigenvalues.dat"

   # 確認したいファイルのパス /home/shyasuda/My_code_randomChains/MyTest_spectral/STO3_chain_6_7
#   file_path = "/home/shyasuda/Didier_15_05_2024/calc/output/eigenvalues.dat"
#   print(data)
#   print(type(data))
#   exit(0)
# ファイルの存在を確認
#   if os.path.exists(file_path):
#          print("The file indicated exists")
#          old_directory_name = "calc"
#          new_directory_name = "calc_{}".format('Cu')
#          old_directory_path = "/home/shyasuda/Didier_15_05_2024/calc"
#          new_directory_path = "/home/shyasuda/Didier_15_05_2024/calc_{}".format('Cu')
#          os.rename(old_directory_path, new_directory_path)
#   else:
#          print("The file indicated doesn't exist")


   with open('optimal_values.dat', 'w') as f:

       f.write("#                         G_n                                Sigma_n                              Pi_1                  \n")
       f.write("#   ke       rho(k)       wx         wy         rho(k)       wx         wy         rho         wx         wy         rho\n")

       for d in data:

           ke = d.kinetic_energy
           rho = d.spectral_radius
           f.write("  {:5.1f}  {:10.5f}".format(ke, rho))

           opt_G_n = find_optimal_spectral_radius(d, renormalization_mode='G_n', initial_guess=guess)
           omega = opt_G_n[-1]['omega']
           rho = opt_G_n[-1]['rho']
           f.write("  {:10.5f} {:10.5f} {:10.5f}".format(omega.real, omega.imag, rho))

           opt_Sigma_n = find_optimal_spectral_radius(d, renormalization_mode='Sigma_n', initial_guess=guess)
           omega = opt_Sigma_n[-1]['omega']
           rho = opt_Sigma_n[-1]['rho']
           f.write("  {:10.5f} {:10.5f} {:10.5f}".format(omega.real, omega.imag, rho))

           opt_Pi_1 = find_optimal_spectral_radius(d, renormalization_mode='Pi_1', initial_guess=guess)
           omega = opt_Pi_1[-1]['omega']
           rho = opt_Pi_1[-1]['rho']
           f.write("  {:10.5f} {:10.5f} {:10.5f}".format(omega.real, omega.imag, rho))

           f.write("\n")

   print("rho(K) = {}".format(data[-1].spectral_radius))
   print(opt_G_n)
   print(opt_Pi_1)
   print(opt_Sigma_n)
