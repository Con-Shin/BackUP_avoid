#!/usr/bin/env python
# coding: utf-8
#
# Source file  : <Will be updated during commit>
# Last modified: <Will be updated during commit>
# Committed by : <Will be updated during commit>
#calc_path = "../"
#import sys
#sys.path.append(calc_path)
#from Pt_Atomchains import cluster_Pt, Cone_SrSr, Cone_SrO, Cone_SrTi    
#from Cone_making import cut_cone
## functions for renormalization studies
#from studies.load_from_file import load_from_stream
#from studies.MsMatrix import MsMatrix
#from studies.find_optimal_spectral_radius import find_optimal_spectral_radius
#calc_path = "../../"
#import sys
#sys.path.append(calc_path)
#from Pt_Atomchains import cluster_Pt, Cone_SrSr, Cone_SrO, Cone_SrTi    
#from Cone_making import Cone_making

import json
import argparse
import logging
import numpy as np
import shutil
import os

from ase  import Atom, Atoms
from ase.build import bulk
from ase.lattice.tetragonal import SimpleTetragonalFactory

from matplotlib import pyplot as plt

from msspec.calculator import MSSPEC,XRaySource
from msspec.iodata import Data
from msspec.data import electron_be
from msspec.misc import set_log_level,get_level_from_electron_configuration
from msspec.looper import Sweep, Looper
from msspec.utils import hemispherical_cluster, get_atom_index, cut_cylinder, cut_sphere, cut_plane
from ase.visualize import view
logging.basicConfig(level=logging.INFO)

do_mi  = True
do_se = True
do_gn  = True

a0 = 3.52
ke = 379.1
symbols = ("Ni",)
chain_range = (3,4)
absorber = 0
level = '2p'

for chain_length in range(*chain_range):
    ## create a Ni chain of atoms along z-axis
    chain = Atoms()
    for i in range(chain_length):
        chain += Atom(symbols[i % len(symbols)], position=(0, 0, i*a0))
    chain.absorber = absorber
    #chain.edit()
    #exit()
    # photon energy
    Z = chain[chain.absorber].number
    WF = 4.5
    Eph = ke + electron_be[Z][get_level_from_electron_configuration(level)] + WF

for i in range(len(chain)):
    a = chain[i]
    [a.set('mt_radius',2.25)]

suffix = 'MISE'
all_data = Data(suffix)

dset_ped = all_data.add_dset(f"PED {2} atoms")
view_ped = dset_ped.add_view(dset_ped.title,
                              title=(f"PED polar scan for "
                                     f"{'2 Ni atoms'}({level}) "
                                     f"@ {ke:.2f} eV"),
                              xlabel=r"$\theta$ ($\deg$)",

                              ylabel="Signal (a.u.)")
if do_mi:

    calc = MSSPEC(spectroscopy='PED',
                folder=(f"./{''.join(symbols)}{level}_{suffix}/"
                            f"PED_MI_{2:0d}"),algorithm='inversion')
#    exit(0)
    calc.source_parameters.theta  = 55.15
    calc.source_parameters.phi    = -44.7
    calc.source_parameters.energy  = Eph
#    calc.source_parameters.energy  = XRaySource.AL_KALPHA
    calc.detector_parameters.angular_acceptance = 1
    calc.detector_parameters.average_sampling   = 'low' #'high'

#    emitter_plane = cluster.info['emitter_plane']
#    calc.calculation_parameters.scattering_order = NDIFF
    calc.tmatrix_parameters.tl_threshold = 1e-4

#    26/06/2024 If it doesn't work, comment out it below.
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29, 'O': 29}
#    calc.tmatrix_parameters.lmax_mode = 'imposed'
#    calc.tmatrix_parameters.lmaxt = 3

    calc.muffintin_parameters.interstitial_potential = 15.1
    # It could be better to add some vibrational damping
#    for atom in cone_111:
#        atom.set('mean_square_vibration',0.022)
    calc.calculation_parameters.vibrational_damping = 'averaged_tl'

    # The amount of sphericity
#    calc.calculation_parameters.RA_cutoff = 3
#    calc.calculation_parameters.RA_cutoff = 2
    calc.phagen_parameters.potype = 'hedin'
    # Filter out paths that are outside of a given scattering cone angle (of 30° here).
    # This saves a lot of useless calculatuions
#    calc.calculation_parameters.RA_cutoff_damping = 1

#    cluster.absorber = get_atom_index(cone_111, 0, 0, 0)
    # Attach the calculator to the cluster...
    calc.set_atoms(chain)
#    cluster.absorber = get_atom_index(cone_111, 0, 0, 0)
#    exit(0)
    # ... And beam me up Scotty
#    data = calc.get_theta_scan(level=level,
#                               theta=theta,
#                               phi=[0, 45],
#                               kinetic_energy=kinetic_energy)
#    exit(0)
    data = calc.get_theta_scan(level=level,
                               theta=np.arange(-80.,80,0.5),
                               phi=[0],
                               kinetic_energy=ke)
#    exit(0)
    if "theta" not in dset_ped.columns():
        dset_ped.add_columns(theta=calc.iodata[-1].theta)
    dset_ped.add_columns(cs_mi=data[-1].cross_section)
    view_ped.select("theta", "cs_mi", legend="Matrix inversion")
    calc.add_cluster_to_dset(dset_ped)

if do_se:

    calc = MSSPEC(spectroscopy='PED',
                folder=(f"./{''.join(symbols)}{level}_{suffix}/"
                        f"PED_SE_{2:0d}"),algorithm='expansion')

    calc.source_parameters.theta  = 55.15
    calc.source_parameters.phi    = -44.7
#    calc.source_parameters.energy  = XRaySource.AL_KALPHA
    calc.source_parameters.energy  = Eph
    calc.detector_parameters.angular_acceptance = 1
    calc.detector_parameters.average_sampling   = 'low' #'high'

#    emitter_plane = cluster.info['emitter_plane']
    calc.calculation_parameters.scattering_order = 5
    calc.tmatrix_parameters.tl_threshold = 1e-4

#    26/06/2024 If it doesn't work, comment out it below.
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29, 'O': 29}
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29}
#    calc.tmatrix_parameters.lmax_mode = 'imposed'
#    calc.tmatrix_parameters.lmaxt = 15

    calc.muffintin_parameters.interstitial_potential = 15.1
    # It could be better to add some vibrational damping
#    for atom in chain:
#        atom.set('mean_square_vibration',0.022)
    calc.calculation_parameters.vibrational_damping = 'averaged_tl'

    # The amount of sphericity
#    calc.calculation_parameters.RA_cutoff = 1
    calc.calculation_parameters.RA_cutoff = 2
    calc.phagen_parameters.potype = 'hedin'
    # Filter out paths that are outside of a given scattering cone angle (of 30° here).
    # This saves a lot of useless calculatuions
    calc.calculation_parameters.RA_cutoff_damping = 0

#    filters = []
#    if FORWARD_ANGLE != -1:
#    filters.append('forward_scattering')
#    [a.set('forward_angle', 1) for a in chain]
#    if BACKWARD_ANGLE != -1:
#    filters.append('backward_scattering')
#    [a.set('backward_angle', 1) for a in chain]
#    if DISTANCE_CUTOFF != -1:
#    filters.append('distance_cutoff')
#    calc.calculation_parameters.distance = 8.0

#    calc.calculation_parameters.path_filtering   = filters
#    calc.calculation_parameters.off_cone_events   = 0

    # Renormalization
#    calc.calculation_parameters.renormalization_mode = RENORMALIZATION_MODE
#    calc.calculation_parameters.renormalization_omega = OMEGA


    # Attach the calculator to the cluster...
    calc.set_atoms(chain)
#    calc.set_chains([chain])
    # ... And beam me up Scotty
#    data = calc.get_theta_scan(level=level,
#                               theta=theta,
#                               phi=[0, 45],
#                               kinetic_energy=kinetic_energy)
#    exit(0)
    data = calc.get_theta_scan(level=level,
                               theta=np.arange(-80,80,0.5),
                               phi=[0],
                               kinetic_energy=ke)

    if "theta" not in dset_ped.columns():
      dset_ped.add_columns(theta=calc.iodata[-1].theta)
    dset_ped.add_columns(cs_se=calc.iodata[-1].cross_section)
    dset_ped.add_columns(cs_sed=calc.iodata[-1].direct_signal)
    view_ped.select("theta", "cs_se", legend="Series expansion")
#    view_ped.select("theta", "cs_sed", legend="Series expansion direct")

if do_gn:

    calc = MSSPEC(spectroscopy='PED',
                folder=(f"./{''.join(symbols)}{level}_{suffix}/"
                        f"PED_SE_{2:0d}"),algorithm='expansion')

    calc.source_parameters.theta  = 55.15
    calc.source_parameters.phi    = -44.7
#    calc.source_parameters.energy  = XRaySource.AL_KALPHA
    calc.source_parameters.energy  = Eph
    calc.detector_parameters.angular_acceptance = 1
    calc.detector_parameters.average_sampling   = 'low' #'high'

#    emitter_plane = cluster.info['emitter_plane']
    calc.calculation_parameters.scattering_order = 5
    calc.tmatrix_parameters.tl_threshold = 1e-4

#    26/06/2024 If it doesn't work, comment out it below.
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29, 'O': 29}
#    calc.tmatrix_parameters.lmax_mode = 'imposed'
#    calc.tmatrix_parameters.lmaxt = 5

    calc.muffintin_parameters.interstitial_potential = 15.1
    # It could be better to add some vibrational damping
#    for atom in cone_111:
#        atom.set('mean_square_vibration',0.022)
    calc.calculation_parameters.vibrational_damping = 'averaged_tl'

    # The amount of sphericity
    calc.calculation_parameters.RA_cutoff = 2
    calc.phagen_parameters.potype = 'hedin'
    # Filter out paths that are outside of a given scattering cone angle (of 30° here).
    # This saves a lot of useless calculatuions
#    calc.calculation_parameters.RA_cutoff_damping = 1

#    filters = []
#    if FORWARD_ANGLE != -1:
#        filters.append('forward_scattering')
#        [a.set('forward_angle', FORWARD_ANGLE) for a in cone_111]
#    if BACKWARD_ANGLE != -1:
#    filters.append('backward_scattering')
#    [a.set('backward_angle', 1) for a in chain]
#    if DISTANCE_CUTOFF != -1:
#    filters.append('distance_cutoff')
#    calc.calculation_parameters.distance = 8

#    calc.calculation_parameters.path_filtering   = filters
#    calc.calculation_parameters.off_cone_events   = 0

    # Renormalization
#    calc.calculation_parameters.renormalization_mode = 'Sigma_n'
    calc.calculation_parameters.renormalization_mode = 'G_n'

    calc.calculation_parameters.renormalization_omega = 0.82564+0.14028j


    # Attach the calculator to the cluster...
    calc.set_atoms(chain)

    # ... And beam me up Scotty
#    data = calc.get_theta_scan(level=level,
#                               theta=theta,
#                               phi=[0, 45],
#                               kinetic_energy=kinetic_energy)
    data = calc.get_theta_scan(level=level,
                               theta=np.arange(-80,80,0.5),
                               phi=[0],
                               kinetic_energy=ke)

    if "theta" not in dset_ped.columns():
       dset_ped.add_columns(theta=calc.iodata[-1].theta)
    dset_ped.add_columns(cs_gn=calc.iodata[-1].cross_section)
    dset_ped.add_columns(cs_gnd=calc.iodata[-1].direct_signal)
    view_ped.select("theta", "cs_gn", legend="SE_Gn")
#    view_ped.select("theta", "cs_gnd", legend="SE_Gn_direct")

#Order = str(NDIFF)
#TiSr = 'TiSr'
#FF = 'FF'
#FFBF = 'FFBF'
#renormalization = 'renormalization'
#if do_mi and do_se:
#    if BACKWARD_ANGLE == -1:

Result = "Result"                                    
all_data.export(os.path.join(Result, 'result_plot'))

#all_data.save(f"{''.join(symbols)}{level}.hdf5")
#    elif BACKWARD_ANGLE != -1:
#        all_data.save(f"{''.join(symbols)}{level}_{TiSr}_{suffix}_{NDIFF}_{FFBF}.hdf5")
#elif do_gn:
#    all_data.save(f"{''.join(symbols)}{level}_{TiSr}_{suffix}_{NDIFF}_{FFBF}_{renormalization}.hdf5")

all_data.view()



#def compute_scan_renorm(cluster, folder='calc_renorm', level='2s', kinetic_energy=[1000.],
#                 theta=np.arange(-60., 60.1, 0.5)):
#    calc = MSSPEC(spectroscopy='PED', algorithm='expansion', folder=folder)
#    calc.source_parameters.theta  = -45.
#    calc.source_parameters.phi    = 90
#    calc.source_parameters.energy  = 3000
#    calc.detector_parameters.angular_acceptance = 1
#    calc.detector_parameters.average_sampling   = 'high'
#    max_ndif=6
#    emitter_plane = cluster.info['emitter_plane']
#    calc.calculation_parameters.scattering_order = int(min(max_ndif,max(1, emitter_plane)))
#    calc.tmatrix_parameters.tl_threshold = 1e-4
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29, 'O': 29}
#    calc.calculation_parameters.RA_cutoff = 3
#    calc.calculation_parameters.renormalization_mode = 'G_n'
#    calc.calculation_parameters.renormalization_omega = omega_gn
#    calc.set_atoms(cluster)
#    data = calc.get_theta_scan(level=level,
#                               theta=theta,
#                               phi=0,
#                               kinetic_energy=kinetic_energy)
#    calc.shutdown()
#    return data
