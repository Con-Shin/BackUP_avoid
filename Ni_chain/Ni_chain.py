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

if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser(description='STO')
    parser.add_argument('--radius', default='5', type=float, help='radius of the cluster (default: %(default)s)')
    parser.add_argument('--termination', default='SrO', help='The SrTiO3 surface termination (default: %(default)s)')
    parser.add_argument('--emitter', default='Ti', help='The emitter symbol (default: %(default)s)')
    parser.add_argument('--level', default='2p', help='The electronic orbital (default: %(default)s)')
    parser.add_argument('--nplanes', default=1, type=int, help='The number of planes (default: %(default)d)')
    parser.add_argument('--ke_min', default=600., type=float, help='The minimum kinetic energy (default: %(default).3f)')
    parser.add_argument('--ke_max', default=1000., type=float, help='The maximum kinetic energy (default: %(default).3f)')
    parser.add_argument('--ke_num', default=10, type=int, help='The number of kinetic energy points (default: %(default)d)')
    parser.add_argument('--ke', default=1000., type=float, help='kinetic energy (default: %(default).3f)')
    parser.add_argument('--offsets', default=[0.], nargs='+', type=float, help='The Ti offsets (default: %(default)s)')
    parser.add_argument('--ncpu', default='1',type =int,  help='ncpu for lopper.run (default: %(default)s)')
    parser.add_argument('--folder', default="calc", type=str,
                        help='The folder where to store results to (default: %(default)s)')
    parser.add_argument('--view', action='store_true', help="View the cluster")
    parser.add_argument('--forward_angle', default=-1, type=float, help='The forward angle for the forward scattering filter (default: %(default).3f)')
    parser.add_argument('--backward_angle', default=-1, type=float, help='The backward angle for the backward scattering filter (default: %(default).3f)')
    parser.add_argument('--distance_cutoff', default=0., type=float, help='The distance cutoff for the distance filter (default: %(default).3f)')
    parser.add_argument('--renormalization_mode', default=None, type=str, help='The renormalization mode (default: %(default)s)')
    parser.add_argument('--omega', default=1+0j, type=complex, help='The omega parameter')
    parser.add_argument('--ndiff', default=6, type=int, help='ndiff (default: %(default)d)')
    parser.add_argument('--average_sampling', default=None, type=str, help='Degree of average sampling (default: %(default)s)')
    parser.add_argument('--potential', default='hedin', type=str, help='Potential (default: %(default)s)')
    parser.add_argument('--mt_radius_Sr', default=0.2, type=float, help='Muffintin radius of Sr (default: %(default)d)')
    parser.add_argument('--mt_radius_Ti', default=0.2, type=float, help='Muffintin radius of Ti (default: %(default)d)')
    parser.add_argument('--mt_radius_O', default=0.2, type=float, help='Muffintin radius of O (default: %(default)d)')
    parser.add_argument('--mi', default=False, type=bool, help='MI')
    parser.add_argument('--se', default=False, type=bool, help='SE')
    parser.add_argument('--gn', default=False, type=bool, help='GN')

        # Parse commandline and set global values
    args = parser.parse_args()

    RA = args.radius
    TERMINATION = args.termination
    EMITTER = args.emitter
    LEVEL = args.level
    NP = args.nplanes
#    KE = [args.ke_min, args.ke_max, args.ke_num]
    KE = args.ke
    OFFSETS = args.offsets
    FOLDER = args.folder
    FORWARD_ANGLE = args.forward_angle
    BACKWARD_ANGLE = args.backward_angle
    DISTANCE_CUTOFF = args.distance_cutoff
    RENORMALIZATION_MODE=args.renormalization_mode
    OMEGA=args.omega
    NDIFF = args.ndiff
    DEGREE = args.average_sampling
    POTENTIAL = args.potential
    RAS = args.mt_radius_Sr
    RAT = args.mt_radius_Ti
    RAO = args.mt_radius_O
    MI = args.mi
    SE = args.se
    GN = args.gn

do_mi  = True
do_se = True
do_gn  = False

# We start by defining a class factory to create perovskites that can be tetragonalized
# (usefull for later...)
class PerovskiteFactory(SimpleTetragonalFactory):
    bravais_basis = [[0, 0, 0], [0.5, 0.5, 0.5],
                     [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    element_basis = (0, 1, 2, 2, 2)
ABO3 = Perovskite = PerovskiteFactory()


def create_cluster(plane=0, termination='SrO', emitter='Ti', a=3.905, c=3.905, radius=10, Ti_offsets=[0.]):
    """ Create the clusters needed to compute a 'substrate' signal.

        The total signal is the sum of inetnsity coming from emitters in all planes of the cluster.
        Ideally, we would compute the signal coming from emitters as deep as the mean free path, but 
        it may be too much calculations. So, the keyword 'nplanes' tells how much planes containing 
        the emitter will be used.

        Parameters:
        -----------
            plane: integer
                The plane containing the emitter. For example if the emitter is 'Ti' and the surface 
                termination is 'SrO', then plane=0 means the first plane containing a Ti, which is 
                plane #1 if counting all planes from the surface (plane #0 is the surface). If plane=1,
                then it is plane #3 from the surface.
            termination: string
                The kind of surface termination of STO. Must be either 'SrO' or 'TiO2'
            emitter: string
                The emitter (or absorber) atom. Must be either 'Ti', 'Sr', or 'O'
            a: float (angstroms)
                The in-plane lattice parameter of STO
            c: float (angstroms)
                The out-of-plane lattice parameter of STO (same as 'a' for cubic cell)
            radius: float (angstroms)
                The in-plane radius of the cluster. (For normal emission, I guess this radius should
                not be too large... something to check for convergence...)
            Ti_offsets: list of float (angstroms)
                This is a list of offsets to apply to Ti atoms. Each item in the list apply to Ti atoms
                in a given plane starting from the surface. For example if the list is [0.2, -0.1], then
                Ti atoms in plane #1 of a SrO-terminated STO will be shifted by 0.2 angstroms along z and Ti 
                atoms in plane #3 will be shifted by -0.1 angstroms. Plane #0 is a SrO plane.

        Returns:
        --------
            An ase.Atoms object
    """
    # Here is the base STO cell
    primitive = Perovskite(('Sr', 'Ti', 'O'), 
                        latticeconstant={'a': a, 'c/a': c/a}, 
                        size=(1,1,1))

    # Each kind of atom will have a different tag number used to identify the emitter
    for atom in primitive:
        atom.tag = ('Sr', 'Ti', 'O').index(atom.symbol) + 1

    # Find the emitter_plane
    maxplanes = 200
    even_numbers = np.arange(0, maxplanes, 2)
    odd_numbers  = np.arange(1, maxplanes, 2)
    if ((termination == 'SrO' and emitter == 'Sr') or
        (termination == 'TiO2' and emitter == 'Ti')):
        emitter_plane = even_numbers[plane]
    elif ((termination == 'SrO' and emitter == 'Ti') or
          (termination == 'TiO2' and emitter == 'Sr')):
        emitter_plane = odd_numbers[plane]

    # Find the emitter_tag
    emitter_tag = ['Sr', 'Ti'].index(emitter) + 1

    # Create a hemispherical cluster with a quite large diameter
    cluster = hemispherical_cluster(primitive, 
                                    emitter_tag=emitter_tag, 
                                    emitter_plane=emitter_plane, 
                                    diameter=60,
                                    #planes=plane,
                                    shape='cylindrical'
                                    )
    # Reduce the diameter by cutting a cylinder of the given radius
    #cluster = cut_cylinder(cluster, radius=radius)
    cluster = cut_sphere(cluster, radius=radius)
    cluster = cut_plane(cluster, z=-1.1*c/2)
#    cluster.positions += [0,0,-19.525]
    radii = (('Sr',1.217),('Ti',1.000),('O',0.85))

    for s, r in radii:
        [atom.set('mt_radius',r) for atom in cluster if atom.symbol == s]

    ## Apply z offsets for Ti atoms (ugly but works...)
    # Get indices of Ti atoms in the cluster
    iTi = cluster.symbols.indices()['Ti']
    # Get the corresponding 'z' coordinate sorted from surface to bulk
    zTi = np.unique(cluster[iTi].positions[:,2])[::-1]
    # Apply each given offset
    for ioffset, offset in enumerate(Ti_offsets[:len(zTi)]):
        i = np.where(cluster[iTi].positions[:,2] == zTi[ioffset])[0]
        for iatom in iTi[i]:
            cluster[iatom].position[2] += offset

    # Define the absorber (when using hemispherical_cluster function, the
    # absorber is always at the origin)
    cluster.absorber = get_atom_index(cluster, 0, 0, 0)
#    cluster.rotate(180.,'x')
    
    # Add some additional information we may need later 
    cluster.info['emitter'] = emitter
    cluster.info['emitter_plane'] = emitter_plane
    cluster.info['Ti_offsets'] = Ti_offsets

    # Append this cluster to the list
    logging.info('Created {} atoms cluster'.format(len(cluster)))

    return cluster
#cluster = create_cluster(plane=4, termination='TiO2', emitter='Ti', a=3.905, c=3.905, radius=21, Ti_offsets=[0.])
#view(cluster)
#exit(0)
#cluster = create_cluster(plane=4, radius=21, termination='TiO2', emitter='Ti',
#                             Ti_offsets=[0.])
#def Cone_111():
#    cluster.rotate(-45., 'y')
#    import math
#    cos_value = 2/(math.sqrt(6))
#    T = np.arccos(cos_value)
#    t = np.degrees(T)
#    cluster.rotate(t,'x')
#    Cone_111  = Cone_making(cluster,0.1,50)
#    Cone_111.rotate(-t,'x')
#    Cone_111.rotate(45.,'y')
#    return Cone_111
a0 = 3.52
ke = 379.1
symbols = ("Ni",)
chain_range = (7,8)
absorber = 0
level = '2p'

########################### New parameters from Yoshiaki ###########################


########################### New parameters from Yoshiaki ###########################
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
    [a.set('mt_radius',1.76)]

#view(chain)
#exit(0)

#for i in range(chain_length):
#    if i%2 == 0:
#        a = cone_111[i]
#        [a.set('mt_radius',1.000)]
#    elif i%2 != 0:
#        a = cone_111[i]
#        [a.set('mt_radius',1.217)]

suffix = 'MISE'
all_data = Data(suffix)

dset_ped = all_data.add_dset(f"PED {7} atoms")
#exit(0)
view_ped = dset_ped.add_view(dset_ped.title,
                              title=(f"PED polar scan for "
                                     f"{'7 Ni atoms'}({level}) "
                                     f"@ {ke:.2f} eV"),
                              xlabel=r"$\theta$ ($\deg$)",
                              ylabel="Signal (a.u.)")
#cluster.absorber = get_atom_index(cone_111, 0, 0, 0)
#exit(0)
if do_mi:

    calc = MSSPEC(spectroscopy='PED',
                folder=(f"./{''.join(symbols)}{level}_{suffix}/"
                            f"PED_MI_{7:0d}"),algorithm='inversion')
#    exit(0)
    calc.source_parameters.theta  = 55.15
    calc.source_parameters.phi    = -44.7
    calc.source_parameters.energy  = XRaySource.AL_KALPHA
#    calc.detector_parameters.angular_acceptance = 1
#    calc.detector_parameters.average_sampling   = DEGREE #'high'

#    emitter_plane = cluster.info['emitter_plane']
#    calc.calculation_parameters.scattering_order = NDIFF
#    calc.tmatrix_parameters.tl_threshold = 1e-4

#    26/06/2024 If it doesn't work, comment out it below.
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29, 'O': 29}
#    calc.tmatrix_parameters.lmax_mode = 'imposed'
    calc.tmatrix_parameters.lmax_mode = 'true_ke'

    #Muffin-tin球の重なり. 原子の球体の重なり具合を調整する. 0.0<=x<=1.0で、値は重なりのパーセンテージ.
    calc.muffintin_parameters.radius_overlapping = 0.0

    #吸収原子がコアホールの周りでリラックスすることを許可するかどうか決定する.
    calc.muffintin_parameters.charge_relaxation = True

    calc.muffintin_parameters.interstitial_potential = 0
    # It could be better to add some vibrational damping
#    for atom in cone_111:
#        atom.set('mean_square_vibration',0.022)
#    calc.calculation_parameters.vibrational_damping = 'averaged_tl'

    # The amount of sphericity
#    calc.calculation_parameters.RA_cutoff = 1
    calc.calculation_parameters.RA_cutoff = 2
#    calc.phagen_parameters.potype = 'hedin_lundqvist_real'
    #交換・相関ポテンシャル. Inelastic effectを反映.
    calc.tmatrix_parameters.exchange_correlation = "hedin_lundqvist_real"

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
                        f"PED_SE_{7:0d}"),algorithm='expansion')

    calc.source_parameters.theta  = 55.15
    calc.source_parameters.phi    = -44.7
    calc.source_parameters.energy  = XRaySource.AL_KALPHA
#    calc.detector_parameters.angular_acceptance = 1
#    calc.detector_parameters.average_sampling   = 'low' #'high'

#    emitter_plane = cluster.info['emitter_plane']
    calc.calculation_parameters.scattering_order = 5
#    calc.tmatrix_parameters.tl_threshold = 1e-4

#    26/06/2024 If it doesn't work, comment out it below.
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29, 'O': 29}
#    calc.tmatrix_parameters.max_tl = {'Sr': 29, 'Ti': 29}
#    calc.tmatrix_parameters.lmax_mode = 'imposed'
    calc.tmatrix_parameters.lmax_mode = 'true_ke'

    calc.muffintin_parameters.interstitial_potential = 0

    #Muffin-tin球の重なり. 原子の球体の重なり具合を調整する. 0.0<=x<=1.0で、値は重なりのパーセンテージ.
    calc.muffintin_parameters.radius_overlapping = 0.0

    #吸収原子がコアホールの周りでリラックスすることを許可するかどうか決定する.
    calc.muffintin_parameters.charge_relaxation = True

    # It could be better to add some vibrational damping
#    for atom in chain:
#        atom.set('mean_square_vibration',0.022)
#    calc.calculation_parameters.vibrational_damping = 'averaged_tl'

    # The amount of sphericity
#    calc.calculation_parameters.RA_cutoff = 1
    calc.calculation_parameters.RA_cutoff = 2
#    calc.phagen_parameters.potype = 'hedin_lundqvist_real'
    #交換・相関ポテンシャル. Inelastic effectを反映.
    calc.tmatrix_parameters.exchange_correlation = "hedin_lundqvist_real"

    # Filter out paths that are outside of a given scattering cone angle (of 30° here).
    # This saves a lot of useless calculatuions
#    calc.calculation_parameters.RA_cutoff_damping = 1

#    filters = []
#    if FORWARD_ANGLE != -1:
#        filters.append('forward_scattering')
#        [a.set('forward_angle', FORWARD_ANGLE) for a in cone_111]
#    if BACKWARD_ANGLE != -1:
#        filters.append('backward_scattering')
#        [a.set('backward_angle', BACKWARD_ANGLE) for a in cone_111]
#    if DISTANCE_CUTOFF != -1:
#        filters.append('distance_cutoff')
#        calc.calculation_parameters.distance = DISTANCE_CUTOFF

#    calc.calculation_parameters.path_filtering   = filters
#    calc.calculation_parameters.off_cone_events   = 0

    # Renormalization
#    calc.calculation_parameters.renormalization_mode = RENORMALIZATION_MODE
#    calc.calculation_parameters.renormalization_omega = OMEGA


    # Attach the calculator to the cluster...
    calc.set_atoms(chain)

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
    view_ped.select("theta", "cs_se", legend="Series expansion")

if do_gn:

    calc = MSSPEC(spectroscopy='PED',
                folder=(f"./{''.join(symbols)}{level}_{suffix}/"
                        f"PED_SE_{7:0d}"),algorithm='expansion')

    calc.source_parameters.theta  = 55.15
    calc.source_parameters.phi    = -44.7
    calc.source_parameters.energy  = XRaySource.AL_KALPHA
    calc.detector_parameters.angular_acceptance = 1
    calc.detector_parameters.average_sampling   = 'low' #'high'

#    emitter_plane = cluster.info['emitter_plane']
    calc.calculation_parameters.scattering_order = 6
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
    calc.calculation_parameters.RA_cutoff = 1
    calc.phagen_parameters.potype = 'hedin'
    # Filter out paths that are outside of a given scattering cone angle (of 30° here).
    # This saves a lot of useless calculatuions
#    calc.calculation_parameters.RA_cutoff_damping = 1

#    filters = []
#    if FORWARD_ANGLE != -1:
#        filters.append('forward_scattering')
#        [a.set('forward_angle', FORWARD_ANGLE) for a in cone_111]
#    if BACKWARD_ANGLE != -1:
#        filters.append('backward_scattering')
#        [a.set('backward_angle', BACKWARD_ANGLE) for a in cone_111]
#    if DISTANCE_CUTOFF != -1:
#        filters.append('distance_cutoff')
#        calc.calculation_parameters.distance = DISTANCE_CUTOFF

#    calc.calculation_parameters.path_filtering   = filters
#    calc.calculation_parameters.off_cone_events   = 0

    # Renormalization
    calc.calculation_parameters.renormalization_mode = 'G_n'
    calc.calculation_parameters.renormalization_omega = 0.79169+0.15257j


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
    view_ped.select("theta", "cs_gn", legend="SE_Gn")

Order = str(NDIFF)
Ni = 'Ni'
#renormalization = 'renormalization'
if do_mi and do_se:
#        all_data.export(os.path.join(Result, 'results'))
        all_data.save(f"{''.join(symbols)}{level}_{Ni}_{suffix}.hdf5")
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
