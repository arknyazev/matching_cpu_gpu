import os
import sys
import numpy as np
from simsopt.field.boozermagneticfield import (
        BoozerRadialInterpolant,
        InterpolatedBoozerField,
        ShearAlfvenHarmonic,
        ShearAlfvenWavesSuperposition
        )
from simsopt.field.tracing import (
        trace_particles_boozer_perturbed,
        MaxToroidalFluxStoppingCriterion
        )
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        ALPHA_PARTICLE_CHARGE as CHARGE,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
        )
from booz_xform import Booz_xform
from stellgap import AE3DEigenvector
from mpi4py import MPI

comm = MPI.COMM_WORLD

max_t_seconds = 1e-3
boozmn_filename = 'boozmn_qhb_100.nc'
saw_file = 'mode/scaled_mode_32.935kHz.npy'
ic_folder = 'initial_conditions'
s_init = np.loadtxt(f'{ic_folder}/s0.txt', ndmin=1)
equil = Booz_xform()
equil.verbose = False
equil.read_boozmn(boozmn_filename)
nfp = equil.nfp

if comm.rank == 0:
    print("Interpolating fields...")

bri = BoozerRadialInterpolant(
        equil=equil,
        order=3,
        no_K=False
)

equil_field = InterpolatedBoozerField(
        field=bri,
        degree=3,
        srange=(0, 1, 15),
        thetarange=(0, np.pi, 15),
        zetarange=(0, 2 * np.pi / nfp, 15),
        extrapolate=True,
        nfp=nfp,
        stellsym=True,
        initialize=['modB','modB_derivs']
)

eigenvector = AE3DEigenvector.load_from_numpy(filename=saw_file)
omega = np.sqrt(eigenvector.eigenvalue)*1000
harmonic_list = []
for harmonic in eigenvector.harmonics:
    sbump = eigenvector.s_coords
    bump = harmonic.amplitudes
    sah = ShearAlfvenHarmonic(
        Phihat_value_or_tuple=(sbump, bump),
        Phim=harmonic.m,
        Phin=harmonic.n,
        omega=omega,
        phase=0.0,
        B0=equil_field
    )
    harmonic_list.append(sah)
saw = ShearAlfvenWavesSuperposition(harmonic_list)

VELOCITY = np.sqrt(2 * ENERGY / MASS)
if comm.rank == 0:
    print('Prepare initial conditions')
    s_init = np.loadtxt(f'{ic_folder}/s0.txt', ndmin=1)
    theta_init = np.loadtxt(f'{ic_folder}/theta0.txt', ndmin=1)
    zeta_init = np.loadtxt(f'{ic_folder}/zeta0.txt', ndmin=1)
    vpar_init = np.loadtxt(f'{ic_folder}/vpar0.txt', ndmin=1)
    points = np.zeros((s_init.size, 3))
    points[:, 0] = s_init
    points[:, 1] = theta_init
    points[:, 2] = zeta_init
    np.savetxt('points.txt', points)
    saw.B0.set_points(points)
    mu_per_mass = (VELOCITY**2 - vpar_init**2) / (2 * saw.B0.modB()[:,0])
else:
    points = None
    vpar_init = None
    mu_per_mass = None

points = comm.bcast(points, root=0)
vpar_init = comm.bcast(vpar_init, root=0)
mu_per_mass = comm.bcast(mu_per_mass, root=0)

if comm.rank == 0:
    print('Begin particle tracking...')

gc_tys, gc_hits = trace_particles_boozer_perturbed(
        perturbed_field=saw,
        stz_inits=points,
        parallel_speeds=vpar_init,
        mus=mu_per_mass,
        tmax=max_t_seconds,
        mass=MASS,
        charge=CHARGE,
        Ekin=ENERGY,
        abstol=1e-5,
        reltol=1e-9,
        comm=comm,
        zetas=[],
        omegas=[],
        vpars=[],
        stopping_criteria=[
            MaxToroidalFluxStoppingCriterion(0.9)
        ],
        dt_save = 1e-7,
        forget_exact_path=True,
        zetas_stop=False,
        vpars_stop=False,
        axis=2
        )

if comm.rank == 0:
    results = {
            'timelost' : [],
            's0' : [],
            'theta0' : [],
            'zeta0' : [],
            'vpar0' : [],
            'slost' : [],
            'thetalost' : [],
            'zetalost' : [],
            'vparlost' : []
        }
    print('Save IDs of lost particles')
    for i in range(len(gc_tys)):
        results['s0'].append(gc_tys[i][0,1])
        results['theta0'].append(gc_tys[i][0,2])
        results['zeta0'].append(gc_tys[i][0,3])
        results['vpar0'].append(gc_tys[i][0,4])
        results['timelost'].append(gc_tys[i][-1,0])
        results['slost'].append(gc_tys[i][-1,1])
        results['thetalost'].append(gc_tys[i][-1,2])
        results['zetalost'].append(gc_tys[i][-1,3])
        results['vparlost'].append(gc_tys[i][-1,4])
        lost_IDs = []
    for name, array in results.items():
        np.savetxt(f'output/{name}.txt', array)
if comm.rank == 0:
    print("All done.")
