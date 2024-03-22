#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Local imports.
from analytic_grad_shafranov import AnalyticGradShafranovSolution, Limiter, SingleNull, DoubleNull, ExtremalPoint, XPoint, UpDownAsymmetric

logger = logging.getLogger(__name__)

save_directory = Path(__file__).parent

'''
Input parameters:
  Major Radius [m]
  Pressure Parameter []
  Inverse Aspect Ratio []
  Elongation []
  Triangularity []
  Magnetic Field at Major Radius [T]
  Plasma Current [MA]
  Kink Safety Factor [] (optional)

One of the plasma current and the kink safety factor must be provided.
'''

# ITER baseline with beta toroidal = 0.05 (parameters chosen as in paper).
iter = (6.3, 0.155, 0.32, 1.7, 0.33, 5.3, 15, None)

# JET
jet = (2.96, 0.2, 0.422, 1.68, 0.3, 3.4, 3.2, None)

# NSTX (parameters chosen as in paper).
nstx = (0.85, 0.05, 0.78, 2.0, 0.35, 1.54, None, 2)

# MAST-U.
mast_u = (0.7, 0.1, 0.714, 2.5, 0.4, 0.8, 1, None)

# STEP baseline with beta normalised = 4.4%.
step = (3.6, 0.355, 0.556, 2.8, 0.5, 3.4, 20, None)

def plot_plasma(plasma: AnalyticGradShafranovSolution, r=None, z=None, title=None):
    e, k, R0 = plasma.inverse_aspect_ratio, plasma.elongation, plasma.major_radius_m
    if r is None:
        r = np.linspace(1 - 1.1*e, 1 + 1.1*e, 51)
    if z is None:
        z = np.linspace(-1.1*e*k, 1.1*e*k, 51)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_F = fig.add_subplot(gs[1, 0])
    ax_eq = fig.add_subplot(gs[:, 1])

    ax_p.set_xlabel("Radius [m]")
    ax_p.set_ylabel("Midplane Pressure [kPa]")
    ax_p.grid()
    ax_F.set_xlabel("Radius [m]")
    ax_F.set_ylabel("Midplane F []")
    ax_F.grid()

    # Plot pressure and F functions at the midplane.
    radius_midplane = R0 * (1 + np.linspace(-1.2*e, 1.2*e, 50))
    psiN_midplane = plasma.psi_norm(radius_midplane, 0)
    ax_p.plot(radius_midplane, plasma.pressure_kPa(psiN_midplane), color="black")
    ax_F.plot(radius_midplane, plasma.f_function(psiN_midplane), color="black")

    ax_eq.set_xlabel(r"Radius [m]")
    ax_eq.set_ylabel(r"Height [m]")
    ax_eq.set_aspect('equal')

    R, Z = r * R0, z * R0
    psiN = plasma.psi_norm(*np.meshgrid(R, Z, indexing='ij'))

    # Plot contours of normalised poloidal flux.
    c = ax_eq.contourf(R, Z, psiN.T, levels=np.linspace(0, 1.2, 13))
    ax_eq.contour(R, Z, psiN.T, colors="black", levels=np.linspace(0.1, 1, 10))

    # Mark magnetic axis location.
    ax_eq.scatter(*plasma.magnetic_axis, color="black", marker="x")
    
    fig.colorbar(c, ax=ax_eq, label="Normalised Poloidal Flux")
    fig.tight_layout()

    if title is not None:
        fig.subplots_adjust(top=0.9)
        fig.suptitle(title)
        logger.info(title)

    # Dump some interesting 0D parameters.
    logger.info(f"Major radius [m] = {plasma.major_radius_m:.2f}")
    logger.info(f"Inverse aspect ratio [] = {plasma.inverse_aspect_ratio:.2f}")
    
    if isinstance(plasma.elongation, tuple):
        logger.info(f"Upper Elongation [] = {plasma.elongation[0]:.2f}")
        logger.info(f"Lower Elongation [] = {plasma.elongation[1]:.2f}")
    else:
        logger.info(f"Elongation [] = {plasma.elongation:.2f}")
    
    if isinstance(plasma.triangularity, tuple):
        logger.info(f"Upper Triangularity [] = {plasma.triangularity[0]:.2f}")
        logger.info(f"Lower Triangularity [] = {plasma.triangularity[1]:.2f}")
    else:
        logger.info(f"Triangularity [] = {plasma.triangularity:.2f}")

    logger.info(f"Reference magnetic field [T] = {plasma.reference_magnetic_field_T:.2f}")
    logger.info(f"Plasma current [MA] = {plasma.plasma_current_MA:.2f}")
    logger.info(f"Kink Safety factor [] = {plasma.kink_safety_factor:.3f}")
    logger.info(f"Normalised circumference [] = {plasma.normalised_circumference:.3f}")
    logger.info(f"Normalised volume [] = {plasma.normalised_volume:.3f}")
    logger.info(f"Poloidal beta [] = {plasma.poloidal_beta:.3f}")
    logger.info(f"Toroidal beta [] = {plasma.toroidal_beta:.4f}")
    logger.info(f"Total beta [] = {plasma.total_beta:.4f}")
    logger.info(f"Normalised beta [%] = {100 * plasma.normalised_beta:.2f}")
    logger.info(f"Pressure parameter = {plasma.pressure_parameter:.3f}")
    logger.info(f"Psi0 = {plasma.psi_0:.3f} Wb")
    logger.info(f"Psi at magnetic axis = {plasma.psi_axis:.3f} Wb")
    logger.info(f"Magnetic axis [m] = ({plasma.magnetic_axis[0]:.3f}, {plasma.magnetic_axis[1]:.3f})")
    logger.info(f"Shafranov shift [m] = {plasma.shafranov_shift:.3f}")

    return fig, ax_eq

def iter_limiter(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = iter
    plasma = Limiter(R, A, e, k, d, B0, Ip)
    fig, ax = plot_plasma(plasma, title="ITER Baseline Limiter")
    if savefig:
        fig.savefig(save_directory.joinpath("iter_limiter.svg"))

def iter_single_null(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = iter
    r = np.linspace(1 - 1.1*e, 1 + 1.1*e, 51)
    z = np.linspace(-1.2*e*k, 1.1*e*k, 51)
    fig, ax = plot_plasma(SingleNull(R, A, e, k, d, B0, Ip), r=r, z=z, title="ITER Single Null")
    if savefig: fig.savefig(save_directory.joinpath("iter_single_null.svg"))

def jet_single_null(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = jet
    r = np.linspace(1 - 1.1*e, 1 + 1.1*e, 51)
    z = np.linspace(-1.2*e*k, 1.1*e*k, 51)
    fig, ax = plot_plasma(SingleNull(R, A, e, k, d, B0, Ip, kink_safety_factor=qstar), r=r, z=z, title="JET Single Null")
    if savefig: fig.savefig(save_directory.joinpath("jet_single_null.svg"))

def nstx_single_null(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = nstx
    r = np.linspace(1 - 1.1*e, 1 + 1.1*e, 51)
    z = np.linspace(-1.2*e*k, 1.1*e*k, 51)
    fig, ax = plot_plasma(SingleNull(R, A, e, k, d, B0, Ip, kink_safety_factor=qstar), r=r, z=z, title="NSTX Single Null")
    if savefig: fig.savefig(save_directory.joinpath("nstx_single_null.svg"))

def nstx_double_null(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = nstx
    r = np.linspace(1 - 1.1*e, 1 + 1.1*e, 51)
    z = np.linspace(-1.2*e*k, 1.2*e*k, 51)
    fig, ax = plot_plasma(DoubleNull(R, A, e, k, d, B0, Ip, kink_safety_factor=qstar), r=r, z=z, title="NSTX Double Null")
    if savefig: fig.savefig(save_directory.joinpath("nstx_double_null.svg"))

def mastu_double_null(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = mast_u
    r = np.linspace(1 - 1.2*e, 1 + 1.2*e, 51)
    z = np.linspace(-1.2*e*k, 1.2*e*k, 51)
    fig, ax = plot_plasma(DoubleNull(R, A, e, k, d, B0, Ip), r=r, z=z, title="MAST-U Double Null")
    if savefig: fig.savefig(save_directory.joinpath("mastu_double_null.svg"))  

def mastu_double_null_up_down_asymmetric(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = mast_u
    r = np.linspace(1 - 1.2*e, 1 + 1.2*e, 51)
    z = np.linspace(-1.2*e*k, 1.2*e*k, 51)

    # Upper triangularity is 5% higher.
    upper_x_point = XPoint(R * (1 - 1.05*1.1*d*e), e*k*R)
    lower_x_point = XPoint(R * (1 - 1.1*d*e), -e*k*R)

    fig, ax = plot_plasma(UpDownAsymmetric(R, A, e, upper_x_point, lower_x_point, B0, Ip), r=r, z=z, title="Up-down Asymmetric MAST-U Double Null")
    if savefig: fig.savefig(save_directory.joinpath("mastu_double_null.svg"))  

def step_double_null(savefig: bool=True):
    R, A, e, k, d, B0, Ip, qstar = step
    r = np.linspace(1 - 1.2*e, 1 + 1.2*e, 51)
    z = np.linspace(-1.2*e*k, 1.2*e*k, 51)
    fig, ax = plot_plasma(DoubleNull(R, A, e, k, d, B0, Ip), r=r, z=z, title="STEP Double Null")
    if savefig: fig.savefig(save_directory.joinpath("step_double_null.svg"))   

def main():
    savefig = False

    iter_limiter(savefig=savefig)
    iter_single_null(savefig=savefig)

    jet_single_null(savefig=savefig)

    nstx_single_null(savefig=savefig)
    nstx_double_null(savefig=savefig)

    mastu_double_null(savefig=savefig)
    mastu_double_null_up_down_asymmetric(savefig=savefig)

    step_double_null(savefig=savefig)
    
    plt.show()
    plt.close('all')

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S"
    )
    main()