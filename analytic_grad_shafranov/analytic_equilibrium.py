#!/usr/bin/python3

# Standard imports
import abc
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.constants as const
from typing import Tuple

# Local imports

logger = logging.getLogger(__name__)

class AnalyticGradShafranovSolution(abc.ABC):
    '''
    Analytic solutions to the Grad Shafranov equation as described in 'Antoine J. Cerfon, Jeffrey P. Freidberg;
    “One size fits all” analytic solutions to the Grad–Shafranov equation. Phys. Plasmas 1 March 2010; 17 (3): 032502.'.

    Some parameters have been flipped sign compared to the paper, see the README of this project for a full description.
    '''
    __slots__ = (
        "major_radius_m", "pressure_parameter", "coefficients", "inverse_aspect_ratio", "elongation",
        "triangularity", "reference_magnetic_field_T", "plasma_current_MA", "psi_0",
        "normalised_circumference", "normalised_volume", "poloidal_beta", "toroidal_beta",
        "total_beta", "normalised_beta", "kink_safety_factor", "psi_axis", "magnetic_axis",
        "plasma_current_anticlockwise", "toroidal_field_anticlockwise"
    )

    def __init__(
        self,
        major_radius_m: float,
        pressure_parameter: float,
        inverse_aspect_ratio: float,
        elongation: float,
        triangularity: float,
        reference_magnetic_field_T: float,
        plasma_current_MA: float,
        kink_safety_factor: float = None,
        plasma_current_anticlockwise: bool = False,
        toroidal_field_anticlockwise: bool = False,
    ):
        '''
        Parameters
        ----------
        major_radius_m: float
            Major radius of plasma R0 [m].
        pressure_parameter: float
            Parameter controlling the size of the pressure function A.
            Larger A gives a higher pressure. This can only really be set
            via trial and error to get the desired beta.
        inverse_aspect_ratio: float
            Inverse aspect ratio epsilon [] = minor radius [m] / major radius [m].
        elongation: float
            Plasma elongation kappa [].
        triangularity: float
            Plasma triangularity delta [].
        reference_magnetic_field_T: float
            Magnetic field strength at the geometric axis R = R0 [T].
        plasma_current_MA: float
            Total plasma current [MA].
        kink_safety_factor: float, optional
            Kink safety factor q_star. If None (default), this is calculated using the plasma current.
            Otherwise the value of the plasma current is calculated using the provided value.
        '''
        self.major_radius_m: float = float(major_radius_m)
        self.pressure_parameter: float = float(pressure_parameter)
        self.inverse_aspect_ratio: float = float(inverse_aspect_ratio)
        self.elongation: float = float(elongation)
        self.triangularity: float = float(triangularity)
        self.reference_magnetic_field_T: float = float(reference_magnetic_field_T)
        plasma_current_anticlockwise: bool = plasma_current_anticlockwise
        toroidal_field_anticlockwise: bool = toroidal_field_anticlockwise

        # Solve for the weighting coefficients for each of the polynomials.
        self.calculate_coefficients()
        
        # Calculate the normalised circumference and volume.
        self.calculate_geometry_factors()

        # Use either the plasma current or kink safety factor to calculate the other.
        e, B0 = self.inverse_aspect_ratio, self.reference_magnetic_field_T
        R0, Cp = self.major_radius_m, self.normalised_circumference

        if kink_safety_factor is None:
            self.plasma_current_MA: float = float(plasma_current_MA)
            self.kink_safety_factor = e*B0*R0*Cp / const.mu_0 / (1e6 * self.plasma_current_MA)
        else:
            self.kink_safety_factor = float(kink_safety_factor)
            self.plasma_current_MA = 1e-6 * e*B0*R0*Cp / const.mu_0 / self.kink_safety_factor
        
        # Set dummy value of psi axis. This will be set in calculate_metrics() to match the prescribed plasma current.
        self.psi_0 = 1.0
        self.calculate_metrics()
    
    @abc.abstractmethod
    def calculate_coefficients(self):
        ''' Solve for the weighting coefficients of the polynomials defining psi. '''
        pass

    @staticmethod
    def psi_homogenous(x: float, y: float) -> Tuple[float]:
        '''
        7 homogenous solutions of the normalised Grad-Shafranov equation expanded up to order x^6 and y^6 which are
        even in y. x = R / R0 and y = Z / R0 are the radius R and height Z normalised to the major radius R0.
        '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        x4, y4 = x2**2, y2**2

        p1 = 1
        p2 = x2
        p3 = y2 - x2*ln_x
        p4 = x2 * (x2 - 4*y2)
        p5 = y2 * (2*y2 - 9*x2) + x2*ln_x*(3*x2 - 12*y2)
        p6 = x2 * (x4 - 12*x2*y2 + 8*y4)
        p7 = (-15*x4 + 180*y2*x2 - 120*y4)*x2*ln_x + (75*x4 - 140*x2*y2 + 8*y4)*y2

        return p1, p2, p3, p4, p5, p6, p7
    @staticmethod
    def psi_homogenous_dx(x: float, y: float) -> Tuple[float]:
        ''' First derivative of the homogeneous polynomials with respect to x. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        x4, y4 = x2**2, y2**2

        dp1_dx = 0
        dp2_dx = 2 * x
        dp3_dx = -x * (1 + 2*ln_x)
        dp4_dx = x * (4 * x2 - 8 * y2)
        dp5_dx = 3 * x * ((4*x2 - 8*y2) * ln_x + x2 - 10*y2)
        dp6_dx = x * (6*x4 - 48*x2*y2 + 16*y4)
        dp7_dx = -5*x*((18*x4 - 144*x2*y2 + 48*y4)*ln_x + 3*x4 - 96*x2*y2+80*y4)

        return dp1_dx, dp2_dx, dp3_dx, dp4_dx, dp5_dx, dp6_dx, dp7_dx
    @staticmethod
    def psi_homogenous_dy(x: float, y: float) -> Tuple[float]:
        ''' First derivative of the homogeneous polynomials with respect to y. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        x4, y4 = x2**2, y2**2

        dp1_dy = 0
        dp2_dy = 0
        dp3_dy = 2 * y
        dp4_dy = -8*x2*y
        dp5_dy = y * (8*y2 - x2 * (18 + 24*ln_x))
        dp6_dy = y * (-24*x4 + 32*x2*y2)
        dp7_dy = y * (48*y4 - (480*ln_x + 560)*x2*y2 + (360*ln_x + 150)*x4)

        return dp1_dy, dp2_dy, dp3_dy, dp4_dy, dp5_dy, dp6_dy, dp7_dy
    @staticmethod
    def psi_homogenous_dx2(x: float, y: float) -> Tuple[float]:
        ''' Second derivative of the homogeneous polynomials with respect to x. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        x4, y4 = x2**2, y2**2

        d2p1_dx2 = 0
        d2p2_dx2 = 2
        d2p3_dx2 = -3 - 2*ln_x
        d2p4_dx2 = 12*x2 - 8*y2
        d2p5_dx2 = (36*x2 - 24*y2) * ln_x + 21*x2 - 54*y2
        d2p6_dx2 = 30*x4 - 144*x2*y2 + 16*y4
        d2p7_dx2 = (-450*x4 + 2160*x2*y2 - 240*y4)*ln_x - 165*x4 + 2160*x2*y2 - 640*y4

        return d2p1_dx2, d2p2_dx2, d2p3_dx2, d2p4_dx2, d2p5_dx2, d2p6_dx2, d2p7_dx2
    @staticmethod
    def psi_homogenous_dy2(x: float, y: float) -> Tuple[float]:
        ''' Second derivative of the homogeneous polynomials with respect to x. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        x4, y4 = x2**2, y2**2

        d2p1_dy2 = 0
        d2p2_dy2 = 0
        d2p3_dy2 = 2
        d2p4_dy2 = -8*x2
        d2p5_dy2 = 24*y2 - x2 * (18 + 24*ln_x)
        d2p6_dy2 = x2 * (-24*x2 + 96*y2)
        d2p7_dy2 = 240*y4 - (1440*ln_x + 1680)*x2*y2 + (360*ln_x + 150)*x4

        return d2p1_dy2, d2p2_dy2, d2p3_dy2, d2p4_dy2, d2p5_dy2, d2p6_dy2, d2p7_dy2
    def psi_particular(self, x: float, y: float) -> float:
        ''' 
        Particular solution of the normalised Grad-Shafranov equation. x = R / R0 and y = Z / R0 are
        the radius R and height Z normalised to the major radius R0.
        '''
        A = self.pressure_parameter
        x2, ln_x = x**2, np.log(x)
        x4 = x2**2
        return 0.5 * A * x2 * ln_x - (x4 / 8) * (1 + A)
    def psi_particular_dx(self, x: float, y: float) -> float:
        ''' First derivative of the particular solution with respect to x. '''
        A = self.pressure_parameter
        x2, ln_x = x**2, np.log(x)
        return 0.5 * x * (A*(2*ln_x + 1) - x2*(1 + A))
    def psi_particular_dx2(self, x: float, y: float) -> float:
        ''' Second derivative of the particular solution with respect to x. '''
        x2, ln_x = x**2, np.log(x)
        A = self.pressure_parameter
        return A * (1.5 + ln_x) - 1.5 * (1 + A) * x2
    
    def psi_bar(self, x: float, y: float) -> float:
        '''
        Poloidal flux function normalised to psi0. This is NOT the commonly encountered psi normalised psiN!
        psi_bar is zero at the separatrix and some positive value at the magnetic axis.
        '''
        psi = self.psi_particular(x, y)

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous(x, y)):
            psi += c * p

        return psi
    def psi_bar_dx(self, x: float, y: float) -> float:
        '''
        First derivative of poloidal flux function normalised to psi0 with respect to x.        
        '''
        psi_dx = self.psi_particular_dx(x, y)

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous_dx(x, y)):
            psi_dx += c * p

        return psi_dx
    def psi_bar_dy(self, x: float, y: float) -> float:
        '''
        First derivative of poloidal flux function normalised to psi0 with respect to y.        
        '''
        psi_dy = 0

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous_dy(x, y)):
            psi_dy += c * p

        return psi_dy
    def psi_bar_dx2(self, x: float, y: float) -> float:
        '''
        Second derivative of poloidal flux function normalised to psi0 with respect to x.        
        '''
        psi_dx2 = self.psi_particular_dx2(x, y)

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous_dx2(x, y)):
            psi_dx2 += c * p

        return psi_dx2
    def psi_bar_dy2(self, x: float, y: float) -> float:
        '''
        Second derivative of poloidal flux function normalised to psi0 with respect to y.        
        '''
        psi_dy2 = 0

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous_dy2(x, y)):
            psi_dy2 += c * p

        return psi_dy2
    
    def psi(self, R: float, Z: float) -> float:
        ''' Poloidal flux function [Wb]. '''
        R0 = self.major_radius_m
        return self.psi_0 * self.psi_bar(R / R0, Z / R0)
    def psi_dR(self, R: float, Z: float) -> float:
        ''' First derivative of the poloidal flux function with respect to R [Wb / m]. '''
        R0 = self.major_radius_m
        return self.psi_0 * self.psi_bar_dx(R / R0, Z / R0) / R0
    def psi_dZ(self, R: float, Z: float) -> float:
        ''' First derivative of the poloidal flux function with respect to Z [Wb / m]. '''
        R0 = self.major_radius_m
        return self.psi_0 * self.psi_bar_dy(R / R0, Z / R0) / R0
    def magnetic_field(self, R: float, Z: float) -> Tuple[float, float, float]:
        ''' (R, phi, Z) components of the magnetic field [T]. '''
        psi_norm = self.psi_norm(R, Z)
        B_R = -self.psi_dZ(R, Z) / R
        B_Z = self.psi_dR(R, Z) / R
        B_toroidal = self.f_function(psi_norm) / R

        return B_R, B_toroidal, B_Z

    def psi_norm(self, R: float, Z: float) -> float:
        R0 = self.major_radius_m
        return self.psi_bar_to_psi_norm(self.psi_bar(R / R0, Z / R0))
    
    def d_shape_boundary(self, theta: float) -> float:
        ''' D shaped boundary contour for the prescribed geometry factors. '''
        e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity
        alpha = np.arcsin(d)
        x = 1 + e * np.cos(theta + alpha*np.sin(theta))
        y = e * k * np.sin(theta)

        return x, y
    def calculate_geometry_factors(self, use_d_shaped_model: bool=True):
        '''
        Calculate the normalised circumference and volume of the plasma. Can either calculate based on the estimated
        boundary contour from the D shaped model or from the fitted poloidal flux function.
        '''
        if use_d_shaped_model:
            e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity
            alpha = np.arcsin(d)

            theta = np.linspace(0, np.pi, 101)
            x, y = self.d_shape_boundary(theta)
            xprime = -e * np.sin(theta + (alpha*np.sin(theta))) * (1 + alpha*np.cos(theta))
            yprime = e * k * np.cos(theta)
            rprime = (xprime**2 + yprime**2)**0.5

            self.normalised_circumference = 2 * np.trapz(rprime, theta)
            self.normalised_volume = -2 * np.trapz(x*xprime*y, theta)
        else:
            raise NotImplementedError("Normalised circumference not calculated.")
            x, y = self.metric_computation_grid()
            x_grid, ygrid = np.meshgrid(x, y, indexing='ij')
            dxdy = (x[1] - x[0]) * (y[1] - y[0])

            psiN = self._psi_bar(x_grid, ygrid) * x_grid
            mask = psiN > 0
            self.normalised_volume = np.sum(mask) * dxdy
    def metric_computation_grid(self) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        '''
        Grid of normalised radius x = R / R0 and height y = Z / R0 used to calculate numerical integrals
        for calculating the plasma current and plasma beta.
        '''
        e, k = self.inverse_aspect_ratio, self.elongation
        x = np.linspace(1 - e, 1 + e, 100)
        y = np.linspace(-e*k, e*k, 101)
        return x, y
    def calculate_metrics(self):
        '''
        Calculate the poloidal flux normalisation and the plasma 'figures of merit':
        beta_poloidal, beta_toroidal, beta_total and beta_normalised.
        '''
        e, k = self.inverse_aspect_ratio, self.elongation
        Cp, V = self.normalised_circumference, self.normalised_volume
        R0, B0 = self.major_radius_m, self.reference_magnetic_field_T
        A, qstar = self.pressure_parameter, self.kink_safety_factor

        # Calculate two integrals appearing in expressions for beta and plasma current.
        x, y = self.metric_computation_grid()
        x_grid, ygrid = np.meshgrid(x, y, indexing='ij')
        dxdy = (x[1] - x[0]) * (y[1] - y[0])

        # Ignore contribution from anything outside the separatrix.
        psi_bar = self.psi_bar(x_grid, ygrid)
        mask = psi_bar < 0

        # This is proportional to the total plasma current.
        Ip_integrand = -(A / x_grid) + (1 + A) * x_grid
        Ip_integrand[mask] = 0
        Ip_integral = np.sum(Ip_integrand) * dxdy

        # This is proportional to the average plasma pressure.
        psix_integrand = psi_bar * x_grid
        psix_integrand[mask] = 0
        psix_integral = np.sum(psix_integrand) * dxdy
        
        # Calculate beta of plasma.
        self.poloidal_beta = (2 * Cp**2 * (1 + A) / V) * psix_integral / Ip_integral**2
        self.toroidal_beta = self.poloidal_beta * e**2 / qstar**2
        self.total_beta = self.poloidal_beta * e**2 / (e**2 + qstar**2)
        self.normalised_beta = (e * R0 * B0 / self.plasma_current_MA) * self.total_beta

        # Calculate value of psi at magnetic axis for given plasma current.
        self.psi_0 = self.plasma_current_MA * 1e6 * const.mu_0 * self.major_radius_m / Ip_integral

        # Find value of psi on magnetic axis. Know it lies on Z=0 so use Netwon's method to find where d(psi)/dx = 0.
        magnetic_axis = np.array([1.0, 0.0])
        correction = np.zeros(2)

        max_iterations = 100
        tol = 1e-4
        for i in range(max_iterations):
            psi_dx = self.psi_bar_dx(*magnetic_axis)
            psi_dy = self.psi_bar_dy(*magnetic_axis)
            psi_dx2 = self.psi_bar_dx2(*magnetic_axis)
            psi_dy2 = self.psi_bar_dy2(*magnetic_axis)

            correction[:] = (psi_dx / psi_dx2), (psi_dy / psi_dy2)
            magnetic_axis -= correction

            if np.linalg.norm(correction) < tol:
                logger.info(f"Found magnetic axis in {i+1} iterations.")
                self.magnetic_axis = magnetic_axis * self.major_radius_m
                break
        
        if i == max_iterations - 1:
            logger.error(f"Failed to find magnetic axis within {max_iterations} iterations.")

        # Evaluate value of psi_norm at the magnetic axis.
        self.psi_axis = self.psi(*self.magnetic_axis)
    @property
    def shafranov_shift(self) -> float:
        return self.magnetic_axis[0] - self.major_radius_m

    def psi_bar_to_psi_norm(self, psi_bar: float) -> float:
        ''' Convert psi_bar parameter used in the normalised Grad Shafranov equation to the standard normalised poloidal flux co-ordinate. '''
        return 1 - (psi_bar * self.psi_0 / self.psi_axis)
    def psi_norm_to_psi_bar(self, psi_norm: float) -> float:
        ''' Convert normalised poloidal flux co-ordinate to the psi_bar parameter used in the normalised Grad Shafranov equation. '''
        return (1 - psi_norm) * self.psi_axis / self.psi_0
    def pressure_kPa(self, psi_norm: float):
        ''' Plasma pressure as a function of the normalised poloidal flux [kPa]. '''
        A = self.pressure_parameter
        # Clip psi to 0 so there is not negative pressure.
        psi_bar = np.clip(self.psi_norm_to_psi_bar(psi_norm), 0, None)
        return 1e-3 * (self.psi_0**2 / self.major_radius_m**4 / const.mu_0) * (1 + A) * psi_bar
    def f_function(self, psi_norm: float):
        ''' F function radius * toroidal magnetic field as a function of the normalised poloidal flux [Wb/m]. '''
        R0, B0 = self.major_radius_m, self.reference_magnetic_field_T
        psi0, A = self.psi_0, self.pressure_parameter
        # Clip psi to 0 to avoid unphysical magnetic fields.
        psi_bar = np.clip(self.psi_norm_to_psi_bar(psi_norm), 0, None)
        return R0 * (B0**2 - (2*psi0**2 / R0**4) * A * psi_bar)**0.5
    def toroidal_current_density_kA_per_m2(self, R, Z):
        ''' Toroidal current density [kA m^-2]. '''
        x = R / R0
        R0, psi0, A = self.major_radius_m, self.psi_0, self.pressure_parameter
        return 1e-3 * psi0 * ((1 + A) * x**2 - A / x) / (const.mu_0 * R0**3)

class Limiter(AnalyticGradShafranovSolution):
    def calculate_coefficients(self):
        '''
        Solve for the weighting coefficients of the polynomials defining psi. We fit to a d shaped contour with the
        required geometry factors at 3 points:
            Inner equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            Outer equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            High point: point of maximum Z on the boundary contour.
        '''
        e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity
        
        # Some coefficients from D shaped model.
        alpha = np.arcsin(d)
        N1 = - (1 + alpha)**2 / e / k**2
        N2 = (1 - alpha)**2 / e / k**2
        N3 = -k / e / (1 - d**2)

        # Points to fit D shaped model at.
        x_eq_inner, y_eq_inner = 1 - e, 0
        x_eq_outer, y_eq_outer = 1 + e, 0
        x_high, y_high = 1 - d*e, k*e

        # We solve the system y = Mx to find the coefficient vector x.
        M = np.zeros((7, 7))
        y = np.zeros(7)

        # Outer equatorial point (psi = 0).
        M[0] = self.psi_homogenous(x_eq_outer, y_eq_outer)
        y[0] = -self.psi_particular(x_eq_outer, y_eq_outer)

        # Inner equatorial point (psi = 0).
        M[1] = self.psi_homogenous(x_eq_inner, y_eq_inner)
        y[1] = -self.psi_particular(x_eq_inner, y_eq_inner)

        # High point (psi = 0).
        M[2] = self.psi_homogenous(x_high, y_high)
        y[2] = -self.psi_particular(x_high, y_high)

        # High point maximum (d(psi)/dx = 0).
        M[3] = self.psi_homogenous_dx(x_high, y_high)
        y[3] = -self.psi_particular_dx(x_high, y_high)

        # Outer equatorial point curvature (d^2(psi)/dy^2 + N1 * d(psi)/dx = 0).
        M[4] = np.array(self.psi_homogenous_dy2(x_eq_outer, y_eq_outer)) + N1 * np.array(self.psi_homogenous_dx(x_eq_outer, y_eq_outer))
        y[4] = -N1 * self.psi_particular_dx(x_eq_outer, y_eq_outer)

        # Inner equatorial point curvature (d^2(psi)/dy^2 + N2 * d(psi)/dx = 0).
        M[5] = np.array(self.psi_homogenous_dy2(x_eq_inner, y_eq_inner)) + N2 * np.array(self.psi_homogenous_dx(x_eq_inner, y_eq_inner))
        y[5] = -N2 * self.psi_particular_dx(x_eq_inner, y_eq_inner)

        # High point curvature (d^2(psi)/dx^2 + N3 * d(psi)/dy = 0).
        M[6] = np.array(self.psi_homogenous_dx2(x_high, y_high)) + N3 * np.array(self.psi_homogenous_dy(x_high, y_high))
        y[6] = -self.psi_particular_dx2(x_high, y_high)

        self.coefficients = np.linalg.solve(M, y)

class DoubleNull(AnalyticGradShafranovSolution):
    def calculate_coefficients(self):
        '''
        Solve for the weighting coefficients of the polynomials defining psi. We fit to a d shaped contour with the
        required geometry factors at 3 points:
            Inner equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            Outer equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            Upper X point: point of maximum Z on the boundary contour.
        '''
        e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity

        # Some coefficients from D shaped model.
        alpha = np.arcsin(d)
        N1 = - (1 + alpha)**2 / e / k**2
        N2 = (1 - alpha)**2 / e / k**2

        # Points to fit D shaped model at.
        x_eq_inner, y_eq_inner = 1 - e, 0
        x_eq_outer, y_eq_outer = 1 + e, 0
        x_sep, y_sep = 1 - 1.1*d*e, 1.1*k*e

        # We solve the system y = Mx to find the coefficient vector x.
        M = np.zeros((7, 7))
        y = np.zeros(7)

        # Outer equatorial point (psi = 0).
        M[0] = self.psi_homogenous(x_eq_outer, y_eq_outer)
        y[0] = -self.psi_particular(x_eq_outer, y_eq_outer)

        # Inner equatorial point (psi = 0).
        M[1] = self.psi_homogenous(x_eq_inner, y_eq_inner)
        y[1] = -self.psi_particular(x_eq_inner, y_eq_inner)

        # Outer equatorial point curvature (d^2(psi)/dy^2 + N1 * d(psi)/dx = 0).
        M[2] = np.array(self.psi_homogenous_dy2(x_eq_outer, y_eq_outer)) + N1 * np.array(self.psi_homogenous_dx(x_eq_outer, y_eq_outer))
        y[2] = -N1 * self.psi_particular_dx(x_eq_outer, y_eq_outer)

        # Inner equatorial point curvature (d^2(psi)/dy^2 + N2 * d(psi)/dx = 0).
        M[3] = np.array(self.psi_homogenous_dy2(x_eq_inner, y_eq_inner)) + N2 * np.array(self.psi_homogenous_dx(x_eq_inner, y_eq_inner))
        y[3] = -N2 * self.psi_particular_dx(x_eq_inner, y_eq_inner)

        # X point (psi = 0).
        M[4] = self.psi_homogenous(x_sep, y_sep)
        y[4] = -self.psi_particular(x_sep, y_sep)

        # B poloidal = 0 at X point (d(psi)/dx = 0).
        M[5] = self.psi_homogenous_dx(x_sep, y_sep)
        y[5] = -self.psi_particular_dx(x_sep, y_sep)

        # B poloidal = 0 at X point (d(psi)/dy = 0).
        M[6] = self.psi_homogenous_dy(x_sep, y_sep)

        self.coefficients = np.linalg.solve(M, y)
    def metric_computation_grid(self):
        e, k = self.inverse_aspect_ratio, self.elongation
        x = np.linspace(1 - e, 1 + e, 100)
        y = np.linspace(-1.1*e*k, 1.1*e*k, 101)
        return x, y

class SingleNull(AnalyticGradShafranovSolution):
    @staticmethod
    def psi_homogenous(x: float, y: float):
        '''
        12 homogenous solutions of the normalised Grad-Shafranov equation expanded up to order x^6 and y^6.
        x = R / R0 and y = Z / R0 are the radius R and height Z normalised to the major radius R0. Solutions which
        are odd in y are also now included.
        '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        x4 = x2**2

        p1, p2, p3, p4, p5, p6, p7 = AnalyticGradShafranovSolution.psi_homogenous(x, y)
        p8 = y
        p9 = y*x2
        p10 = y * (y2 - 3*x2*ln_x)
        p11 = y * (3*x4 - 4*y2*x2)
        p12 = y * ((8*y2 - 80*x2*ln_x)*y2 + (60*ln_x - 45)*x4)

        return p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12
    @staticmethod
    def psi_homogenous_dx(x: float, y: float):
        ''' First derivative of the homogeneous polynomials with respect to x. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)

        dp1_dx, dp2_dx, dp3_dx, dp4_dx, dp5_dx, dp6_dx, dp7_dx = AnalyticGradShafranovSolution.psi_homogenous_dx(x, y)
        dp8_dx = 0
        dp9_dx = 2*x*y
        dp10_dx = -3*x*y*(2*ln_x + 1)
        dp11_dx = x*y*(12*x2 - 8*y2)
        dp12_dx = 40*x*y*((6*x2 - 4*y2)*ln_x - 3*x2 - 2*y2)

        return dp1_dx, dp2_dx, dp3_dx, dp4_dx, dp5_dx, dp6_dx, dp7_dx, dp8_dx, dp9_dx, dp10_dx, dp11_dx, dp12_dx
    @staticmethod
    def psi_homogenous_dx2(x: float, y: float):
        ''' Second derivative of the homogeneous polynomials with respect to x. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)

        d2p1_dx2, d2p2_dx2, d2p3_dx2, d2p4_dx2, d2p5_dx2, d2p6_dx2, d2p7_dx2 = AnalyticGradShafranovSolution.psi_homogenous_dx2(x, y)
        d2p8_dx2 = 0
        d2p9_dx2 = 2*y
        d2p10_dx2 = -3*y*(2*ln_x + 3)
        d2p11_dx2 = y*(36*x2 - 8*y2)
        d2p12_dx2 = y*((720*x2 - 160*y2)*ln_x - 120*x2-240*y2)

        return d2p1_dx2, d2p2_dx2, d2p3_dx2, d2p4_dx2, d2p5_dx2, d2p6_dx2, d2p7_dx2, d2p8_dx2, d2p9_dx2, d2p10_dx2, d2p11_dx2, d2p12_dx2
    @staticmethod
    def psi_homogenous_dy(x: float, y: float):
        ''' First derivative of the homogeneous polynomials with respect to y. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        y4 = y2**2

        dp1_dy, dp2_dy, dp3_dy, dp4_dy, dp5_dy, dp6_dy, dp7_dy = AnalyticGradShafranovSolution.psi_homogenous_dy(x, y)
        dp8_dy = 1
        dp9_dy = x2
        dp10_dy = 3*(y2 - x2*ln_x)
        dp11_dy = x2*(3*x2 - 12*y2)
        dp12_dy = 40*y4 + 15*x2*((-16*y2 + 4*x2)*ln_x - 3*x2)

        return dp1_dy, dp2_dy, dp3_dy, dp4_dy, dp5_dy, dp6_dy, dp7_dy, dp8_dy, dp9_dy, dp10_dy, dp11_dy, dp12_dy
    @staticmethod
    def psi_homogenous_dy2(x: float, y: float):
        ''' Second derivative of the homogeneous polynomials with respect to y. '''
        x2, y2, ln_x = x**2, y**2, np.log(x)

        d2p1_dy2, d2p2_dy2, d2p3_dy2, d2p4_dy2, d2p5_dy2, d2p6_dy2, d2p7_dy2 = AnalyticGradShafranovSolution.psi_homogenous_dy2(x, y)
        d2p8_dy2 = 0
        d2p9_dy2 = 0
        d2p10_dy2 = 6*y
        d2p11_dy2 = -24*x2*y
        d2p12_dy2 = y*(160*y2- 480*x2*ln_x)

        return d2p1_dy2, d2p2_dy2, d2p3_dy2, d2p4_dy2, d2p5_dy2, d2p6_dy2, d2p7_dy2, d2p8_dy2, d2p9_dy2, d2p10_dy2, d2p11_dy2, d2p12_dy2
    def calculate_coefficients(self):
        '''
        Solve for the weighting coefficients of the polynomials defining psi. We fit to a d shaped contour with the
        required geometry factors at 3 points:
            Inner equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            Outer equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            High point: point of maximum Z on the boundary contour.
            Upper X point: point of maximum Z on the boundary contour.
        '''
        e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity

        # Some coefficients from D shaped model.
        alpha = np.arcsin(d)
        N1 = - (1 + alpha)**2 / e / k**2
        N2 = (1 - alpha)**2 / e / k**2
        N3 = -k / e / (1 - d**2)

        # Points to fit D shaped model at.
        x_eq_inner, y_eq_inner = 1 - e, 0
        x_eq_outer, y_eq_outer = 1 + e, 0

        x_high, y_high = 1 - d*e, k*e
        x_sep, y_sep = 1 - 1.1*d*e, -1.1*k*e

        # We solve the system y = Mx to find the coefficient vector x.
        M = np.zeros((12, 12))
        y = np.zeros(12)

        # Outer equatorial point (psi = 0).
        M[0] = self.psi_homogenous(x_eq_outer, y_eq_outer)
        y[0] = -self.psi_particular(x_eq_outer, y_eq_outer)

        # Inner equatorial point (psi = 0).
        M[1] = self.psi_homogenous(x_eq_inner, y_eq_inner)
        y[1] = -self.psi_particular(x_eq_inner, y_eq_inner)

        # Upper high point (psi = 0).
        M[2] = self.psi_homogenous(x_high, y_high)
        y[2] = -self.psi_particular(x_high, y_high)

        # Lower X point (psi = 0).
        M[3] = self.psi_homogenous(x_sep, y_sep)
        y[3] = -self.psi_particular(x_sep, y_sep)

        # Outer equatorial point up down symmetry (d(psi)/dy = 0).
        M[4] = self.psi_homogenous_dy(x_eq_outer, y_eq_outer)

        # Inner equatorial point up down symmetry (d(psi)/dy = 0).
        M[5] = self.psi_homogenous_dy(x_eq_inner, y_eq_inner)

        # Upper high point maximum (d(psi)/dx = 0).
        M[6] = self.psi_homogenous_dx(x_high, y_high)
        y[6] = -self.psi_particular_dx(x_high, y_high)

        # B poloidal = 0 at lower X point (d(psi)/dx = 0)).
        M[7] = self.psi_homogenous_dx(x_sep, y_sep)
        y[7] = -self.psi_particular_dx(x_sep, y_sep)

        # B poloidal = 0 at lower X point (d(psi)/dy = 0).
        M[8] = self.psi_homogenous_dy(x_sep, y_sep)

        # Outer equatorial point curvature (d^2(psi)/dy^2 + N1 * d(psi)/dx = 0).
        M[9] = np.array(self.psi_homogenous_dy2(x_eq_outer, y_eq_outer)) + N1 * np.array(self.psi_homogenous_dx(x_eq_outer, y_eq_outer))
        y[9] = -N1 * self.psi_particular_dx(x_eq_outer, y_eq_outer)

        # Inner equatorial point curvature (d^2(psi)/dy^2 + N2 * d(psi)/dx = 0).
        M[10] = np.array(self.psi_homogenous_dy2(x_eq_inner, y_eq_inner)) + N2 * np.array(self.psi_homogenous_dx(x_eq_inner, y_eq_inner))
        y[10] = -N2 * self.psi_particular_dx(x_eq_inner, y_eq_inner)

        # High point curvature (d^2(psi)/dx^2 + N3 * d(psi)/dy = 0).
        M[11] = np.array(self.psi_homogenous_dx2(x_high, y_high)) + N3 * np.array(self.psi_homogenous_dy(x_high, y_high))
        y[11] = -self.psi_particular_dx2(x_high, y_high)

        self.coefficients = np.linalg.solve(M, y)
    def metric_computation_grid(self):
        e, k = self.inverse_aspect_ratio, self.elongation
        x = np.linspace(1 - e, 1 + e, 100)
        y = np.linspace(-1.1*e*k, e*k, 101)
        return x, y
