#!/usr/bin/python3

# Standard imports
import logging
import numpy as np
import numpy.typing as npt
import scipy.constants as const
from scipy.interpolate import interp1d
from skimage import measure
from typing import List, Tuple, Union

# Local imports

logger = logging.getLogger(__name__)

class ExtremalPoint:
    __slots__ = ("_elongation", "_triangularity", "_x_point", "_squareness")
    x_point: bool = False

    def __init__(self, elongation: float, triangularity: float, x_point: bool, squareness: float=0.0):
        self._elongation = float(elongation)
        self._triangularity = float(triangularity)
        self._x_point = (x_point == True)
        self._squareness = float(squareness)

        if self.elongation <= 0:
            raise ValueError(f"Elongation must be positive: {self.elongation}")
        if abs(self.triangularity) > 1:
            raise ValueError(f"Triangularity must be between -1 and 1: {self.triangularity}")
        if abs(self.squareness) > 0.5:
            raise ValueError(f"squareness must be between -0.5 and 0.5: {self.squareness}")

    @classmethod
    def at_coordinates(cls, r: float, z: float, x_point: bool, inverse_aspect_ratio: float, major_radius_m: float):
        elongation = abs(z) / major_radius_m / inverse_aspect_ratio
        triangularity = (major_radius_m - r) / major_radius_m / inverse_aspect_ratio

        if x_point:
            elongation /= 1.1
            triangularity /= 1.1

        return cls(elongation, triangularity, x_point)

    @property
    def elongation(self) -> float:
        return self._elongation
    @property
    def triangularity(self) -> float:
        return self._triangularity
    @property
    def x_point(self) -> float:
        return self._x_point
    @property
    def squareness(self) -> float:
        return self._squareness

class AnalyticGradShafranovSolution:
    '''
    Analytic solutions to the Grad Shafranov equation as described in 'Antoine J. Cerfon, Jeffrey P. Freidberg;
    “One size fits all” analytic solutions to the Grad–Shafranov equation. Phys. Plasmas 1 March 2010; 17 (3): 032502.'.

    Some parameters have been flipped sign compared to the paper, see the README of this project for a full description.
    '''
    __slots__ = (
        "major_radius_m", "pressure_parameter", "coefficients", "inverse_aspect_ratio",
        "reference_magnetic_field_T", "plasma_current_MA", "psi_0",
        "normalised_circumference", "normalised_volume", "beta_poloidal", "beta_toroidal",
        "beta_total", "beta_normalised", "kink_safety_factor", "psi_axis", "magnetic_axis",
        "plasma_current_anticlockwise", "toroidal_field_anticlockwise",

        "upper_point", "lower_point", "upper_point_xy", "lower_point_xy", "equatorial_point_inner_xy",
        "equatorial_point_outer_xy", "boundary_radius", "boundary_height", "q_profile", "poloidal_to_toroidal_flux",
    )

    def __init__(
        self,
        major_radius_m: float,
        pressure_parameter: float,
        inverse_aspect_ratio: float,
        upper_point: ExtremalPoint,
        lower_point: ExtremalPoint,
        reference_magnetic_field_T: float,
        plasma_current_MA: float,
        kink_safety_factor: float = None,
        plasma_current_anticlockwise: bool = True,
        toroidal_field_anticlockwise: bool = True,
        use_d_shaped_model: bool = False,
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
            Plasma elongation kappa []. Can be a tuple with the upper and lower triangularity.
        triangularity: float
            Plasma triangularity delta []. Can be a tuple with the upper and lower elongation.
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

        if not isinstance(upper_point, ExtremalPoint):
            raise ValueError("upper_point must be ExtremalPoint")
        if not isinstance(lower_point, ExtremalPoint):
            raise ValueError("lower_point must be ExtremalPoint")
        
        self.upper_point = upper_point
        self.lower_point = lower_point
        
        self.reference_magnetic_field_T: float = float(reference_magnetic_field_T)
        self.plasma_current_anticlockwise: bool = plasma_current_anticlockwise
        self.toroidal_field_anticlockwise: bool = toroidal_field_anticlockwise

        # Solve for the weighting coefficients for each of the polynomials.
        self.calculate_coefficients()
        
        # Calculate magnetic axis location.
        self.calculate_magnetic_axis()

        # Calculate (R, Z) of boundary contour.
        self.calcuate_boundary_contour()

        # Calculate the normalised circumference and volume.
        self.calculate_geometry_factors(use_d_shaped_model=use_d_shaped_model)

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
        self.calculate_q_profile()

        # Add interpolator to convert poloidal flux to toroidal flux.
        self.add_poloidal_toroidal_convertor()
    
    @property
    def upper_elongation(self) -> float:
        return self.upper_point.elongation
    @property
    def lower_elongation(self) -> float:
        return self.lower_point.elongation
    @property
    def upper_triangularity(self) -> float:
        return self.upper_point.triangularity
    @property
    def lower_triangularity(self) -> float:
        return self.lower_point.triangularity
    @property
    def upper_squareness(self) -> float:
        return self.upper_point.squareness
    @property
    def lower_squareness(self) -> float:
        return self.lower_point.squareness

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
        p8 = y
        p9 = y*x2
        p10 = y * (y2 - 3*x2*ln_x)
        p11 = y * (3*x4 - 4*y2*x2)
        p12 = y * ((8*y2 - 80*x2*ln_x)*y2 + (60*ln_x - 45)*x4)
        
        return p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12
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
        dp8_dx = 0
        dp9_dx = 2*x*y
        dp10_dx = -3*x*y*(2*ln_x + 1)
        dp11_dx = x*y*(12*x2 - 8*y2)
        dp12_dx = 40*x*y*((6*x2 - 4*y2)*ln_x - 3*x2 - 2*y2)

        return dp1_dx, dp2_dx, dp3_dx, dp4_dx, dp5_dx, dp6_dx, dp7_dx, dp8_dx, dp9_dx, dp10_dx, dp11_dx, dp12_dx
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
        dp8_dy = 1
        dp9_dy = x2
        dp10_dy = 3*(y2 - x2*ln_x)
        dp11_dy = x2*(3*x2 - 12*y2)
        dp12_dy = 40*y4 + 15*x2*((-16*y2 + 4*x2)*ln_x - 3*x2)

        return dp1_dy, dp2_dy, dp3_dy, dp4_dy, dp5_dy, dp6_dy, dp7_dy, dp8_dy, dp9_dy, dp10_dy, dp11_dy, dp12_dy
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
        d2p8_dx2 = 0
        d2p9_dx2 = 2*y
        d2p10_dx2 = -3*y*(2*ln_x + 3)
        d2p11_dx2 = y*(36*x2 - 8*y2)
        d2p12_dx2 = y*((720*x2 - 160*y2)*ln_x - 120*x2-240*y2)

        return d2p1_dx2, d2p2_dx2, d2p3_dx2, d2p4_dx2, d2p5_dx2, d2p6_dx2, d2p7_dx2, d2p8_dx2, d2p9_dx2, d2p10_dx2, d2p11_dx2, d2p12_dx2
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
        d2p8_dy2 = 0
        d2p9_dy2 = 0
        d2p10_dy2 = 6*y
        d2p11_dy2 = -24*x2*y
        d2p12_dy2 = y*(160*y2- 480*x2*ln_x)

        return d2p1_dy2, d2p2_dy2, d2p3_dy2, d2p4_dy2, d2p5_dy2, d2p6_dy2, d2p7_dy2, d2p8_dy2, d2p9_dy2, d2p10_dy2, d2p11_dy2, d2p12_dy2
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
        ''' First derivative of poloidal flux function normalised to psi0 with respect to x. '''
        psi_dx = self.psi_particular_dx(x, y)

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous_dx(x, y)):
            psi_dx += c * p

        return psi_dx
    def psi_bar_dy(self, x: float, y: float) -> float:
        ''' First derivative of poloidal flux function normalised to psi0 with respect to y. '''
        psi_dy = 0

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous_dy(x, y)):
            psi_dy += c * p

        return psi_dy
    def psi_bar_dx2(self, x: float, y: float) -> float:
        ''' Second derivative of poloidal flux function normalised to psi0 with respect to x. '''
        psi_dx2 = self.psi_particular_dx2(x, y)

        # Add weighted sum of the homogeneous solutions.
        for c, p in zip(self.coefficients, self.psi_homogenous_dx2(x, y)):
            psi_dx2 += c * p

        return psi_dx2
    def psi_bar_dy2(self, x: float, y: float) -> float:
        ''' Second derivative of poloidal flux function normalised to psi0 with respect to y. '''
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
    
    def psi_norm(self, R: float, Z: float) -> float:
        R0 = self.major_radius_m
        return self.psi_bar_to_psi_norm(self.psi_bar(R / R0, Z / R0))
    def magnetic_field(self, R: float, Z: float) -> Tuple[float, float, float]:
        ''' (R, phi, Z) components of the magnetic field [T]. '''
        psi_norm = self.psi_norm(R, Z)
        # NOTE: Need a test for this!
        B_R = self.psi_dZ(R, Z) / R
        B_Z = -self.psi_dR(R, Z) / R
        B_toroidal = self.f_function(psi_norm) / R

        return B_R, B_toroidal, B_Z
    
    def calculate_coefficients(self):
        '''
        Solve for the weighting coefficients of the polynomials defining psi. We fit to a d shaped contour with the
        required geometry factors at 3 points:
            Inner equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            Outer equatorial point: point of minimum R at midplane (Z=0) on the boundary contour.
            High point: point of maximum Z on the boundary contour.
            Upper X point: point of maximum Z on the boundary contour.
        '''
        R0, e = self.major_radius_m, self.inverse_aspect_ratio

        # Some coefficients from D shaped model. Use average elongation, triangularity and squareness at midplane.
        
        k_mid = 0.5 * (self.upper_elongation + self.lower_elongation)
        d_mid = 0.5 * (self.upper_triangularity + self.lower_triangularity)
        s_mid = 0.5 * (self.upper_squareness + self.lower_squareness)
        alpha_mid = np.arcsin(d_mid)
        N1_mid = - (1 + alpha_mid)**2 / e / k_mid**2 / (1 + 2*s_mid**2)
        N2_mid = (1 - alpha_mid)**2 / e / k_mid**2 / (1 + 2*s_mid**2)

        # Points to fit D shaped model at.
        self.equatorial_point_inner_xy = (1 - e, 0)
        self.equatorial_point_outer_xy = (1 + e, 0)
        
        # We solve the system y = Mx to find the coefficient vector x.
        M = np.zeros((12, 12))
        y = np.zeros(12)

        # Outer equatorial point (psi = 0).
        M[0] = self.psi_homogenous(*self.equatorial_point_outer_xy)
        y[0] = -self.psi_particular(*self.equatorial_point_outer_xy)

        # Inner equatorial point (psi = 0).
        M[1] = self.psi_homogenous(*self.equatorial_point_inner_xy)
        y[1] = -self.psi_particular(*self.equatorial_point_inner_xy)

        # Outer equatorial point up down symmetry (d(psi)/dy = 0).
        M[2] = self.psi_homogenous_dy(*self.equatorial_point_outer_xy)

        # Inner equatorial point up down symmetry (d(psi)/dy = 0).
        M[3] = self.psi_homogenous_dy(*self.equatorial_point_inner_xy)

        # Outer equatorial point curvature (d^2(psi)/dy^2 + N1 * d(psi)/dx = 0).
        M[4] = np.array(self.psi_homogenous_dy2(*self.equatorial_point_outer_xy)) + N1_mid * np.array(self.psi_homogenous_dx(*self.equatorial_point_outer_xy))
        y[4] = -N1_mid * self.psi_particular_dx(*self.equatorial_point_outer_xy)

        # Inner equatorial point curvature (d^2(psi)/dy^2 + N2 * d(psi)/dx = 0).
        M[5] = np.array(self.psi_homogenous_dy2(*self.equatorial_point_inner_xy)) + N2_mid * np.array(self.psi_homogenous_dx(*self.equatorial_point_inner_xy))
        y[5] = -N2_mid * self.psi_particular_dx(*self.equatorial_point_inner_xy)

        if self.upper_point.x_point:
            k, d = self.upper_elongation, self.upper_triangularity
            self.upper_point_xy = (1 - 1.1*d*e, 1.1*k*e)

            # Upper X point (psi = 0).
            M[6] = self.psi_homogenous(*self.upper_point_xy)
            y[6] = -self.psi_particular(*self.upper_point_xy)

            # B poloidal = 0 at upper X point (d(psi)/dx = 0).
            M[7] = self.psi_homogenous_dx(*self.upper_point_xy)
            y[7] = -self.psi_particular_dx(*self.upper_point_xy)

            # B poloidal = 0 at upper X point (d(psi)/dy = 0).
            M[8] = self.psi_homogenous_dy(*self.upper_point_xy)
        else:
            # Upper high point.
            k, d, s = self.upper_elongation, self.upper_triangularity, self.upper_squareness
            N3 = -k * (1 - 2*s**2) / e / (1 - d**2)
            self.upper_point_xy = (1 - d*e, k*e)

            # Upper high point (psi = 0).
            M[6] = self.psi_homogenous(*self.upper_point_xy)
            y[6] = -self.psi_particular(*self.upper_point_xy)

            # Upper high point maximum (d(psi)/dx = 0).
            M[7] = self.psi_homogenous_dx(*self.upper_point_xy)
            y[7] = -self.psi_particular_dx(*self.upper_point_xy)

            # Upper high point curvature (d^2(psi)/dx^2 + N3 * d(psi)/dy = 0).
            M[8] = np.array(self.psi_homogenous_dx2(*self.upper_point_xy)) + N3 * np.array(self.psi_homogenous_dy(*self.upper_point_xy))
            y[8] = -self.psi_particular_dx2(*self.upper_point_xy)

        if self.lower_point.x_point:
            k, d = self.lower_elongation, self.lower_triangularity
            self.lower_point_xy = (1 - 1.1*d*e, -1.1*k*e)

            # Lower X point (psi = 0).
            M[9] = self.psi_homogenous(*self.lower_point_xy)
            y[9] = -self.psi_particular(*self.lower_point_xy)

            # B poloidal = 0 at lower X point (d(psi)/dx = 0).
            M[10] = self.psi_homogenous_dx(*self.lower_point_xy)
            y[10] = -self.psi_particular_dx(*self.lower_point_xy)

            # B poloidal = 0 at lower X point (d(psi)/dy = 0).
            M[11] = self.psi_homogenous_dy(*self.lower_point_xy)
        else:
            # Lower high point.
            k, d, s = self.lower_elongation, self.lower_triangularity, self.lower_squareness
            N3 = k * (1 - 2*s**2) / e / (1 - d**2)
            self.lower_point_xy = (1 - d*e, -k*e)

            # Lower high point (psi = 0).
            M[9] = self.psi_homogenous(*self.lower_point_xy)
            y[9] = -self.psi_particular(*self.lower_point_xy)

            # Lower high point maximum (d(psi)/dx = 0).
            M[10] = self.psi_homogenous_dx(*self.lower_point_xy)
            y[10] = -self.psi_particular_dx(*self.lower_point_xy)

            # Lower high point curvature (d^2(psi)/dx^2 + N3 * d(psi)/dy = 0).
            M[11] = np.array(self.psi_homogenous_dx2(*self.lower_point_xy)) + N3 * np.array(self.psi_homogenous_dy(*self.lower_point_xy))
            y[11] = -self.psi_particular_dx2(*self.lower_point_xy)

        self.coefficients = np.linalg.solve(M, y)
    def calculate_magnetic_axis(self, tolerance: float=1.0e-4, max_iterations: int=100):
        # Find value of psi on magnetic axis. Know it lies near (x, y) = (1, 0) so use Netwon's method to find where d(psi)/dx = 0 and d(psi)/dy = 0.
        magnetic_axis = np.array([1.0, 0.0])
        correction = np.zeros(2)

        for i in range(max_iterations):
            psi_dx = self.psi_bar_dx(*magnetic_axis)
            psi_dy = self.psi_bar_dy(*magnetic_axis)
            psi_dx2 = self.psi_bar_dx2(*magnetic_axis)
            psi_dy2 = self.psi_bar_dy2(*magnetic_axis)

            correction[:] = (psi_dx / psi_dx2), (psi_dy / psi_dy2)
            magnetic_axis -= correction

            if np.linalg.norm(correction) < tolerance:
                logger.info(f"Found magnetic axis in {i+1} iterations.")
                self.magnetic_axis = magnetic_axis * self.major_radius_m
                break
        
        if i == max_iterations - 1:
            logger.error(f"Failed to find magnetic axis within {max_iterations} iterations.")
    @property
    def shafranov_shift(self) -> float:
        return self.magnetic_axis[0] - self.major_radius_m

    def d_shape_boundary(self, theta: float) -> float:
        ''' D shaped boundary contour for the prescribed geometry factors. '''
        theta = np.array(theta)
        x, y = np.zeros_like(theta), np.zeros_like(theta)
        mask = np.logical_and(theta >= 0, theta <= np.pi)

        e = self.inverse_aspect_ratio

        def d_shape(theta, k, alpha, s):
            x = 1 + e * np.cos(theta + alpha*np.sin(theta))
            y = e * k * np.sin(theta + s*np.sin(2*theta))
            return x, y

        # Above midplane.
        k, d, s = self.upper_elongation, self.upper_triangularity, self.upper_squareness
        alpha = np.arcsin(d)
        x[mask], y[mask] = d_shape(theta[mask], k, alpha, s)

        # Below midplane.
        k, d, s = self.lower_elongation, self.lower_triangularity, self.lower_squareness
        alpha = np.arcsin(d)
        x[~mask], y[~mask] = d_shape(theta[~mask], k, alpha, s)

        return x, y
    def d_shape_boundary_derivatives(self, theta: float) -> float:
        theta = np.array(theta)
        xprime, yprime = np.zeros_like(theta), np.zeros_like(theta)
        mask = np.logical_and(theta >= 0, theta <= np.pi)

        e = self.inverse_aspect_ratio

        def d_shape_prime(theta, k, alpha, s):
            xprime = -e * np.sin(theta + alpha*np.sin(theta)) * (1 + alpha*np.cos(theta))
            yprime = e * k * np.cos(theta + s*np.sin(2*theta)) * (1 + 2*s*np.cos(2*theta))
            return xprime, yprime

        # Above midplane.
        k, d, s = self.upper_elongation, self.upper_triangularity, self.upper_squareness
        alpha = np.arcsin(d)
        xprime[mask], yprime[mask] = d_shape_prime(theta[mask], k, alpha, s)

        # Below midplane.
        k, d, s = self.lower_elongation, self.lower_triangularity, self.lower_squareness
        alpha = np.arcsin(d)
        xprime[~mask], yprime[~mask] = d_shape_prime(theta[~mask], k, alpha, s)

        return xprime, yprime
    def calcuate_boundary_contour(self, n_points: int=101, psi_norm_threshold: float=1.0e-3, max_iterations: int=100):
        ''' Calculate boundary contour of psi map using Newton's method '''
        # Calculate the extremal value of psi so we can calculate psi norm.
        psi_bar_0 = self.psi_bar(*self.magnetic_axis / self.major_radius_m)
        
        # Distort d shaped value such that psi=0.
        theta = np.linspace(0, 2*np.pi, n_points)
        x_d_shape, y_d_shape = self.d_shape_boundary(theta)
        x_boundary, y_boundary = np.copy(x_d_shape), np.copy(y_d_shape)

        for i, t in enumerate(theta):
            for j in range(max_iterations):
                psi_bar = self.psi_bar(x_boundary[i], y_boundary[i])
                psi_norm = psi_bar / psi_bar_0

                # If psi norm is close enough to zero, break.
                if abs(psi_norm) < psi_norm_threshold:
                    break
                
                # Use Netwon's method to update boundary position.
                dpsi_dx = self.psi_bar_dx(x_boundary[i], y_boundary[i])
                dpsi_dy = self.psi_bar_dy(x_boundary[i], y_boundary[i])
                
                cos_t, sin_t = np.cos(t), np.sin(t)
                dpsi_dv = cos_t * dpsi_dx + sin_t * dpsi_dy

                x_boundary[i] -= cos_t * psi_bar / dpsi_dv
                y_boundary[i] -= sin_t * psi_bar / dpsi_dv
        
            if j == max_iterations - 1:
                raise ValueError("Too many iterations to calculate boundary contour.")

        self.boundary_radius = x_boundary * self.major_radius_m
        self.boundary_height = y_boundary * self.major_radius_m
    def calculate_geometry_factors(self, use_d_shaped_model: bool=True):
        '''
        Calculate the normalised circumference and volume of the plasma. Can either calculate based on the estimated
        boundary contour from the D shaped model or from the fitted poloidal flux function.
        '''
        if use_d_shaped_model:
            theta = np.linspace(0, 2*np.pi, 101)
            x, y = self.d_shape_boundary(theta)
            xprime, yprime = self.d_shape_boundary_derivatives(theta)
            rprime = (xprime**2 + yprime**2)**0.5

            self.normalised_circumference = np.trapz(rprime, theta)
            self.normalised_volume = -np.trapz(x*xprime*y, theta)
        else:
            circumference = 0
            volume = 0

            for dR, dZ in zip(np.diff(self.boundary_radius), np.diff(self.boundary_height)):
                circumference += (dR**2 + dZ**2)**0.5
            
            for i in range(len(self.boundary_radius) - 1):
                R1, Z1 = self.boundary_radius[i], self.boundary_height[i]
                R2, Z2 = self.boundary_radius[i + 1], self.boundary_height[i + 1]
                volume += 0.5 * (R1*Z1 + R2*Z2) * (R2 - R1)
            
            self.normalised_circumference = circumference / self.major_radius_m
            self.normalised_volume = -volume / self.major_radius_m**3
    def metric_computation_grid(self) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        '''
        Grid of normalised radius x = R / R0 and height y = Z / R0 used to calculate numerical integrals
        for calculating the plasma current and plasma beta.
        '''
        e, k_up, k_down = self.inverse_aspect_ratio, self.upper_elongation, self.lower_elongation
        x = np.linspace(1 - e, 1 + e, 100)
        y = np.linspace(-e*k_down, e*k_up, 101)
        return x, y
    def calculate_metrics(self):
        '''
        Calculate the poloidal flux coordinate normalisation (psi_0) and the plasma 'figures of merit':
        beta_poloidal, beta_toroidal, beta_total and beta_normalised.
        '''
        e = self.inverse_aspect_ratio
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
        Ip_integrand = (A / x_grid) - (1 + A) * x_grid
        Ip_integrand[mask] = 0
        Ip_integral = np.sum(Ip_integrand) * dxdy

        # This is proportional to the average plasma pressure.
        psix_integrand = psi_bar * x_grid
        psix_integrand[mask] = 0
        psix_integral = np.sum(psix_integrand) * dxdy
        
        # Calculate beta of plasma.
        self.beta_poloidal = (2 * Cp**2 * (1 + A) / V) * psix_integral / Ip_integral**2
        self.beta_toroidal = self.beta_poloidal * e**2 / qstar**2
        self.beta_total = self.beta_poloidal * e**2 / (e**2 + qstar**2)
        self.beta_normalised = (e * R0 * B0 / self.plasma_current_MA) * self.beta_total

        # Calculate value of psi at magnetic axis for given plasma current.
        self.psi_0 = self.plasma_current_MA * 1e6 * const.mu_0 * self.major_radius_m / Ip_integral

        # If the plasma current is clockwise flip the sign of the poloidal flux.
        if not self.plasma_current_anticlockwise:
            self.psi_0 *= -1

        # Evaluate value of psi_norm at the magnetic axis.
        self.psi_axis = self.psi(*self.magnetic_axis)
    def calculate_q_profile(self, mesh_size: int=30):
        ''' Calculate q profile. '''
        R0 = self.major_radius_m

        # The q profile q(psi) = 1/(2*pi) * int{F(psi) / (R |grad{psi}|)} dl_p where
        # l_p is the poloidal distance along the surface of constant psi.
        q_profile = np.zeros(mesh_size)

        # Calculate (x, y) locations of contours.
        xmesh, ymesh = self.metric_computation_grid()
        dxmesh, dymesh = xmesh[-1] - xmesh[0], ymesh[-1] - ymesh[0]
        psi_bar_axis = self.psi_bar(*self.magnetic_axis / R0)
        psi_bar_norm_grid = self.psi_bar(*np.meshgrid(xmesh, ymesh, indexing='ij')) / psi_bar_axis

        def integrand(x, y):
            # Return 1 / (R |grad{psi}|). Technically we use use
            # 1 / (x |grad_x,y{psi}|) but the factor of major radius cancels.
            dpsi_dx = self.psi_0 * self.psi_bar_dx(x, y)
            dpsi_dy = self.psi_0 * self.psi_bar_dy(x, y)
            mod_grad_psi = (dpsi_dx**2 + dpsi_dy**2)**0.5
            return 1 / (mod_grad_psi * x)
        
        def calculate_arclength(x, y):
            # Calculate arclength around contour.
            lp = np.zeros_like(x)
            for k, (dx, dy) in enumerate(zip(np.diff(x), np.diff(y))):
                lp[k + 1] = lp[k] + (dx**2 + dy**2)**0.5
            
            return lp
        
        # NOTE: This is psi_bar normalised, so it is 1 at the magnetic axis
        # and zero at the boundary. We will use the computed boundary contour
        # to do psi_bar_norm = 0. Also skip psi_bar_norm = 1 as there is just a single point (bad numerics).
        psi_norm_mesh = np.linspace(1, 0, mesh_size + 1)[1:-1]

        for i, psi_norm in enumerate(psi_norm_mesh):
            # Use this instead of matplotlib.contour as the latter forces figure creation.
            contour = measure.find_contours(psi_bar_norm_grid, level=psi_norm)

            if len(contour) == 0:
                raise ValueError(f"Unable to find contour for psi norm = {psi_norm}")

            x = xmesh[0] + dxmesh * (contour[0][:, 0] / psi_bar_norm_grid.shape[0])
            y = ymesh[0] + dymesh * (contour[0][:, 1] / psi_bar_norm_grid.shape[1])

            # F function is a flux function so we can move it out the integral (F / R is toroidal field).
            F = self.f_function(psi_norm)
            
            # Integrate using trapezium rule.
            lp = R0 * calculate_arclength(x, y) # Poloidal arclength [m].
            q_profile[i] = F * np.trapz(integrand(x, y), lp)
        
        # Use pre-computed boundary contour for separatrix. As there is a
        # saddle point the matplotlib contours will sometimes follow the contours
        # towards the divertor instead of following the high field side boundary.
        x_bdy, y_bdy = self.boundary_radius / R0, self.boundary_height / R0

        # Integrate using trapezium rule.
        F = self.f_function(1)
        lp = R0 * calculate_arclength(x_bdy, y_bdy) # Poloidal arclength [m].
        q_profile[-1] = F * np.trapz(integrand(x_bdy, y_bdy), lp)
        
        # Scale q profile by 2*pi to match definition of poloidal flux function psi.
        q_profile /= 2 * np.pi

        # Set q profile.
        self.q_profile = q_profile
    def add_poloidal_toroidal_convertor(self):
        ''' Define function to convert from poloidal flux to toroidal flux. '''
        poloidal_flux = np.linspace(self.psi_axis, 0, len(self.q_profile))

        # Toroidal flux is int{q dpsi_poloidal}
        toroidal_flux = np.zeros_like(poloidal_flux)

        for i in range(len(toroidal_flux) - 1):
            dpsi_tor = 0.5 * (self.q_profile[i] + self.q_profile[i + 1]) * (poloidal_flux[i + 1] - poloidal_flux[i])
            toroidal_flux[i + 1] = toroidal_flux[i] + dpsi_tor

        self.poloidal_to_toroidal_flux = interp1d(poloidal_flux, toroidal_flux, bounds_error=False, fill_value=(toroidal_flux[0], toroidal_flux[-1]))
    def psi_bar_to_psi_norm(self, psi_bar: float) -> float:
        ''' Convert psi_bar parameter used in the normalised Grad Shafranov equation to the standard normalised poloidal flux co-ordinate. '''
        return 1 - (psi_bar * self.psi_0 / self.psi_axis)
    def psi_norm_to_psi_bar(self, psi_norm: float) -> float:
        ''' Convert normalised poloidal flux co-ordinate to the psi_bar parameter used in the normalised Grad Shafranov equation. '''
        return (1 - psi_norm) * self.psi_axis / self.psi_0
    def psi_toroidal(self, R, Z):
        ''' Toroidal flux function. '''
        psi_poloidal = self.psi(R, Z)
        return self.poloidal_to_toroidal_flux(psi_poloidal)
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
        f = R0 * (B0**2 - (2*psi0**2 / R0**4) * A * psi_bar)**0.5

        # If toroidal field is clockwise flip the sign of f.
        if not self.toroidal_field_anticlockwise:
            f *= -1

        return f
    def toroidal_current_density_kA_per_m2(self, R, Z):
        ''' Toroidal current density [kA m^-2]. '''
        R0, psi0, A = self.major_radius_m, self.psi_0, self.pressure_parameter
        x = R / R0
        return 1e-3 * psi0 * (A / x - (1 + A) * x**2) / (const.mu_0 * R0**3)
    def plotting_xy_arrays(self, padding=1.05, n_points: int=100):
        ''' Grid of (x, y) points that encloses the entire plasma boundary plus some padding '''
        e = self.inverse_aspect_ratio
        xmin, xmax = (1 - padding*e), (1 + padding*e)
        ymin, ymax = padding * self.lower_point_xy[1], padding * self.upper_point_xy[1]

        return np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)
    def plotting_rz_arrays(self, **kwargs):
        ''' Arrays of (R, Z) points that encloses the entire plasma boundary plus some padding. '''
        x, y = self.plotting_xy_arrays(**kwargs)
        return x * self.major_radius_m, y * self.major_radius_m
    def save_as_eqdsk(self, filename: str, rz_shape: Tuple[int, int]=None, flux_grid_shape: int=50):
        """
        Meant to be 48/52 characters then 3 4 character integers nxeqd, nyeqd, nveqd

        GENRAY read routine says 52 characters then 3 integers
        Geqdsk specification document says 48 characters then 3 integers

        nveqd is NOT part of the standard eqdsk format! GENRAY and CQL3D can handle it, so can pyEquilibrium. Need to check JETTO and LUKE.


        lengths of boundary and limiter contours are meant to be provided as 5 character integers

        nxeqd and nyeqk are x, y shapes of psi array
        nveqd is the number of flux surfaces on which p, f, p', f' and q are tabulated.
        GENRAY and CQL3D support this?
        """
        # Default shape tries to have equal grid point spacing in R and Z with a 50 point radial mesh.
        if rz_shape is None:
            rz_shape = (50, int(np.floor(25 * (self.upper_elongation + self.lower_elongation))))
        else:
            if len(rz_shape) != 2:
                raise ValueError("rz_shape must be 2 integers")
            
            rz_shape = tuple([int(x) for x in rz_shape])

            for x in rz_shape:
                if x <= 0:
                    raise ValueError(f"rz_shape dimensions must be positive: {rz_shape}")

        # R, Z grid.
        x_plot, y_plot = self.plotting_xy_arrays()
        r = np.linspace(x_plot[0], x_plot[-1], rz_shape[0]) * self.major_radius_m
        z = np.linspace(y_plot[0], y_plot[-1], rz_shape[1]) * self.major_radius_m

        rmin, rmax = r[0], r[-1]
        zmin, zmax = z[0], z[-1]

        # Flux grid used to define f, p, ffprime, pprime and q.
        flux_grid_shape = int(flux_grid_shape)

        # Data.
        _HEADER_VARIABLES = (
            'rdim', 'zdim', 'rcentr', 'rleft', 'zmid', 'rmaxis', 'zmaxis', 'simag',
            'sibry', 'bcentr', 'current', 'simag', 'xdum', 'rmaxis', 'xdum', 'zmaxis',
            'xdum', 'sibry', 'xdum', 'xdum'
        )
        _ARRAY_VARIABLES = ('fpol', 'pres', 'ffprime', 'pprime', 'psirz', 'qpsi')

        data_0d = {
            'nw': rz_shape[0], # Number of radial points.
            'nh': rz_shape[1], # Number of height points.
            'nflux': flux_grid_shape, # Dummy value that is not used.
            'rdim': rmax - rmin, # Range of radial mesh [m].
            'zdim': zmax - zmin, # Range of height mesh [m].
            'rcentr': self.major_radius_m, # Major radius [m].
            'rleft': rmin, # Minimum value of radial mesh.
            'zmid': 0.5 * (zmin + zmax), # Middle value of height mesh [m].
            'rmaxis': self.magnetic_axis[0], # Radius of magnetic axis [m].
            'zmaxis': self.magnetic_axis[1], # Height of magnetic axis [m].
            'simag': self.psi_axis, # Psi value at magnetic axis [Wb].
            'sibry': 0, # Psi value at separatrix / last closed flux surface [Wb].
            'bcentr': self.reference_magnetic_field_T, # Vacuum toroidal magnetic field at major radius [T].
            'current': self.plasma_current_MA * 1.0e6, # Total plasma current [A].
            'xdum': 0, # Dummy value that is not used.
            'nbbbs': len(self.boundary_radius), # Number of points in boundary contour.
            'limitr': 0, # Number of points in limiter contour. We don't provide this so set to 0.
        }

        # Psi array is same shape as radial mesh.
        psi_norm = np.linspace(0, 1, flux_grid_shape)
        ffprime_const = -self.pressure_parameter * self.psi_0**2 / self.major_radius_m**2
        pprime_const = (1 + self.pressure_parameter) * self.psi_0**2 / self.major_radius_m**4 / const.mu_0
        
        # Interpolate safety factor onto desired values.
        qpsi = np.interp(psi_norm, np.linspace(0, 1, len(self.q_profile) + 1)[1:], self.q_profile)

        data_1d = {
            'fpol': self.f_function(psi_norm),
            'pres': 1.0e3 * self.pressure_kPa(psi_norm),
            'ffprime': ffprime_const * np.ones_like(psi_norm),
            'pprime': pprime_const * np.ones_like(psi_norm),
            'psirz': self.psi(*np.meshgrid(r, z, indexing='ij')),
            'qpsi': qpsi,
        }

        # Format strings for data.
        _ENTRIES_PER_LINE = 5
        _VALUE_FORMAT = "{:16.9e}"
        _INTEGER_FORMAT_1 = "{:4n}"
        _INTEGER_FORMAT_2 = "{:5n}"
        _COMMENT_FORMAT = "{:>52}"

        def format_lines(data: npt.NDArray[float]) -> List[str]:
            str_values = [_VALUE_FORMAT.format(x) for x in data]
            folded_values = [str_values[i:i + _ENTRIES_PER_LINE] for i in range(0, len(str_values), _ENTRIES_PER_LINE)]
            lines = [''.join(value_set) for value_set in folded_values]
            return lines

        # List of lines to write to text file.
        file_lines = []

        # First line containing comment and array sizes.
        comment = _COMMENT_FORMAT.format("Analytic Grad Shafranov Solution")
        sizes = sizes = "".join([_INTEGER_FORMAT_1.format(data_0d[variable_name]) for variable_name in ('nw', 'nh', 'nflux')])
        file_lines.append(comment + sizes)

        # 0D variables.
        file_lines.extend(
            format_lines([data_0d[name] for name in _HEADER_VARIABLES])
        )

        # 1D variables.
        for name in _ARRAY_VARIABLES:
            logger.info(f"Writing {name} ({data_1d[name].shape})")
            file_lines.extend(
                format_lines(data_1d[name].flatten("F")) # Flatten in Fortran style order so 2D arrays are read correctly.
            )

        # Size of boundary and limiter contours.
        boundary_size = _INTEGER_FORMAT_2.format(data_0d['nbbbs'])
        limiter_size = _INTEGER_FORMAT_2.format(data_0d['limitr'])
        file_lines.append(boundary_size + limiter_size)

        # Boundary contour data. There is no limiter contour data.
        boundary_data = np.vstack((self.boundary_radius, self.boundary_height)).T.flatten()
        file_lines.extend(format_lines(boundary_data))

        logger.info(f"Writing EQDSK to {filename}")
        with open(filename, "w") as writefile:
            writefile.write("\n".join(file_lines))
    
class Limiter(AnalyticGradShafranovSolution):
    __slots__ = ()

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
        squareness: float = 0.0,
        plasma_current_anticlockwise: bool = True,
        toroidal_field_anticlockwise: bool = True,
        use_d_shaped_model: bool = False,
    ):
        upper_point = ExtremalPoint(elongation, triangularity, False, squareness=squareness)
        lower_point = upper_point
        super().__init__(
            major_radius_m,
            pressure_parameter,
            inverse_aspect_ratio,
            upper_point,
            lower_point,
            reference_magnetic_field_T,
            plasma_current_MA,
            kink_safety_factor,
            plasma_current_anticlockwise,
            toroidal_field_anticlockwise,
            use_d_shaped_model
        )

class DoubleNull(AnalyticGradShafranovSolution):
    __slots__ = ()

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
        squareness: float = 0.0,
        plasma_current_anticlockwise: bool = True,
        toroidal_field_anticlockwise: bool = True,
        use_d_shaped_model: bool = False,
    ):
        upper_point = ExtremalPoint(elongation, triangularity, True, squareness=squareness)
        lower_point = upper_point

        super().__init__(
            major_radius_m,
            pressure_parameter,
            inverse_aspect_ratio,
            upper_point,
            lower_point,
            reference_magnetic_field_T,
            plasma_current_MA,
            kink_safety_factor,
            plasma_current_anticlockwise,
            toroidal_field_anticlockwise,
            use_d_shaped_model
        )

class SingleNull(AnalyticGradShafranovSolution):
    __slots__ = ()

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
        squareness: float = 0.0,
        plasma_current_anticlockwise: bool = True,
        toroidal_field_anticlockwise: bool = True,
        use_d_shaped_model: bool = False,
        lower_x: bool = True,
    ):
        if lower_x:
            upper_point = ExtremalPoint(elongation, triangularity, False, squareness=squareness)
            lower_point = ExtremalPoint(elongation, triangularity, True, squareness=squareness)
        else:
            upper_point = ExtremalPoint(elongation, triangularity, True, squareness=squareness)
            lower_point = ExtremalPoint(elongation, triangularity, False, squareness=squareness)

        super().__init__(
            major_radius_m,
            pressure_parameter,
            inverse_aspect_ratio,
            upper_point,
            lower_point,
            reference_magnetic_field_T,
            plasma_current_MA,
            kink_safety_factor,
            plasma_current_anticlockwise,
            toroidal_field_anticlockwise,
            use_d_shaped_model
        )
