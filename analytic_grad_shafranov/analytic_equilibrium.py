#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

# Local imports

logger = logging.getLogger(__name__)

class AnalyticGradShafranovSolution:
    ''' Cerfon and Freidberg 2010 '''
    __slots__ = (
        "major_radius_m", "pressure_parameter", "coefficients", "inverse_aspect_ratio", "elongation",
        "triangularity", "reference_magnetic_field_T", "plasma_current_MA", "psi_0",
        "normalised_circumference", "normalised_volume", "poloidal_beta", "toroidal_beta",
        "total_beta", "normalised_beta", "kink_safety_factor"
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
        kink_safety_factor: float = None
    ):
        self.major_radius_m: float = float(major_radius_m)
        self.pressure_parameter: float = float(pressure_parameter)
        self.inverse_aspect_ratio: float = float(inverse_aspect_ratio)
        self.elongation: float = float(elongation)
        self.triangularity: float = float(triangularity)
        self.reference_magnetic_field_T: float = float(reference_magnetic_field_T)

        self.calculate_coefficients()
        self.calculate_geometry_factors()

        e, B0 = self.inverse_aspect_ratio, self.reference_magnetic_field_T
        R0, Cp = self.major_radius_m, self.normalised_circumference

        if kink_safety_factor is None:
            self.plasma_current_MA: float = float(plasma_current_MA)
            self.kink_safety_factor = e*B0*R0*Cp / const.mu_0 / (1e6 * self.plasma_current_MA)
        else:
            self.kink_safety_factor = float(kink_safety_factor)
            self.plasma_current_MA = 1e-6 * e*B0*R0*Cp / const.mu_0 / self.kink_safety_factor
        
        # Set initial value of psi axis. This will be set in calculate_metrics() to match
        # the prescribed plasma current.
        self.psi_0 = 1.0
        self.calculate_metrics()
    
    def calculate_coefficients(self):
        raise NotImplementedError()

    @staticmethod
    def homogeneous_polynomials(x, y):
        ''' x = r / Rmaj, y = z / Rmaj '''
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
    def homogeneous_polynomials_dx(x, y):
        ''' x = r / Rmaj, y = z / Rmaj '''
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
    def homogeneous_polynomials_dy(x, y):
        ''' x = r / Rmaj, y = z / Rmaj '''
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
    def homogeneous_polynomials_dx2(x, y):
        ''' x = r / Rmaj, y = z / Rmaj '''
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
    def homogeneous_polynomials_dy2(x, y):
        ''' x = r / Rmaj, y = z / Rmaj '''
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
    
    def _psi0(self, x, y):
        A = self.pressure_parameter
        x2, ln_x = x**2, np.log(x)
        x4 = x2**2
        return 0.5 * A * x2 * ln_x - (x4 / 8) * (1 + A)
    def _psi0_dx(self, x, y):
        A = self.pressure_parameter
        x2, ln_x = x**2, np.log(x)
        return 0.5 * x * (A*(2*ln_x + 1) - x2*(1 + A))
    def _psi0_dx2(self, x, y):
        x2, ln_x = x**2, np.log(x)
        A = self.pressure_parameter
        return A * (1.5 + ln_x) - 1.5 * (1 + A) * x2
    def _psi_bar(self, x, y):
        psi = self._psi0(x, y)

        for c, p in zip(self.coefficients, self.homogeneous_polynomials(x, y)):
            psi += c * p

        return psi
    def psi(self, R, Z):
        return self.psi_0 * self._psi_bar(R / self.major_radius_m, Z / self.major_radius_m)
    
    def d_shape_boundary(self, theta):
        e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity
        alpha = np.arcsin(d)
        x = 1 + e * np.cos(theta + alpha*np.sin(theta))
        y = e * k * np.sin(theta)

        return x, y
    def calculate_geometry_factors(self):
        e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity
        alpha = np.arcsin(d)

        theta = np.linspace(0, np.pi, 101)
        x, y = self.d_shape_boundary(theta)
        xprime = -e * np.sin(theta + (alpha*np.sin(theta))) * (1 + alpha*np.cos(theta))
        yprime = e * k * np.cos(theta)
        rprime = (xprime**2 + yprime**2)**0.5

        self.normalised_circumference = 2 * np.trapz(rprime, theta)
        self.normalised_volume = -2 * np.trapz(x*xprime*y, theta)

        # x, y = self.metric_computation_grid()
        # x_grid, ygrid = np.meshgrid(x, y, indexing='ij')
        # dxdy = (x[1] - x[0]) * (y[1] - y[0])

        # psiN = self._psi_bar(x_grid, ygrid) * x_grid
        # mask = psiN > 0
        # logger.info(np.sum(mask) * dxdy)
    def metric_computation_grid(self):
        e, k = self.inverse_aspect_ratio, self.elongation
        x = np.linspace(1 - e, 1 + e, 100)
        y = np.linspace(-e*k, e*k, 101)
        return x, y
    def calculate_metrics(self):
        e, k = self.inverse_aspect_ratio, self.elongation
        Cp, V = self.normalised_circumference, self.normalised_volume
        R0, B0 = self.major_radius_m, self.reference_magnetic_field_T
        A, qstar = self.pressure_parameter, self.kink_safety_factor

        # Calculate two integrals appearing in expressions for beta and plasma current.
        x, y = self.metric_computation_grid()
        x_grid, ygrid = np.meshgrid(x, y, indexing='ij')
        dxdy = (x[1] - x[0]) * (y[1] - y[0])

        psi_bar = self._psi_bar(x_grid, ygrid)
        mask = psi_bar < 0

        # This is proportional to the total plasma current.
        Ip_integrand = (1 + A) * x_grid - (A / x_grid)
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

    def pressure_kPa(self, psi_norm):
        A = self.pressure_parameter
        return 1e-3 * (self.psi_0**2 / self.major_radius_m**4 / const.mu_0) * (1 + A) * psi_norm 
    def f_function(self, psi_norm):
        R0, B0 = self.major_radius_m, self.reference_magnetic_field_T
        psi0, A = self.psi_0, self.pressure_parameter
        return R0 * B0 * (1 - (2*psi0**2 / R0**4) * A * psi_norm)**0.5

class Limiter(AnalyticGradShafranovSolution):
    def calculate_coefficients(self):
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

        # Outer equatorial point.
        M[0] = self.homogeneous_polynomials(x_eq_outer, y_eq_outer)
        y[0] = -self._psi0(x_eq_outer, y_eq_outer)

        # Inner equatorial point.
        M[1] = self.homogeneous_polynomials(x_eq_inner, y_eq_inner)
        y[1] = -self._psi0(x_eq_inner, y_eq_inner)

        # High point.
        M[2] = self.homogeneous_polynomials(x_high, y_high)
        y[2] = -self._psi0(x_high, y_high)

        # High point maximum.
        M[3] = self.homogeneous_polynomials_dx(x_high, y_high)
        y[3] = -self._psi0_dx(x_high, y_high)

        # Outer equatorial point curvature.
        M[4] = np.array(self.homogeneous_polynomials_dy2(x_eq_outer, y_eq_outer)) + N1 * np.array(self.homogeneous_polynomials_dx(x_eq_outer, y_eq_outer))
        y[4] = -N1 * self._psi0_dx(x_eq_outer, y_eq_outer)

        # Inner equatorial point curvature.
        M[5] = np.array(self.homogeneous_polynomials_dy2(x_eq_inner, y_eq_inner)) + N2 * np.array(self.homogeneous_polynomials_dx(x_eq_inner, y_eq_inner))
        y[5] = -N2 * self._psi0_dx(x_eq_inner, y_eq_inner)

        # High point curvature.
        M[6] = np.array(self.homogeneous_polynomials_dx2(x_high, y_high)) + N3 * np.array(self.homogeneous_polynomials_dy(x_high, y_high))
        y[6] = -self._psi0_dx2(x_high, y_high)

        self.coefficients = np.linalg.solve(M, y)

class DoubleNull(AnalyticGradShafranovSolution):
    def calculate_coefficients(self):
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

        # Outer equatorial point.
        M[0] = self.homogeneous_polynomials(x_eq_outer, y_eq_outer)
        y[0] = -self._psi0(x_eq_outer, y_eq_outer)

        # Inner equatorial point.
        M[1] = self.homogeneous_polynomials(x_eq_inner, y_eq_inner)
        y[1] = -self._psi0(x_eq_inner, y_eq_inner)

        # Outer equatorial point curvature.
        M[2] = np.array(self.homogeneous_polynomials_dy2(x_eq_outer, y_eq_outer)) + N1 * np.array(self.homogeneous_polynomials_dx(x_eq_outer, y_eq_outer))
        y[2] = -N1 * self._psi0_dx(x_eq_outer, y_eq_outer)

        # Inner equatorial point curvature.
        M[3] = np.array(self.homogeneous_polynomials_dy2(x_eq_inner, y_eq_inner)) + N2 * np.array(self.homogeneous_polynomials_dx(x_eq_inner, y_eq_inner))
        y[3] = -N2 * self._psi0_dx(x_eq_inner, y_eq_inner)

        # X point.
        M[4] = self.homogeneous_polynomials(x_sep, y_sep)
        y[4] = -self._psi0(x_sep, y_sep)

        # B poloidal = 0 at X point.
        M[5] = self.homogeneous_polynomials_dx(x_sep, y_sep)
        y[5] = -self._psi0_dx(x_sep, y_sep)

        # B poloidal = 0 at X point.
        M[6] = self.homogeneous_polynomials_dy(x_sep, y_sep)

        self.coefficients = np.linalg.solve(M, y)
    def metric_computation_grid(self):
        e, k = self.inverse_aspect_ratio, self.elongation
        x = np.linspace(1 - e, 1 + e, 100)
        y = np.linspace(-1.1*e*k, 1.1*e*k, 101)
        return x, y

class SingleNull(AnalyticGradShafranovSolution):
    @staticmethod
    def homogeneous_polynomials(x, y):
        ''' x = r / Rmaj, y = z / Rmaj '''
        x2, y2, ln_x = x**2, y**2, np.log(x)
        x4 = x2**2

        p1, p2, p3, p4, p5, p6, p7 = AnalyticGradShafranovSolution.homogeneous_polynomials(x, y)
        p8 = y
        p9 = y*x2
        p10 = y * (y2 - 3*x2*ln_x)
        p11 = y * (3*x4 - 4*y2*x2)
        p12 = y * ((8*y2 - 80*x2*ln_x)*y2 + (60*ln_x - 45)*x4)

        return p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12
    @staticmethod
    def homogeneous_polynomials_dx(x, y):
        x2, y2, ln_x = x**2, y**2, np.log(x)

        dp1_dx, dp2_dx, dp3_dx, dp4_dx, dp5_dx, dp6_dx, dp7_dx = AnalyticGradShafranovSolution.homogeneous_polynomials_dx(x, y)
        dp8_dx = 0
        dp9_dx = 2*x*y
        dp10_dx = -3*x*y*(2*ln_x + 1)
        dp11_dx = x*y*(12*x2 - 8*y2)
        dp12_dx = 40*x*y*((6*x2 - 4*y2)*ln_x - 3*x2 - 2*y2)

        return dp1_dx, dp2_dx, dp3_dx, dp4_dx, dp5_dx, dp6_dx, dp7_dx, dp8_dx, dp9_dx, dp10_dx, dp11_dx, dp12_dx
    @staticmethod
    def homogeneous_polynomials_dx2(x, y):
        x2, y2, ln_x = x**2, y**2, np.log(x)

        d2p1_dx2, d2p2_dx2, d2p3_dx2, d2p4_dx2, d2p5_dx2, d2p6_dx2, d2p7_dx2 = AnalyticGradShafranovSolution.homogeneous_polynomials_dx2(x, y)
        d2p8_dx2 = 0
        d2p9_dx2 = 2*y
        d2p10_dx2 = -3*y*(2*ln_x + 3)
        d2p11_dx2 = y*(36*x2 - 8*y2)
        d2p12_dx2 = y*((720*x2 - 160*y2)*ln_x - 120*x2-240*y2)

        return d2p1_dx2, d2p2_dx2, d2p3_dx2, d2p4_dx2, d2p5_dx2, d2p6_dx2, d2p7_dx2, d2p8_dx2, d2p9_dx2, d2p10_dx2, d2p11_dx2, d2p12_dx2
    @staticmethod
    def homogeneous_polynomials_dy(x, y):
        x2, y2, ln_x = x**2, y**2, np.log(x)
        y4 = y2**2

        dp1_dy, dp2_dy, dp3_dy, dp4_dy, dp5_dy, dp6_dy, dp7_dy = AnalyticGradShafranovSolution.homogeneous_polynomials_dy(x, y)
        dp8_dy = 1
        dp9_dy = x2
        dp10_dy = 3*(y2 - x2*ln_x)
        dp11_dy = x2*(3*x2 - 12*y2)
        dp12_dy = 40*y4 + 15*x2*((-16*y2 + 4*x2)*ln_x - 3*x2)

        return dp1_dy, dp2_dy, dp3_dy, dp4_dy, dp5_dy, dp6_dy, dp7_dy, dp8_dy, dp9_dy, dp10_dy, dp11_dy, dp12_dy
    @staticmethod
    def homogeneous_polynomials_dy2(x, y):
        x2, y2, ln_x = x**2, y**2, np.log(x)

        d2p1_dy2, d2p2_dy2, d2p3_dy2, d2p4_dy2, d2p5_dy2, d2p6_dy2, d2p7_dy2 = AnalyticGradShafranovSolution.homogeneous_polynomials_dy2(x, y)
        d2p8_dy2 = 0
        d2p9_dy2 = 0
        d2p10_dy2 = 6*y
        d2p11_dy2 = -24*x2*y
        d2p12_dy2 = y*(160*y2- 480*x2*ln_x)

        return d2p1_dy2, d2p2_dy2, d2p3_dy2, d2p4_dy2, d2p5_dy2, d2p6_dy2, d2p7_dy2, d2p8_dy2, d2p9_dy2, d2p10_dy2, d2p11_dy2, d2p12_dy2
    def calculate_coefficients(self):
        e, k, d = self.inverse_aspect_ratio, self.elongation, self.triangularity

        # Some coefficients from D shaped model.
        alpha = np.arcsin(d)
        N1 = - (1 + alpha)**2 / e / k**2
        N2 = (1 - alpha)**2 / e / k**2
        N3 = -k / e / (1 - d**2)

        # Points to fit D shaped model at.
        x_eq_inner, y_eq_inner = 1 - e, 0
        x_eq_outer, y_eq_outer = 1 + e, 0
        x_high, y_high = 1 - d*e, -k*e
        x_sep, y_sep = 1 - 1.1*d*e, 1.1*k*e

        # We solve the system y = Mx to find the coefficient vector x.
        M = np.zeros((12, 12))
        y = np.zeros(12)

        # Outer equatorial point.
        M[0] = self.homogeneous_polynomials(x_eq_outer, y_eq_outer)
        y[0] = -self._psi0(x_eq_outer, y_eq_outer)

        # Inner equatorial point.
        M[1] = self.homogeneous_polynomials(x_eq_inner, y_eq_inner)
        y[1] = -self._psi0(x_eq_inner, y_eq_inner)

        # Upper high point.
        M[2] = self.homogeneous_polynomials(x_high, y_high)
        y[2] = -self._psi0(x_high, y_high)

        # Lower X point.
        M[3] = self.homogeneous_polynomials(x_sep, y_sep)
        y[3] = -self._psi0(x_sep, y_sep)

        # Outer equatorial point up down symmetry.
        M[4] = self.homogeneous_polynomials_dy(x_eq_outer, y_eq_outer)

        # Inner equatorial point up down symmetry.
        M[5] = self.homogeneous_polynomials_dy(x_eq_inner, y_eq_inner)

        # Upper high point maximum.
        M[6] = self.homogeneous_polynomials_dx(x_high, y_high)
        y[6] = -self._psi0_dx(x_high, y_high)

        # B poloidal = 0 at lower X point.
        M[7] = self.homogeneous_polynomials_dx(x_sep, y_sep)
        y[7] = -self._psi0_dx(x_sep, y_sep)

        # B poloidal = 0 at lower X point.
        M[8] = self.homogeneous_polynomials_dy(x_sep, y_sep)

        # Outer equatorial point curvature.
        M[9] = np.array(self.homogeneous_polynomials_dy2(x_eq_outer, y_eq_outer)) + N1 * np.array(self.homogeneous_polynomials_dx(x_eq_outer, y_eq_outer))
        y[9] = -N1 * self._psi0_dx(x_eq_outer, y_eq_outer)

        # Inner equatorial point curvature.
        M[10] = np.array(self.homogeneous_polynomials_dy2(x_eq_inner, y_eq_inner)) + N2 * np.array(self.homogeneous_polynomials_dx(x_eq_inner, y_eq_inner))
        y[10] = -N2 * self._psi0_dx(x_eq_inner, y_eq_inner)

        # High point curvature.
        M[11] = np.array(self.homogeneous_polynomials_dx2(x_high, y_high)) + N3 * np.array(self.homogeneous_polynomials_dy(x_high, y_high))
        y[11] = -self._psi0_dx2(x_high, y_high)

        self.coefficients = np.linalg.solve(M, y)
    def metric_computation_grid(self):
        e, k = self.inverse_aspect_ratio, self.elongation
        x = np.linspace(1 - e, 1 + e, 100)
        y = np.linspace(-1.1*e*k, e*k, 101)
        return x, y
