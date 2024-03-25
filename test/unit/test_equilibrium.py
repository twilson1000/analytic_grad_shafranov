#!/usr/bin/python3

# Standard imports
import logging
import pytest
import numpy as np
from typing import Callable

# Local imports
from analytic_grad_shafranov import AnalyticGradShafranovSolution, ExtremalPoint

logger = logging.getLogger(__name__)

def finite_difference_first_derivative(func: Callable, x: float, h: float=1e-4):
    f_plus = func(x + h)
    f_minus = func(x - h)
    return (f_plus - f_minus) / h / 2

def finite_difference_second_derivative(func: Callable, x: float, h: float=1e-4):
    f = func(x)
    f_plus = func(x + h)
    f_minus = func(x - h)
    return (f_plus + f_minus - 2 * f) / h**2

class TestPolynomials:
    test_data = ((0.5, -0.5), (1, 0), (1.5, 1.0),)

    @pytest.mark.parametrize('x, y', test_data)
    def test_first_x_derivatives(self, x, y):
        dp_dx_finite_difference = finite_difference_first_derivative(lambda x: np.array(AnalyticGradShafranovSolution.psi_homogenous(x, y)), x)
        assert np.allclose(dp_dx_finite_difference, AnalyticGradShafranovSolution.psi_homogenous_dx(x, y))

    @pytest.mark.parametrize('x, y', test_data)
    def test_first_y_derivatives(self, x, y):
        dp_dy_finite_difference = finite_difference_first_derivative(lambda y: np.array(AnalyticGradShafranovSolution.psi_homogenous(x, y)), y)
        assert np.allclose(dp_dy_finite_difference, AnalyticGradShafranovSolution.psi_homogenous_dy(x, y))

    @pytest.mark.parametrize('x, y', test_data)
    def test_second_x_derivative(self, x, y):
        d2p_dx2_finite_difference = finite_difference_second_derivative(lambda x: np.array(AnalyticGradShafranovSolution.psi_homogenous(x, y)), x)
        assert np.allclose(d2p_dx2_finite_difference, AnalyticGradShafranovSolution.psi_homogenous_dx2(x, y))

    @pytest.mark.parametrize('x, y', test_data)
    def test_second_y_derivative(self, x, y):
        d2p_dy2_finite_difference = finite_difference_second_derivative(lambda y: np.array(AnalyticGradShafranovSolution.psi_homogenous(x, y)), y)
        assert np.allclose(d2p_dy2_finite_difference, AnalyticGradShafranovSolution.psi_homogenous_dy2(x, y))

    @pytest.fixture
    def plasma(self):
        return AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, False), ExtremalPoint(1.0, 0.0, False), 1.0, 1.0)

    @pytest.mark.parametrize('x, y', test_data)
    def test_particular_first_x_derivative(self, plasma, x, y):
        dp_dx_finite_difference = finite_difference_first_derivative(lambda x: plasma.psi_particular(x, y), x)
        actual_value = plasma.psi_particular_dx(x, y)
        assert np.allclose(dp_dx_finite_difference, actual_value)
    
    @pytest.mark.parametrize('x, y', test_data)
    def test_particular_second_x_derivative(self, plasma, x, y):
        d2p_dx2_finite_difference = finite_difference_second_derivative(lambda x: plasma.psi_particular(x, y), x)
        actual_value = plasma.psi_particular_dx2(x, y)
        assert np.allclose(d2p_dx2_finite_difference, actual_value)

def assert_psi_bar_eq_0(plasma, x, y):
    psi_bar = plasma.psi_bar(x, y)
    assert np.isclose(psi_bar, 0)

def assert_psi_bar_dx_eq_0(plasma, x, y):
    psi_bar_dx = plasma.psi_bar_dx(x, y)
    assert np.isclose(psi_bar_dx, 0)

def assert_psi_bar_dy_eq_0(plasma, x, y):
    psi_bar_dy = plasma.psi_bar_dy(x, y)
    assert np.isclose(psi_bar_dy, 0)

class TestBoundaryConstraints:
    symmetric_limiter = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, False), ExtremalPoint(1.0, 0.0, False), 1.0, 1.0)
    asymmetric_limiter = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, False), ExtremalPoint(1.0, 0.1, False), 1.0, 1.0)
    symmetric_upper_null = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, True), ExtremalPoint(1.0, 0.0, False), 1.0, 1.0)
    symmetric_lower_null = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, False), ExtremalPoint(1.0, 0.0, True), 1.0, 1.0)
    asymmetric_upper_null = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, True), ExtremalPoint(1.0, 0.1, False), 1.0, 1.0)
    asymmetric_lower_null = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, False), ExtremalPoint(1.0, 0.1, True), 1.0, 1.0)
    symmetric_double_null = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, True), ExtremalPoint(1.0, 0.1, True), 1.0, 1.0)
    asymmetric_double_null = AnalyticGradShafranovSolution(1.0, 0.1, 0.1, ExtremalPoint(1.0, 0.0, True), ExtremalPoint(1.0, 0.1, True), 1.0, 1.0)

    all_plasmas = (
        symmetric_limiter,
        asymmetric_limiter,
        symmetric_upper_null,
        symmetric_lower_null,
        asymmetric_upper_null,
        asymmetric_lower_null,
    )

    @pytest.fixture
    def plasma(self, request) -> AnalyticGradShafranovSolution:
        return AnalyticGradShafranovSolution(*request.param)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_outer_eq_psi_bar_eq_0(self, plasma):
        ''' Test normalised psi = 0 at outer equatorial point. '''
        assert_psi_bar_eq_0(plasma, *plasma.equatorial_point_outer_xy)
    
    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_inner_eq_psi_bar_eq_0(self, plasma):
        ''' Test normalised psi = 0 at inner equatorial point. '''
        assert_psi_bar_eq_0(plasma, *plasma.equatorial_point_inner_xy)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_outer_eq_psi_bar_dy_eq_0(self, plasma):
        ''' Test y derivative of normalised psi = 0 at outer equatorial point. '''
        assert_psi_bar_dy_eq_0(plasma, *plasma.equatorial_point_outer_xy)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_inner_eq_psi_bar_dy_eq_0(self, plasma):
        ''' Test y derivative normalised psi = 0 at inner equatorial point. '''
        assert_psi_bar_dy_eq_0(plasma, *plasma.equatorial_point_inner_xy)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_outer_eq_curvature(self, plasma):
        ''' Test curvature at outer equatorial point matches d shaped model. '''
        k_mid = 0.5 * (plasma.upper_elongation + plasma.lower_elongation)
        d_mid = 0.5 * (plasma.upper_triangularity + plasma.lower_triangularity)
        alpha_mid = np.arcsin(d_mid)
        N1_mid = - (1 + alpha_mid)**2 / plasma.inverse_aspect_ratio / k_mid**2

        psi_bar_dx = plasma.psi_bar_dx(*plasma.equatorial_point_outer_xy)
        psi_bar_dy2 = plasma.psi_bar_dy2(*plasma.equatorial_point_outer_xy)

        actual_value = N1_mid * psi_bar_dx + psi_bar_dy2
        assert np.isclose(actual_value, 0)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_inner_eq_curvature(self, plasma):
        ''' Test curvature at inner equatorial point matches d shaped model. '''
        k_mid = 0.5 * (plasma.upper_elongation + plasma.lower_elongation)
        d_mid = 0.5 * (plasma.upper_triangularity + plasma.lower_triangularity)
        alpha_mid = np.arcsin(d_mid)
        N2_mid = (1 - alpha_mid)**2 / plasma.inverse_aspect_ratio / k_mid**2

        psi_bar_dx = plasma.psi_bar_dx(*plasma.equatorial_point_inner_xy)
        psi_bar_dy2 = plasma.psi_bar_dy2(*plasma.equatorial_point_inner_xy)

        actual_value = N2_mid * psi_bar_dx + psi_bar_dy2
        assert np.isclose(actual_value, 0)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_upper_point_psi_bar_eq_0(self, plasma):
        ''' Test normalised psi = 0 at upper point. '''
        assert_psi_bar_eq_0(plasma, *plasma.upper_point_xy)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_lower_point_psi_bar_eq_0(self, plasma):
        ''' Test normalised psi = 0 at lower point. '''
        assert_psi_bar_eq_0(plasma, *plasma.lower_point_xy)
    
    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_upper_point_psi_bar_dx_eq_0(self, plasma):
        ''' Test x derivative of normalised psi = 0 at upper point. '''
        assert_psi_bar_dx_eq_0(plasma, *plasma.upper_point_xy)

    @pytest.mark.parametrize('plasma', all_plasmas)
    def test_lower_point_psi_bar_dx_eq_0(self, plasma):
        ''' Test x derivative of normalised psi = 0 at lower point. '''
        assert_psi_bar_dx_eq_0(plasma, *plasma.lower_point_xy)
    
    @pytest.mark.parametrize('plasma', (
        symmetric_upper_null, asymmetric_upper_null, symmetric_double_null, asymmetric_double_null
    ))
    def test_upper_point_psi_bar_dy_eq_0(self, plasma):
        ''' Test y derivative of normalised psi = 0 at upper point. Only for X points. '''
        assert_psi_bar_dy_eq_0(plasma, *plasma.upper_point_xy)

    @pytest.mark.parametrize('plasma', (
        symmetric_lower_null, asymmetric_lower_null, symmetric_double_null, asymmetric_double_null
    ))
    def test_lower_point_psi_bar_dy_eq_0(self, plasma):
        ''' Test y derivative of normalised psi = 0 at lower point. Only for X points. '''
        assert_psi_bar_dy_eq_0(plasma, *plasma.lower_point_xy)

    @pytest.mark.parametrize('plasma', (
        symmetric_limiter, asymmetric_limiter, symmetric_lower_null, asymmetric_lower_null
    ))
    def test_upper_point_curvature(self, plasma):
        ''' Test curvature at upper point matches d shaped model. Not for X points. '''
        k, d = plasma.upper_elongation, plasma.upper_triangularity
        N3 = -k / plasma.inverse_aspect_ratio / (1 - d**2)

        psi_bar_dx2 = plasma.psi_bar_dx(*plasma.equatorial_point_inner_xy)
        psi_bar_dy = plasma.psi_bar_dy2(*plasma.equatorial_point_inner_xy)

        actual_value = N3 * psi_bar_dx2 + psi_bar_dy
        assert np.isclose(actual_value, 0)

    @pytest.mark.parametrize('plasma', (
        symmetric_limiter, asymmetric_limiter, symmetric_lower_null, asymmetric_lower_null
    ))
    def test_lower_point_curvature(self, plasma):
        ''' Test curvature at lower point matches d shaped model. Not for X points. '''
        k, d = plasma.lower_elongation, plasma.lower_triangularity
        N3 = -k / plasma.inverse_aspect_ratio / (1 - d**2)

        psi_bar_dx2 = plasma.psi_bar_dx(*plasma.equatorial_point_inner_xy)
        psi_bar_dy = plasma.psi_bar_dy2(*plasma.equatorial_point_inner_xy)

        actual_value = -N3 * psi_bar_dx2 + psi_bar_dy
        assert np.isclose(actual_value, 0)
    
# e = 0.1
# k = 1.0
# d = 0.2
# alpha = np.arcsin(d)

# N1 = - (1 + alpha)**2 / e / k**2
# N2 = (1 - alpha)**2 / e / k**2
# N3 = -k / e / (1 - d**2)

# @pytest.fixture
# def limiter() -> Limiter:
#     plasma = Limiter(1, 0, e, k, d, 1, 1)
#     return plasma

# @pytest.fixture
# def double_null() -> DoubleNull:
#     plasma = DoubleNull(1, 0, e, k, d, 1, 1)
#     return plasma

# @pytest.fixture
# def single_null() -> SingleNull:
#     plasma = SingleNull(1, 0, e, k, d, 1, 1)
#     return plasma

# eq_inner = (1 - e, 0)
# eq_outer = (1 + e, 0)
# high_point = (1 - d*e, k*e)
# lower_x_point = (1 - 1.1*d*e, 1.1*k*e)
# upper_x_point = (1 - 1.1*d*e, -1.1*k*e)

# @pytest.mark.parametrize(
#     'x, y',
#     (eq_inner, eq_outer, high_point)
# )
# def test_limiter_psi_eq_0(limiter, x, y):
#     psi = limiter.psi_bar(x, y)
#     assert np.isclose(psi, 0)

# def test_limiter_high_point_maximum(limiter):
#     dpsi_dx = limiter.psi_bar_dx(*high_point)
#     assert np.isclose(dpsi_dx, 0)

# def test_limiter_outer_eq_curvature(limiter):
#     value = limiter.psi_bar_dy2(*eq_inner) + N1 * limiter.psi_bar_dx(*eq_inner)
#     assert np.isclose(value, 0)

# def test_limiter_inner_eq_curvature(limiter):
#     value = limiter.psi_bar_dy2(*eq_inner) + N2 * limiter.psi_bar_dx(*eq_inner)
#     assert np.isclose(value, 0)

# def test_limiter_high_point_curvature(limiter):
#     value = limiter.psi_bar_dx2(*high_point) + N3 * limiter.psi_bar_dy(*high_point)
#     assert np.isclose(value, 0)

# @pytest.mark.parametrize(
#     'x, y',
#     (eq_inner, eq_outer, lower_x_point, upper_x_point)
# )
# def test_double_null_psi_eq_0(double_null, x, y):
#     psi = double_null.psi_bar(x, y)
#     assert np.isclose(psi, 0)

# def test_double_null_outer_eq_curvature(double_null):
#     value = double_null.psi_bar_dy2(*eq_inner) + N1 * double_null.psi_bar_dx(*eq_inner)
#     assert np.isclose(value, 0)

# def test_double_null_inner_eq_curvature(double_null):
#     value = double_null.psi_bar_dy2(*eq_inner) + N2 * double_null.psi_bar_dx(*eq_inner)
#     assert np.isclose(value, 0)

# @pytest.mark.parametrize(
#     'x, y',
#     (upper_x_point, lower_x_point)
# )
# def test_double_null_x_point_poloidal_field_eq_0(double_null, x, y):
#     bpol = (
#         double_null.psi_bar_dx(x, y),
#         double_null.psi_bar_dy(x, y)
#     )
#     assert np.allclose(bpol, 0)

# @pytest.mark.parametrize(
#     'x, y',
#     (eq_inner, eq_outer, high_point, lower_x_point)
# )
# def test_single_null_psi_eq_0(single_null, x, y):
#     # Single null has higher error.
#     psi = single_null.psi_bar(x, y)
#     assert np.isclose(psi, 0, atol=1e-3)

# @pytest.mark.parametrize(
#     'x, y',
#     (eq_inner, eq_outer)
# )
# def test_single_null_eq_up_down_symmetry(single_null, x, y):
#     dpsi_dy = single_null.psi_bar_dy(x, y)
#     assert np.isclose(dpsi_dy, 0)

# def test_single_null_high_point_maximum(single_null):
#     dpsi_dx = single_null.psi_bar_dx(*high_point)
#     assert np.isclose(dpsi_dx, 0)

# def test_single_null_x_point_poloidal_field_eq_0(single_null):
#     bpol = (
#         single_null.psi_bar_dx(*lower_x_point),
#         single_null.psi_bar_dy(*lower_x_point)
#     )
#     assert np.allclose(bpol, 0)

# def test_single_null_outer_eq_curvature(single_null):
#     value = single_null.psi_bar_dy2(*eq_inner) + N1 * single_null.psi_bar_dx(*eq_inner)
#     assert np.isclose(value, 0)

# def test_single_null_inner_eq_curvature(single_null):
#     value = single_null.psi_bar_dy2(*eq_inner) + N2 * single_null.psi_bar_dx(*eq_inner)
#     assert np.isclose(value, 0)

# def test_single_null_high_point_curvature(single_null):
#     value = single_null.psi_bar_dx2(*high_point) + N3 * single_null.psi_bar_dy(*high_point)
#     assert np.isclose(value, 0)

# @pytest.mark.parametrize(
#     'plasma_current_anticlockwise, expected_signs',
#     (
#         (True, (-1, 1)),
#         (False, (1, -1))
#     )
# )
# def test_poloidal_field_direction(plasma_current_anticlockwise, expected_signs):
#     plasma = Limiter(1, 0, e, k, d, 1, 1, plasma_current_anticlockwise=plasma_current_anticlockwise)
    
#     Bz_sign = np.sign([
#         plasma.magnetic_field(plasma.magnetic_axis[0] * 0.99, 0)[2],
#         plasma.magnetic_field(plasma.magnetic_axis[0] * 1.01, 0)[2],
#     ])

#     assert np.allclose(Bz_sign, expected_signs)

# @pytest.mark.parametrize(
#     'toroidal_field_anticlockwise, expected_sign',
#     (
#         (True, 1),
#         (False, -1)
#     )
# )
# def test_toroidal_field_direction(toroidal_field_anticlockwise, expected_sign):
#     plasma = Limiter(1, 0, e, k, d, 1, 1, toroidal_field_anticlockwise=toroidal_field_anticlockwise)
#     Btor = plasma.magnetic_field(*plasma.magnetic_axis)[1]
#     assert np.sign(Btor) == expected_sign
