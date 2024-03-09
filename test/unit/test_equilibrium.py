#!/usr/bin/python3

# Standard imports
import logging
import pytest
import numpy as np
from typing import Callable

# Local imports
from analytic_grad_shafranov import Limiter, DoubleNull, SingleNull

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

@pytest.mark.parametrize(
    'x, y',
    (
        (0.5, -0.5),
        (1, 0),
        (1.5, 1.0),
    )
)
def test_polynomial_first_derivatives_in_x(x, y):
    dp_dx_finite_difference = finite_difference_first_derivative(lambda x: np.array(SingleNull.psi_homogenous(x, y)), x)
    
    assert np.allclose(dp_dx_finite_difference, SingleNull.psi_homogenous_dx(x, y))

@pytest.mark.parametrize(
    'x, y',
    (
        (0.5, -0.5),
        (1, 0),
        (1.5, 1.0),
    )
)
def test_polynomial_first_derivatives_in_y(x, y):
    dp_dx_finite_difference = finite_difference_first_derivative(lambda y: np.array(SingleNull.psi_homogenous(x, y)), y)
    
    assert np.allclose(dp_dx_finite_difference, SingleNull.psi_homogenous_dy(x, y))

@pytest.mark.parametrize(
    'x, y',
    (
        (0.5, -0.5),
        (1, 0),
        (1.5, 1.0),
    )
)
def test_polynomial_second_derivatives_in_x(x, y):
    dp_dx_finite_difference = finite_difference_second_derivative(lambda x: np.array(SingleNull.psi_homogenous(x, y)), x)
    
    assert np.allclose(dp_dx_finite_difference, SingleNull.psi_homogenous_dx2(x, y))

@pytest.mark.parametrize(
    'x, y',
    (
        (0.5, -0.5),
        (1, 0),
        (1.5, 1.0),
    )
)
def test_polynomial_second_derivatives_in_y(x, y):
    dp_dx_finite_difference = finite_difference_second_derivative(lambda y: np.array(SingleNull.psi_homogenous(x, y)), y)
    
    assert np.allclose(dp_dx_finite_difference, SingleNull.psi_homogenous_dy2(x, y))

e = 0.1
k = 1.0
d = 0.2
alpha = np.arcsin(d)

N1 = - (1 + alpha)**2 / e / k**2
N2 = (1 - alpha)**2 / e / k**2
N3 = -k / e / (1 - d**2)

@pytest.fixture
def limiter() -> Limiter:
    plasma = Limiter(1, 0, e, k, d, 1, 1)
    return plasma

@pytest.fixture
def double_null() -> DoubleNull:
    plasma = DoubleNull(1, 0, e, k, d, 1, 1)
    return plasma

@pytest.fixture
def single_null() -> SingleNull:
    plasma = SingleNull(1, 0, e, k, d, 1, 1)
    return plasma

eq_inner = (1 - e, 0)
eq_outer = (1 + e, 0)
high_point = (1 - d*e, k*e)
lower_x_point = (1 - 1.1*d*e, 1.1*k*e)
upper_x_point = (1 - 1.1*d*e, -1.1*k*e)

@pytest.mark.parametrize(
    'x, y',
    (eq_inner, eq_outer, high_point)
)
def test_limiter_psi_eq_0(limiter, x, y):
    psi = limiter.psi_bar(x, y)
    assert np.isclose(psi, 0)

def test_limiter_high_point_maximum(limiter):
    dpsi_dx = limiter.psi_bar_dx(*high_point)
    assert np.isclose(dpsi_dx, 0)

def test_limiter_outer_eq_curvature(limiter):
    value = limiter.psi_bar_dy2(*eq_inner) + N1 * limiter.psi_bar_dx(*eq_inner)
    assert np.isclose(value, 0)

def test_limiter_inner_eq_curvature(limiter):
    value = limiter.psi_bar_dy2(*eq_inner) + N2 * limiter.psi_bar_dx(*eq_inner)
    assert np.isclose(value, 0)

def test_limiter_high_point_curvature(limiter):
    value = limiter.psi_bar_dx2(*high_point) + N3 * limiter.psi_bar_dy(*high_point)
    assert np.isclose(value, 0)

@pytest.mark.parametrize(
    'x, y',
    (eq_inner, eq_outer, lower_x_point, upper_x_point)
)
def test_double_null_psi_eq_0(double_null, x, y):
    psi = double_null.psi_bar(x, y)
    assert np.isclose(psi, 0)

def test_double_null_outer_eq_curvature(double_null):
    value = double_null.psi_bar_dy2(*eq_inner) + N1 * double_null.psi_bar_dx(*eq_inner)
    assert np.isclose(value, 0)

def test_double_null_inner_eq_curvature(double_null):
    value = double_null.psi_bar_dy2(*eq_inner) + N2 * double_null.psi_bar_dx(*eq_inner)
    assert np.isclose(value, 0)

@pytest.mark.parametrize(
    'x, y',
    (upper_x_point, lower_x_point)
)
def test_double_null_x_point_poloidal_field_eq_0(double_null, x, y):
    bpol = (
        double_null.psi_bar_dx(x, y),
        double_null.psi_bar_dy(x, y)
    )
    assert np.allclose(bpol, 0)

@pytest.mark.parametrize(
    'x, y',
    (eq_inner, eq_outer, high_point, lower_x_point)
)
def test_single_null_psi_eq_0(single_null, x, y):
    # Single null has higher error.
    psi = single_null.psi_bar(x, y)
    assert np.isclose(psi, 0, atol=1e-3)

@pytest.mark.parametrize(
    'x, y',
    (eq_inner, eq_outer)
)
def test_single_null_eq_up_down_symmetry(single_null, x, y):
    dpsi_dy = single_null.psi_bar_dy(x, y)
    assert np.isclose(dpsi_dy, 0)

def test_single_null_high_point_maximum(single_null):
    dpsi_dx = single_null.psi_bar_dx(*high_point)
    assert np.isclose(dpsi_dx, 0)

def test_single_null_x_point_poloidal_field_eq_0(single_null):
    bpol = (
        single_null.psi_bar_dx(*lower_x_point),
        single_null.psi_bar_dy(*lower_x_point)
    )
    assert np.allclose(bpol, 0)

def test_single_null_outer_eq_curvature(single_null):
    value = single_null.psi_bar_dy2(*eq_inner) + N1 * single_null.psi_bar_dx(*eq_inner)
    assert np.isclose(value, 0)

def test_single_null_inner_eq_curvature(single_null):
    value = single_null.psi_bar_dy2(*eq_inner) + N2 * single_null.psi_bar_dx(*eq_inner)
    assert np.isclose(value, 0)

def test_single_null_high_point_curvature(single_null):
    value = single_null.psi_bar_dx2(*high_point) + N3 * single_null.psi_bar_dy(*high_point)
    assert np.isclose(value, 0)

@pytest.mark.parametrize(
    'plasma_current_anticlockwise, expected_signs',
    (
        (True, (-1, 1)),
        (False, (1, -1))
    )
)
def test_poloidal_field_direction(plasma_current_anticlockwise, expected_signs):
    plasma = Limiter(1, 0, e, k, d, 1, 1, plasma_current_anticlockwise=plasma_current_anticlockwise)
    
    Bz_sign = np.sign([
        plasma.magnetic_field(plasma.magnetic_axis[0] * 0.99, 0)[2],
        plasma.magnetic_field(plasma.magnetic_axis[0] * 1.01, 0)[2],
    ])

    assert np.allclose(Bz_sign, expected_signs)

@pytest.mark.parametrize(
    'toroidal_field_anticlockwise, expected_sign',
    (
        (True, 1),
        (False, -1)
    )
)
def test_toroidal_field_direction(toroidal_field_anticlockwise, expected_sign):
    plasma = Limiter(1, 0, e, k, d, 1, 1, toroidal_field_anticlockwise=toroidal_field_anticlockwise)
    Btor = plasma.magnetic_field(*plasma.magnetic_axis)[1]
    assert np.sign(Btor) == expected_sign
