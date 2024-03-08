#!/usr/bin/python3

# Standard imports
import logging
import pytest
import numpy as np
from typing import Callable

# Local imports
from analytic_grad_shafranov import Limiter, SingleNull

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
    dp_dx_finite_difference = finite_difference_first_derivative(lambda x: np.array(SingleNull.homogeneous_polynomials(x, y)), x)
    
    assert np.allclose(dp_dx_finite_difference, SingleNull.homogeneous_polynomials_dx(x, y))

@pytest.mark.parametrize(
    'x, y',
    (
        (0.5, -0.5),
        (1, 0),
        (1.5, 1.0),
    )
)
def test_polynomial_first_derivatives_in_y(x, y):
    dp_dx_finite_difference = finite_difference_first_derivative(lambda y: np.array(SingleNull.homogeneous_polynomials(x, y)), y)
    
    assert np.allclose(dp_dx_finite_difference, SingleNull.homogeneous_polynomials_dy(x, y))

@pytest.mark.parametrize(
    'x, y',
    (
        (0.5, -0.5),
        (1, 0),
        (1.5, 1.0),
    )
)
def test_polynomial_second_derivatives_in_x(x, y):
    dp_dx_finite_difference = finite_difference_second_derivative(lambda x: np.array(SingleNull.homogeneous_polynomials(x, y)), x)
    
    assert np.allclose(dp_dx_finite_difference, SingleNull.homogeneous_polynomials_dx2(x, y))

@pytest.mark.parametrize(
    'x, y',
    (
        (0.5, -0.5),
        (1, 0),
        (1.5, 1.0),
    )
)
def test_polynomial_second_derivatives_in_y(x, y):
    dp_dx_finite_difference = finite_difference_second_derivative(lambda y: np.array(SingleNull.homogeneous_polynomials(x, y)), y)
    
    assert np.isclose(dp_dx_finite_difference, SingleNull.homogeneous_polynomials_dy2(x, y))
