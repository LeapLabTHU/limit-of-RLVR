from .context import assert_equal
import pytest
from sympy import asinh, Symbol

def test_arcsinh():
    assert_equal("\\operatorname{arcsinh}\\left(1\\right)", asinh(1, evaluate=False))
