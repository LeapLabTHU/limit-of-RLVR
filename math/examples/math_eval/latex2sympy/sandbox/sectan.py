from sympy import *
import sys
sys.path.append("..")

x = Symbol('x', real=True)
y = Symbol('y', real=True)

# BUG: 1 + tan^2(x+1) should be == sec^2(x+1) but isnt
lhs = (1 + (tan(x + 1))**2)
rhs = (sec(x + 1))**2
eq = lhs - rhs
print(simplify(lhs))
print(simplify(rhs))
print(simplify(eq))
print(simplify(lhs) == simplify(rhs))

# 1 + tan^2(x) == sec^2(x) but isnt
lhs = (1 + (tan(x))**2)
rhs = (sec(x))**2
eq = lhs - rhs
print(simplify(lhs))
print(simplify(rhs))
print(simplify(eq))
print(simplify(lhs) == simplify(rhs))
