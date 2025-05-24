from latex2sympy import process_sympy
import sys
sys.path.append("..")

latex = "\\frac{a^{2} \\left(3 \\pi - 4 \\sin{\\left(\\pi \\right)} + \\frac{\\sin{\\left(2 \\pi \\right)}}{2}\\right)}{2}"
math = process_sympy(latex)

print(type(math))
print("latex: %s to math: %s" % (latex, math))
