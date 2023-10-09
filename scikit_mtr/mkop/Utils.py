# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:29:03 2019

@author: Jimena Ferreira - FIng- UdelaR
"""
__all__ = ['tree2symb', 'round_expr', 'Prediction']

import numpy as np
from sympy import Add, Mul, log, cos, sin, exp, Abs, sqrt
from sympy import Lambda
from sympy import symbols, lambdify
from sympy import sympify


def tree2symb(ind, n_vars):
    x = symbols('x', real=True)
    y = symbols('y', real=True)

    locals = {
        "add": Add,
        "mul": Mul,
        "div": Lambda((x, y), x / y),
        "protectedDiv": Lambda((x, y), x / y),
        "sub": Lambda((x, y), x - y),
        "neg": Lambda((x), (-1 * x)),
        "log": Lambda((x), log(x)),
        "protectedLog": Lambda((x), log(x)),
        "cos": Lambda((x), cos(x)),
        "sin": Lambda((x), sin(x)),
        "exp": Lambda((x), exp(x)),
        "protectedExp": Lambda((x), exp(x)),
        "abs": Lambda((x), Abs(x)),
        "protectedSqrt": sqrt,
        "sqrt": sqrt,
        "square": Lambda((x), x ** 2),
        "powerReal": Lambda((x, y), x ** y),
    }

    # Dynamically add variable symbols
    for i in range(n_vars):
        var = symbols(f'X{i}', real=True)
        locals[f'X{i}'] = var
    final = sympify(ind, locals=locals)
    return final


def round_expr(expr, num_digits):
    from sympy import Float, Number
    expr = expr.xreplace({n: Float(n, num_digits) for n in expr.atoms(Number)})

    return expr


def Prediction(expr, x):
    num_vars = x.shape[1]

    # Generate symbols dynamically based on the number of variables
    symbol_list = [symbols(f"X{i}", real=True) for i in range(num_vars)]

    # Create a lambdified function with the dynamically generated symbols
    f = lambdify(symbol_list, expr, "numpy")

    # Evaluate the expression for each data point
    y_est = [f(*x[pos, :]) for pos in range(x.shape[0])]

    y_est = np.asarray(y_est).reshape((-1, 1))
    y_est = np.nan_to_num(y_est)
    return y_est
