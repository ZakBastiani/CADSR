import numpy as np
import math
from sympy import symbols, simplify
from scipy import optimize
from enums import *


def NMSE_reward_func(pred_y, y, y_std, v=0, var_count=0, node_count=0, bic_scaler=1.0):
    NRMSE = np.sqrt(np.mean((pred_y - y) ** 2)) / y_std
    return 1 / (1 + NRMSE)


def NMSE_reg_reward_func(pred_y, y, y_std, v, var_count, node_count, lambdavar=0.1):
    NMSE = np.mean((pred_y - y) ** 2) / y_std
    return 1 / (1 + NMSE) + lambdavar * np.exp(-node_count/32)


def SPL_reg_reward_func(pred_y, y, y_std, v, var_count, node_count, nu=0.99):
    RMSE = np.sqrt(np.mean((pred_y - y) ** 2))
    return (nu**var_count) / (1 + RMSE)


def calc_r_squared(pred_y, y, normalizer, v=0, var_count=0, node_count=0, bic_scaler=1.0):
    return 1 - np.sum((y - pred_y) ** 2) / normalizer


def BIC_np_calc_loss(pred_y, y, normalizer, v, var_count, node_count, bic_scaler=1.0):
    sample_size = len(y)
    return bic_scaler * (var_count + node_count) * math.log(sample_size) + sample_size * math.log(v)


def simplify_equation(equation, c_count, x_count=10):
    symbols(f'c:{c_count}')
    for k in range(c_count):
        equation = equation.replace(f"c[{k}]", f"c{k}")
    symbols('x:10')
    for j in range(x_count):
        equation = equation.replace(f"x[{j}]", f"x{j}")
    equation = equation.replace("torch.tensor(1, device=device)", "1").replace("torch.", "").replace("np.", "")
    equation = str(simplify(equation))
    for k in range(c_count):
        equation = equation.replace(f"c{k}", f"c[{k}]")
    for j in range(x_count):
        equation = equation.replace(f"x{j}", f"x[{j}]")
    equation = equation.replace("sin", "torch.sin").replace("log", "torch.log").replace("cos", "torch.cos")
    return equation


# Define optimization functions at the top level for pickling
def ls_func(c, x, y, equation):
    return y - eval(equation)


def min_func(c, x, y, equation):
    return np.sum((y - eval(equation))**2)


# Initializer for multiprocessing pool to set global data
def init_pool(x_full_, y_full_):
    global x_full_global, y_full_global
    x_full_global = x_full_
    y_full_global = y_full_


# Worker function to process a single equation
def process_equation(task, max_dataset_size, reward_func, opt_lm, std, bic_scaler, reward_function_value):
    np.seterr(all='ignore')
    i, equation, constants, incremental_constant, prod_count, node_count = task
    x_full = x_full_global
    y_full = y_full_global

    data_set_size = len(y_full)
    if data_set_size > max_dataset_size:
        perm = np.random.permutation(data_set_size)[:max_dataset_size]
        x = x_full.T[perm].T
        y = y_full[perm]
    else:
        x = x_full
        y = y_full

    try:
        if incremental_constant == 0:
            x = x_full
            pred_y = eval(equation)
            v = np.mean((pred_y - y_full) ** 2)
            reward_val = reward_func(pred_y, y_full, std, v, incremental_constant if reward_function_value != RewardFunctions.SPLReward.value else prod_count, node_count, bic_scaler)
            return i, constants, reward_val, v, None

        if opt_lm:
            result = optimize.least_squares(ls_func, constants, args=(x, y, equation), method='lm')
        else:
            result = optimize.minimize(min_func, constants, args=(x, y, equation), method='L-BFGS-B')

        c = result.x
        x = x_full
        pred_y = eval(equation)
        v = np.mean((pred_y - y_full) ** 2)
        reward_val = reward_func(pred_y, y_full, std, v, incremental_constant if reward_function_value != RewardFunctions.SPLReward.value else prod_count, node_count, bic_scaler)
        return i, c, reward_val, v, None

    except Exception as e:
        error_msg = f"Equation {equation} failed: {str(e)}"
        return i, constants, np.nan, np.nan, error_msg
