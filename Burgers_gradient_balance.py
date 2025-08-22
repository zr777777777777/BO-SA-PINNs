import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab as pylabplt
from scipy.interpolate import griddata
import time
import math
import scipy.special as sp
import os
import random
import hashlib

os.environ["DDEBACKEND"] = "pytorch"
os.environ["TORCHDYNAMO_DISABLE"] = "1" 
import deepxde as dde
from deepxde.backend import backend_name, tf, torch
from deepxde import utils
from bayes_opt import BayesianOptimization
from scipy.stats import qmc  # LHS
from scipy.optimize import differential_evolution

if backend_name == "tensorflow" or backend_name == "tensorflow.compat.v1":
   be = tf
elif backend_name == "pytorch":
   be = torch

# Default configuration for floating-point numbers
dde.config.set_default_float("float64")
torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#-------------------------------------------------------------------------------#
# Problem definition: Burgers equation
#-------------------------------------------------------------------------------#
def gen_testdata():
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

start = time.time()

#-------------------------------------------------------------------------------#
# TG activation function
#-------------------------------------------------------------------------------#
def wavelet_tanh_gaussian(x):
    return torch.tanh(x) * torch.exp(-x**2 / 2)

#-------------------------------------------------------------------------------#
# Random seed
#-------------------------------------------------------------------------------#
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    dde.config.set_random_seed(seed)

set_global_seed(42)

#-------------------------------------------------------------------------------#
# BO for selecting hyperparameters with loss reduction rate optimization
#-------------------------------------------------------------------------------#
# Global variables to store evaluation history
global_evaluation_history = []
global_best_error = float("inf")
global_best_weights = None

def train_and_evaluate(hparams):
    global global_best_error, global_best_weights, global_evaluation_history
    
    # Record start time
    start_time = time.time()
    
    # loss function weights, initial values are from others' research
    w_pde = hparams.get("w_pde", 0.01)
    w_abc = hparams.get("w_abc", 0.125)
    w_ini = hparams.get("w_ini", 0.01)

    lr = hparams.get("lr", 0.005)

    num_domain_idx = int(round(hparams.get("num_domain_idx",100)))
    num_boundary_idx = int(round(hparams.get("num_boundary_idx",50)))
    num_initial_idx = int(round(hparams.get("num_initial_idx",50)))
    num_domain=num_domain_idx * 50
    num_boundary = num_boundary_idx * 50
    num_initial = num_initial_idx * 50

    num_layers = int(round(hparams.get("num_layers", 3)))
    num_neurons = int(round(hparams.get("num_neurons", 50)))

    loss_weights = [w_pde, w_abc, w_ini]
    
    # Define PDE Data
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=num_domain,    # 内部点数量
        num_boundary=num_boundary,   # 边界点数量
        num_initial=num_initial,    # 初始点数量
        num_test=2500
    )

    layer_sizes = [2] + [num_neurons] * num_layers + [1]
    net = dde.nn.FNN(layer_sizes, wavelet_tanh_gaussian, "Glorot uniform")
    model = dde.Model(data, net)

    # 编译模型
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    
    try:
        # 方案1（推荐）：训练1步来获取step 0的初始损失
        print("Getting initial loss (step 0)...")
        losshistory_initial, train_state_initial = model.train(iterations=1, display_every=1)
        
        # 从训练历史中提取step 0的损失（就是日志中显示的第一行损失）
        initial_loss_pde = losshistory_initial.loss_train[0][0]  # step 0, PDE loss
        initial_loss_abc = losshistory_initial.loss_train[0][1]  # step 0, ABC loss  
        initial_loss_ini = losshistory_initial.loss_train[0][2]  # step 0, INI loss
        initial_total_loss = initial_loss_pde + initial_loss_abc + initial_loss_ini
        
        print(f"Step 0 initial losses - PDE: {initial_loss_pde:.6e}, ABC: {initial_loss_abc:.6e}, INI: {initial_loss_ini:.6e}")
        print(f"Step 0 total loss: {initial_total_loss:.6e}")
        
        # 重新编译模型并继续训练99步（总共100步）
        model.compile("adam", lr=lr, loss_weights=loss_weights)
        print("Continuing training for 99 more steps...")
        losshistory, train_state = model.train(iterations=99, display_every=50)
        
        # 获取最终损失（step 100）
        final_loss_pde = train_state.loss_train[0]
        final_loss_abc = train_state.loss_train[1] 
        final_loss_ini = train_state.loss_train[2]
        final_total_loss = final_loss_pde + final_loss_abc + final_loss_ini
        
        training_successful = True
        
    except Exception as e:
        print(f"Training failed: {e}")
        # 如果训练失败，返回很低的分数
        initial_total_loss = 1.0  # 默认初始损失
        final_total_loss = initial_total_loss * 10  # 假设损失变差了
        training_successful = False

    # Record end time and calculate training duration
    end_time = time.time()
    training_time = end_time - start_time

    # Get current model parameters (weights)
    if training_successful:
        current_weights = list(model.net.parameters())
    else:
        current_weights = None
    
    # 计算损失变化率 - 基于step 0和step 100
    if initial_total_loss > 1e-10 and final_total_loss > 1e-10 and training_successful:
        # loss_reduction_rate = (initial_total_loss - final_total_loss) / initial_total_loss
        loss_reduction_rate = (np.log(initial_total_loss) - np.log(final_total_loss))/ abs(np.log(initial_total_loss))
        
        # 如果损失增加太多，给予惩罚
        if loss_reduction_rate < -0.5:  # 损失增加超过50%
            loss_reduction_rate = -0.5
            
    else:
        loss_reduction_rate = -1.0  # 训练失败的惩罚分数

    # Store evaluation history
    evaluation_record = {
        'hparams': hparams.copy(),
        'loss_reduction_rate': loss_reduction_rate,
        'training_time': training_time,
        'weights': current_weights,
        'initial_total_loss': initial_total_loss,
        'final_total_loss': final_total_loss,
        'training_successful': training_successful
    }
    global_evaluation_history.append(evaluation_record)

    # Update global best weights if the new loss reduction rate is higher
    if loss_reduction_rate > global_best_error and training_successful:
        global_best_error = loss_reduction_rate
        global_best_weights = current_weights

    print(f"Loss Reduction Rate: {loss_reduction_rate:.6f}, Training Time: {training_time:.2f}s")
    print(f"  Initial Loss (Step 0): {initial_total_loss:.6e}")
    print(f"  Final Loss (Step 100): {final_total_loss:.6e}")
    print(f"  Training Successful: {training_successful}")
    
    if initial_total_loss > 1e-10 and final_total_loss > 1e-10:
        print(f"  Log Reduction: {np.log(initial_total_loss) - np.log(final_total_loss):.4f}")
    
    return loss_reduction_rate

# 修改贝叶斯优化的目标函数
def objective_for_bayes(w_pde, w_abc, w_ini, lr, num_layers, num_neurons, num_domain_idx, num_boundary_idx, num_initial_idx):
    # step:0.0001
    w_pde = round(w_pde, 4)
    w_abc = round(w_abc, 4)
    w_ini = round(w_ini, 4)

    # step:0.001
    lr = round(lr, 3)

    # step:1
    num_layers = int(round(num_layers))
    num_neurons = int(round(num_neurons))

    # step:50
    num_domain_idx = int(round(num_domain_idx))
    num_boundary_idx = int(round(num_boundary_idx))
    num_initial_idx = int(round(num_initial_idx))

    # types of hyperparameters
    hparams = {
        "w_pde": w_pde,
        "w_abc": w_abc,
        "w_ini": w_ini,
        "lr": lr,
        "num_domain_idx": num_domain_idx,
        "num_boundary_idx": num_boundary_idx,
        "num_initial_idx": num_initial_idx,
        "num_layers": num_layers,
        "num_neurons": num_neurons,
    }

    loss_reduction_rate = train_and_evaluate(hparams)

    return loss_reduction_rate  # 直接返回损失变化率（贝叶斯优化会最大化这个值）

# 修改复合得分计算函数
def calculate_composite_score(evaluations, top_k=5, reduction_weight=0.5, time_weight=0.5):
    """
    Calculate composite score based on loss reduction rate and training time trade-off
    """
    if len(evaluations) < top_k:
        top_k = len(evaluations)
    
    # Sort by loss reduction rate (descending order) and get top k
    sorted_by_reduction = sorted(evaluations, key=lambda x: x['loss_reduction_rate'], reverse=True)
    top_k_candidates = sorted_by_reduction[:top_k]
    
    if len(top_k_candidates) == 1:
        return top_k_candidates[0]
    
    # Get worst (lowest) loss reduction rate and longest training time among top k
    worst_reduction = min(candidate['loss_reduction_rate'] for candidate in top_k_candidates)
    longest_time = max(candidate['training_time'] for candidate in top_k_candidates)
    
    # Calculate composite scores for each candidate
    for candidate in top_k_candidates:
        # Loss reduction rate improvement percentage (higher is better)
        best_reduction = max(c['loss_reduction_rate'] for c in top_k_candidates)
        if best_reduction > worst_reduction:
            # reduction_improvement = (candidate['loss_reduction_rate'] - worst_reduction) / (best_reduction - worst_reduction)
            reduction_improvement = (candidate['loss_reduction_rate'] - worst_reduction) / worst_reduction
        else:
            reduction_improvement = 1.0  # All candidates have the same reduction rate
        
        # Time improvement percentage (higher is better - shorter time is better)
        if longest_time > 0:
            time_improvement = (longest_time - candidate['training_time']) / longest_time
        else:
            time_improvement = 0
        
        # Composite score (weighted sum of improvements)
        composite_score = reduction_weight * reduction_improvement + time_weight * time_improvement
        candidate['composite_score'] = composite_score
        
        print(f"Candidate - Loss Reduction: {candidate['loss_reduction_rate']:.6f}, "
              f"Time: {candidate['training_time']:.2f}s, "
              f"Reduction Imp: {reduction_improvement:.3f}, Time Imp: {time_improvement:.3f}, "
              f"Composite: {composite_score:.3f}")
    
    # Return the candidate with the highest composite score
    best_candidate = max(top_k_candidates, key=lambda x: x['composite_score'])
    return best_candidate

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

class HybridAcquisitionFunction:
    """
    结合EI和UCB的混合获取函数
    """
    def __init__(self, ei_weight=0.6, ucb_weight=0.4, kappa=2.576):
        """
        Parameters:
        - ei_weight: EI的权重
        - ucb_weight: UCB的权重  
        - kappa: UCB的探索参数
        """
        self.ei_weight = ei_weight
        self.ucb_weight = ucb_weight
        self.kappa = kappa
        
    def expected_improvement(self, X, gp, y_max, xi=0.01):
        """
        计算Expected Improvement
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        with np.errstate(divide='warn'):
            imp = mu - y_max - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
    
    def upper_confidence_bound(self, X, gp):
        """
        计算Upper Confidence Bound
        """
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma
    
    def hybrid_acquisition(self, X, gp, y_max):
        """
        混合获取函数：EI + UCB
        """
        ei = self.expected_improvement(X, gp, y_max)
        ucb = self.upper_confidence_bound(X, gp)
        
        # 标准化EI和UCB到相同范围
        if len(ei) > 1:
            ei_norm = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-10)
            ucb_norm = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb) + 1e-10)
        else:
            ei_norm = ei
            ucb_norm = ucb
            
        return self.ei_weight * ei_norm + self.ucb_weight * ucb_norm

class ImprovedBayesianOptimizer:
    """
    改进的贝叶斯优化器，使用混合获取函数
    """
    def __init__(self, f, pbounds, acquisition_func=None, random_state=None):
        self.f = f
        self.pbounds = pbounds
        self.bounds = np.array(list(pbounds.values()))
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)
        
        if acquisition_func is None:
            self.acquisition_func = HybridAcquisitionFunction()
        else:
            self.acquisition_func = acquisition_func
            
        # 初始化GP
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=1e-6, 
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state
        )
        
        self.X_sample = []
        self.Y_sample = []
        self.best_params = {}
        self.best_target = -np.inf
        
    def _dict_to_array(self, param_dict):
        """将参数字典转换为数组"""
        return np.array([param_dict[key] for key in self.keys])
    
    def _array_to_dict(self, param_array):
        """将参数数组转换为字典"""
        return {key: param_array[i] for i, key in enumerate(self.keys)}
    
    def probe(self, params, lazy=True):
        """探测单个参数点"""
        x = self._dict_to_array(params)
        if not lazy:
            y = self.f(**params)
            self.X_sample.append(x)
            self.Y_sample.append(y)
            
            if y > self.best_target:
                self.best_target = y
                self.best_params = params.copy()
        else:
            # lazy模式，只添加到待评估列表
            self.X_sample.append(x)
            self.Y_sample.append(None)  # 占位符
    
    def _optimize_acquisition(self, n_candidates=1000, n_restarts=5):
        """优化获取函数以找到下一个采样点"""
        if len(self.Y_sample) < 2:
            # 如果样本太少，随机采样
            candidates = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1], 
                size=(n_candidates, self.dim)
            )
            return candidates[0]
        
        y_max = np.max(self.Y_sample)
        
        # 定义负获取函数（因为scipy.optimize.minimize是最小化）
        def negative_acquisition(x):
            x = x.reshape(1, -1)
            acq_value = self.acquisition_func.hybrid_acquisition(x, self.gp, y_max)
            return -acq_value[0]  # 返回负值用于最小化
        
        # 多次随机重启优化
        best_x = None
        best_acq = np.inf
        
        for _ in range(n_restarts):
            # 随机初始点
            x0 = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1], 
                size=self.dim
            )
            
            # 优化获取函数
            try:
                result = minimize(
                    negative_acquisition,
                    x0,
                    bounds=list(zip(self.bounds[:, 0], self.bounds[:, 1])),
                    method='L-BFGS-B'
                )
                
                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            except:
                # 如果优化失败，使用随机候选点方法
                candidates = np.random.uniform(
                    self.bounds[:, 0], 
                    self.bounds[:, 1], 
                    size=(n_candidates, self.dim)
                )
                
                acquisition_values = self.acquisition_func.hybrid_acquisition(
                    candidates, self.gp, y_max
                )
                
                best_idx = np.argmax(acquisition_values)
                if best_x is None:
                    best_x = candidates[best_idx]
        
        # 如果所有优化都失败，回退到随机采样
        if best_x is None:
            candidates = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1], 
                size=(n_candidates, self.dim)
            )
            best_x = candidates[0]
            
        return best_x
    
    def maximize(self, init_points=5, n_iter=25):
        """执行贝叶斯优化"""
        # 初始随机采样
        for _ in range(init_points):
            x_random = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1], 
                size=self.dim
            )
            params = self._array_to_dict(x_random)
            y = self.f(**params)
            
            self.X_sample.append(x_random)
            self.Y_sample.append(y)
            
            if y > self.best_target:
                self.best_target = y
                self.best_params = params.copy()
        
        # 贝叶斯优化迭代
        for iteration in range(n_iter):
            # 拟合GP模型
            X_array = np.array(self.X_sample)
            Y_array = np.array(self.Y_sample)
            
            self.gp.fit(X_array, Y_array)
            
            # 找到下一个采样点
            x_next = self._optimize_acquisition()
            params_next = self._array_to_dict(x_next)
            
            # 评估目标函数
            y_next = self.f(**params_next)
            
            # 更新样本
            self.X_sample.append(x_next)
            self.Y_sample.append(y_next)
            
            # 更新最优解
            if y_next > self.best_target:
                self.best_target = y_next
                self.best_params = params_next.copy()
            
            print(f"Iteration {iteration + 1}/{n_iter}: "
                  f"Current value = {y_next:.6f}, "
                  f"Best value = {self.best_target:.6f}")
    
    @property
    def max(self):
        """返回最优结果"""
        return {
            'target': self.best_target,
            'params': self.best_params
        }

# 修改原有的HeSBOWrapper类以适配新的优化器
class ImprovedHeSBOWrapper:
    def __init__(self, original_objective, embedding_dim, high_dim_bounds):
        self.original_objective = original_objective
        self.embedding_dim = embedding_dim
        self.high_dim_names = list(high_dim_bounds.keys())
        self.high_dim_bounds = np.array(list(high_dim_bounds.values()))
        self.high_dim = len(self.high_dim_names)
        
        # Generate random embedding matrix A (d x D)
        np.random.seed(42)  # For reproducibility
        self.A = np.random.normal(0, 1, (self.embedding_dim, self.high_dim))
        
    def normalize_params(self, z):
        """Normalize parameters from [-1, 1] to their actual bounds"""
        return 0.5 * (z + 1) * (self.high_dim_bounds[:, 1] - self.high_dim_bounds[:, 0]) + self.high_dim_bounds[:, 0]
    
    def embedded_objective(self, **kwargs):
        """Objective function in the embedded space"""
        # Extract parameters from the low-dimensional space
        z_low = np.array([kwargs[f'z{i}'] for i in range(self.embedding_dim)])
        
        # Project to the high-dimensional space using the matrix A
        z_high_normalized = self.A.T @ z_low
        
        # Clip to the range [-1, 1] to ensure parameters stay within bounds
        z_high_normalized = np.clip(z_high_normalized, -1, 1)
        
        # Transform from [-1, 1] to the actual parameter ranges
        x_high = self.normalize_params(z_high_normalized)
        
        # Convert to original parameter dictionary
        original_params = {self.high_dim_names[i]: x_high[i] for i in range(self.high_dim)}
        
        # Call the original objective function
        return self.original_objective(**original_params)

# 修改优化函数的参数名称和说明
def run_improved_bayesian_optimization_with_time_trade_off(initial_points=10, total_iterations=20, 
                                                         random_seed=1234, custom_pbounds=None, 
                                                         embedding_dim=2, top_k=5,
                                                         reduction_weight=0.5, time_weight=0.5,
                                                         ei_weight=0.4, ucb_weight=0.6):
    
    # Reset global variables    
    global global_best_error, global_best_weights, global_evaluation_history
    global_best_error = 0.0  # 改为0.0，因为我们要最大化损失变化率
    global_best_weights = None
    global_evaluation_history = []
    
    set_global_seed(random_seed)

    # Default hyperparameter search space for Burgers equation
    default_pbounds = {
        'w_pde': (0.01, 0.10),
        'w_abc': (0.01, 0.25),
        'w_ini': (0.01, 0.10),
        'lr': (0.001, 0.01),
        'num_domain_idx': (10, 100),
        'num_boundary_idx': (2, 60),
        'num_initial_idx': (2, 30),
        'num_layers': (2, 8),
        'num_neurons': (20, 80),
    }

    # Use custom bounds if provided, otherwise use defaults
    high_dim_bounds = custom_pbounds if custom_pbounds is not None else default_pbounds
    
    # Create improved HeSBO wrapper
    hesbo_wrapper = ImprovedHeSBOWrapper(
        original_objective=objective_for_bayes,
        embedding_dim=embedding_dim,
        high_dim_bounds=high_dim_bounds
    )
    
    # Define bounds for the embedded space
    embedded_pbounds = {f'z{i}': (-1, 1) for i in range(embedding_dim)}
    
    # 创建混合获取函数
    hybrid_acquisition = HybridAcquisitionFunction(
        ei_weight=ei_weight, 
        ucb_weight=ucb_weight, 
        kappa=2.576
    )
    
    # Initialize the improved optimizer
    optimizer = ImprovedBayesianOptimizer(
        f=hesbo_wrapper.embedded_objective,
        pbounds=embedded_pbounds,
        acquisition_func=hybrid_acquisition,
        random_state=random_seed
    )
    
    # Run optimization
    optimizer.maximize(init_points=initial_points, n_iter=total_iterations)
    
    print("\n" + "="*50)
    print("IMPROVED BAYESIAN OPTIMIZATION COMPLETED")
    print("="*50)
    
    # Perform post-processing to find the best trade-off solution
    print(f"\nPerforming post-processing analysis with top {top_k} candidates...")
    print(f"Loss reduction weight: {reduction_weight}, Time weight: {time_weight}")
    print(f"Acquisition function weights - EI: {ei_weight}, UCB: {ucb_weight}")
    print("-" * 50)
    
    best_trade_off = calculate_composite_score(
        global_evaluation_history, 
        top_k=top_k, 
        reduction_weight=reduction_weight, 
        time_weight=time_weight
    )
    
    print("-" * 50)
    print("FINAL SELECTION BASED ON COMPOSITE SCORE:")
    print(f"Selected Loss Reduction Rate: {best_trade_off['loss_reduction_rate']:.6f}")
    print(f"Selected Training Time: {best_trade_off['training_time']:.2f}s")
    print(f"Composite Score: {best_trade_off['composite_score']:.3f}")
    print(f"Initial Loss: {best_trade_off['initial_total_loss']:.6e}")
    print(f"Final Loss: {best_trade_off['final_total_loss']:.6e}")
    
    # Extract best parameters
    final_best_params = best_trade_off['hparams'].copy()
    final_best_reduction_rate = best_trade_off['loss_reduction_rate']
    final_best_weights = best_trade_off['weights']
    
    # Apply rounding
    final_best_params['w_pde'] = round(final_best_params['w_pde'], 4)
    final_best_params['w_abc'] = round(final_best_params['w_abc'], 4)
    final_best_params['w_ini'] = round(final_best_params['w_ini'], 4)
    final_best_params['lr'] = round(final_best_params['lr'], 3)
    final_best_params['num_layers'] = int(round(final_best_params['num_layers']))
    final_best_params['num_neurons'] = int(round(final_best_params['num_neurons']))
    final_best_params['num_domain_idx'] = int(round(final_best_params['num_domain_idx']))
    final_best_params['num_boundary_idx'] = int(round(final_best_params['num_boundary_idx']))
    final_best_params['num_initial_idx'] = int(round(final_best_params['num_initial_idx']))

    print("Final Best Parameters:", final_best_params)
    print("="*50)

    return final_best_params, final_best_reduction_rate, final_best_weights

#-------------------------------------------------------------------------------#
# Gradient
#-------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class DeepXDEGradientExtractor:
    """
    专门用于从DeepXDE模型中提取各个损失分量梯度的类
    """
    def __init__(self, model, data, pde_func, enable_logging=False):
        self.model = model
        self.data = data
        self.pde_func = pde_func
        self.enable_logging = enable_logging
        
        # 存储梯度信息
        self.gradients = {
            'pde': [],
            'bc': [],
            'ic': []
        }
        
        # 存储损失历史
        self.loss_history = []
        
    def extract_individual_gradients(self):
        """
        提取各个损失分量的梯度信息
        返回梯度范数和方向信息
        """
        self.model.net.train()
        
        # 🔧 修复：使用与模型相同的数据类型（float64）
        train_x = torch.tensor(self.data.train_x_all, dtype=torch.float64, requires_grad=True)
        
        # 分离不同类型的点
        pde_points = train_x[:self.data.num_domain]
        bc_points = train_x[self.data.num_domain:self.data.num_domain + self.data.num_boundary]
        ic_points = train_x[self.data.num_domain + self.data.num_boundary:]
        
        gradients_info = {}
        
        # 1. 计算PDE损失的梯度
        if len(pde_points) > 0:
            try:
                pde_pred = self.model.net(pde_points)
                pde_residual = self._compute_pde_residual(pde_points, pde_pred)
                pde_loss = torch.mean(pde_residual ** 2)
                
                # 计算梯度
                pde_gradients = torch.autograd.grad(
                    pde_loss, 
                    self.model.net.parameters(), 
                    retain_graph=True, 
                    create_graph=False,
                    allow_unused=True  # 🔧 添加这个参数处理未使用的参数
                )
                
                # 过滤None梯度并计算梯度范数
                valid_pde_gradients = [g for g in pde_gradients if g is not None]
                if valid_pde_gradients:
                    pde_grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in valid_pde_gradients))
                else:
                    pde_grad_norm = torch.tensor(0.0, dtype=torch.float64)
                
                gradients_info['pde'] = {
                    'loss': pde_loss.item(),
                    'grad_norm': pde_grad_norm.item(),
                    'gradients': [g.clone().detach() if g is not None else None for g in pde_gradients]
                }
            except Exception as e:
                print(f"Error computing PDE gradients: {e}")
                gradients_info['pde'] = {'loss': 0.0, 'grad_norm': 0.0, 'gradients': []}
        
        # 2. 计算边界条件损失的梯度
        if len(bc_points) > 0:
            try:
                bc_pred = self.model.net(bc_points)
                bc_loss = torch.mean(bc_pred ** 2)  # 假设边界条件为0
                
                bc_gradients = torch.autograd.grad(
                    bc_loss, 
                    self.model.net.parameters(), 
                    retain_graph=True, 
                    create_graph=False,
                    allow_unused=True
                )
                
                valid_bc_gradients = [g for g in bc_gradients if g is not None]
                if valid_bc_gradients:
                    bc_grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in valid_bc_gradients))
                else:
                    bc_grad_norm = torch.tensor(0.0, dtype=torch.float64)
                
                gradients_info['bc'] = {
                    'loss': bc_loss.item(),
                    'grad_norm': bc_grad_norm.item(),
                    'gradients': [g.clone().detach() if g is not None else None for g in bc_gradients]
                }
            except Exception as e:
                print(f"Error computing BC gradients: {e}")
                gradients_info['bc'] = {'loss': 0.0, 'grad_norm': 0.0, 'gradients': []}
        
        # 3. 计算初始条件损失的梯度
        if len(ic_points) > 0:
            try:
                ic_pred = self.model.net(ic_points)
                # 初始条件: u(x,0) = -sin(π*x)
                ic_target = -torch.sin(torch.pi * ic_points[:, 0:1])
                ic_loss = torch.mean((ic_pred - ic_target) ** 2)
                
                ic_gradients = torch.autograd.grad(
                    ic_loss, 
                    self.model.net.parameters(), 
                    retain_graph=False, 
                    create_graph=False,
                    allow_unused=True
                )
                
                valid_ic_gradients = [g for g in ic_gradients if g is not None]
                if valid_ic_gradients:
                    ic_grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in valid_ic_gradients))
                else:
                    ic_grad_norm = torch.tensor(0.0, dtype=torch.float64)
                
                gradients_info['ic'] = {
                    'loss': ic_loss.item(),
                    'grad_norm': ic_grad_norm.item(),
                    'gradients': [g.clone().detach() if g is not None else None for g in ic_gradients]
                }
            except Exception as e:
                print(f"Error computing IC gradients: {e}")
                gradients_info['ic'] = {'loss': 0.0, 'grad_norm': 0.0, 'gradients': []}
        
        # 存储历史记录
        self.gradients['pde'].append(gradients_info.get('pde', {'grad_norm': 0.0}))
        self.gradients['bc'].append(gradients_info.get('bc', {'grad_norm': 0.0}))
        self.gradients['ic'].append(gradients_info.get('ic', {'grad_norm': 0.0}))
        
        if self.enable_logging:
            print(f"Gradient Norms - PDE: {gradients_info.get('pde', {}).get('grad_norm', 0):.6e}, "
                  f"BC: {gradients_info.get('bc', {}).get('grad_norm', 0):.6e}, "
                  f"IC: {gradients_info.get('ic', {}).get('grad_norm', 0):.6e}")
        
        return gradients_info
    
    def _compute_pde_residual(self, x, y):
        """计算PDE残差"""
        try:
            x.requires_grad_(True)
            y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                                      create_graph=True, retain_graph=True)[0]
            y_xx = torch.autograd.grad(y_x[:, 0:1], x, grad_outputs=torch.ones_like(y_x[:, 0:1]), 
                                       create_graph=True, retain_graph=True)[0][:, 0:1]
            
            dy_x = y_x[:, 0:1]
            dy_t = y_x[:, 1:2]
            dy_xx = y_xx
            
            # Burgers方程: du/dt + u * du/dx - 0.01/π * d²u/dx² = 0
            pde_residual = dy_t + y * dy_x - 0.01 / np.pi * dy_xx
            return pde_residual.squeeze()
        except Exception as e:
            print(f"Error computing PDE residual: {e}")
            # 返回零残差以避免训练中断
            return torch.zeros(y.shape[0], dtype=torch.float64)
    
    def get_gradient_statistics(self, window_size=10):
        """获取梯度统计信息"""
        if len(self.gradients['pde']) < 2:
            return None
            
        recent_pde = [g['grad_norm'] for g in self.gradients['pde'][-window_size:]]
        recent_bc = [g['grad_norm'] for g in self.gradients['bc'][-window_size:]]
        recent_ic = [g['grad_norm'] for g in self.gradients['ic'][-window_size:]]
        
        return {
            'pde_grad_mean': np.mean(recent_pde),
            'pde_grad_std': np.std(recent_pde),
            'bc_grad_mean': np.mean(recent_bc),
            'bc_grad_std': np.std(recent_bc),
            'ic_grad_mean': np.mean(recent_ic),
            'ic_grad_std': np.std(recent_ic),
        }

#-------------------------------------------------------------------------------#
# EMA for adjusting loss function weights
#-------------------------------------------------------------------------------#
class FixedTrulyDynamicGradientLbPINNsUpdater:
    
    def __init__(self, gradient_extractor, beta=0.9, gamma=0.8,  # 🔧 使用更稳定的平滑参数
                 init_pde=1.0, init_abc=1.0, init_ini=1.0,
                 init_w_pde=0.1, init_w_abc=0.1, init_w_ini=0.1, 
                 learning_rate=0.01, gradient_weight=1.0):  # 🔧 降低学习率和梯度权重
        """
        使用更稳定的参数，适合500步更新一次的策略
        """
        self.gradient_extractor = gradient_extractor
        
        # 🔧 使用更稳定的平滑参数
        self.beta = beta  # 增加到0.9，更平滑的EMA
        self.gamma = gamma  # 增加到0.8，更稳定的权重更新
        self.m_pde = init_pde
        self.m_abc = init_abc
        self.m_ini = init_ini
        self.gradient_weight = gradient_weight  # 降低到1.0
        
        # 处理tensor到float的转换
        if torch.is_tensor(init_w_pde):
            init_w_pde = init_w_pde.detach().cpu().item()
        if torch.is_tensor(init_w_abc):
            init_w_abc = init_w_abc.detach().cpu().item()
        if torch.is_tensor(init_w_ini):
            init_w_ini = init_w_ini.detach().cpu().item()
        
        # 从初始权重推导初始log_variance
        init_log_var_pde = -np.log(2 * init_w_pde + 1e-8)
        init_log_var_abc = -np.log(2 * init_w_abc + 1e-8)
        init_log_var_ini = -np.log(2 * init_w_ini + 1e-8)
        
        # 可学习的log variance参数
        self.log_variance_pde = torch.tensor(init_log_var_pde, requires_grad=True, dtype=torch.float64)
        self.log_variance_abc = torch.tensor(init_log_var_abc, requires_grad=True, dtype=torch.float64)
        self.log_variance_ini = torch.tensor(init_log_var_ini, requires_grad=True, dtype=torch.float64)
        
        # 🔧 使用更稳定的学习率
        self.optimizer = torch.optim.Adam([self.log_variance_pde, self.log_variance_abc, self.log_variance_ini], 
                                         lr=learning_rate)
        
        # 梯度相关参数
        self.grad_ema = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}
        
        # 当前权重
        self.current_w_pde = init_w_pde
        self.current_w_abc = init_w_abc
        self.current_w_ini = init_w_ini
        self.update_count = 0
        
        # 🔧 适应500步更新策略：只在真正需要时强制更新
        self.force_update_first_steps = 2  # 只强制前2次更新
        
        # 🔧 添加调试标志
        self.debug_mode = True
        self.gradient_extraction_success_count = 0
        self.gradient_extraction_fail_count = 0
        
        # 历史记录
        self.history = []
        
        print(f"Fixed Truly Dynamic Gradient-Enhanced lbPINNs Updater initialized:")
        print(f"  Initial weights: PDE={init_w_pde:.4f}, BC={init_w_abc:.4f}, IC={init_w_ini:.4f}")
        print(f"  Smooth parameters: beta={beta:.3f}, gamma={gamma:.3f}")
        print(f"  Learning rate: {learning_rate:.4f}, Gradient weight: {gradient_weight:.3f}")
        print(f"  Strategy: Stable updates every 500 Adam steps")
    
    def update(self, loss_pde, loss_abc, loss_ini):
        """
        修复的权重更新，确保每步都有明显变化
        """
        self.update_count += 1
        
        # 🔧 1. 尝试提取梯度信息
        gradient_available = False
        try:
            gradient_info = self.gradient_extractor.extract_individual_gradients()
            gradient_available = True
            self.gradient_extraction_success_count += 1
            
            if self.debug_mode and self.update_count <= 10:
                print(f"  Step {self.update_count}: Gradient extraction successful")
                
        except Exception as e:
            self.gradient_extraction_fail_count += 1
            if self.debug_mode:
                print(f"  Step {self.update_count}: Gradient extraction failed: {str(e)[:100]}")
            
            # 使用估计的梯度信息
            gradient_info = {
                'pde': {'grad_norm': abs(loss_pde) * 1e3},
                'bc': {'grad_norm': abs(loss_abc) * 1e3},
                'ic': {'grad_norm': abs(loss_ini) * 1e3}
            }
        
        # 2. 更新损失EMA
        self.m_pde = self.beta * self.m_pde + (1 - self.beta) * loss_pde
        self.m_abc = self.beta * self.m_abc + (1 - self.beta) * loss_abc
        self.m_ini = self.beta * self.m_ini + (1 - self.beta) * loss_ini
        
        # 3. 提取梯度信息
        pde_grad_norm = max(gradient_info.get('pde', {}).get('grad_norm', 1e-8), 1e-10)
        bc_grad_norm = max(gradient_info.get('bc', {}).get('grad_norm', 1e-8), 1e-10)
        ic_grad_norm = max(gradient_info.get('ic', {}).get('grad_norm', 1e-8), 1e-10)
        
        # 4. 更新梯度EMA
        self.grad_ema['pde'] = self.beta * self.grad_ema['pde'] + (1 - self.beta) * pde_grad_norm
        self.grad_ema['bc'] = self.beta * self.grad_ema['bc'] + (1 - self.beta) * bc_grad_norm
        self.grad_ema['ic'] = self.beta * self.grad_ema['ic'] + (1 - self.beta) * ic_grad_norm
        
        # 5. 转换为tensor
        loss_pde_tensor = torch.tensor(float(loss_pde), dtype=torch.float64)
        loss_abc_tensor = torch.tensor(float(loss_abc), dtype=torch.float64)
        loss_ini_tensor = torch.tensor(float(loss_ini), dtype=torch.float64)
        
        # 6. 计算当前权重
        w_pde_current = torch.exp(-self.log_variance_pde) / 2.0
        w_abc_current = torch.exp(-self.log_variance_abc) / 2.0
        w_ini_current = torch.exp(-self.log_variance_ini) / 2.0
        
        # 7. 计算增强的目标函数
        weighted_loss = (w_pde_current * loss_pde_tensor + 
                        w_abc_current * loss_abc_tensor + 
                        w_ini_current * loss_ini_tensor)
        regularization = self.log_variance_pde + self.log_variance_abc + self.log_variance_ini
        
        # 🚀 强化的梯度平衡项
        gradient_balance_loss = torch.tensor(0.0, dtype=torch.float64)
        if self.gradient_weight > 0:
            # 计算梯度不平衡度
            grad_tensor_pde = torch.tensor(float(pde_grad_norm), dtype=torch.float64)
            grad_tensor_bc = torch.tensor(float(bc_grad_norm), dtype=torch.float64)
            grad_tensor_ic = torch.tensor(float(ic_grad_norm), dtype=torch.float64)
            
            # 计算加权梯度
            weighted_grad_pde = w_pde_current * grad_tensor_pde
            weighted_grad_bc = w_abc_current * grad_tensor_bc
            weighted_grad_ic = w_ini_current * grad_tensor_ic
            
            # 目标：让加权梯度尽可能接近
            mean_weighted_grad = (weighted_grad_pde + weighted_grad_bc + weighted_grad_ic) / 3.0
            
            gradient_balance_loss = (
                ((weighted_grad_pde - mean_weighted_grad) ** 2) +
                ((weighted_grad_bc - mean_weighted_grad) ** 2) +
                ((weighted_grad_ic - mean_weighted_grad) ** 2)
            )
        
        total_loss = weighted_loss + regularization + self.gradient_weight * gradient_balance_loss
        
        # 🔧 8. 修复更新条件：前几步强制更新，之后每2步更新一次
        should_update = (self.update_count <= self.force_update_first_steps or 
                        self.update_count % 2 == 0)
        
        if should_update:
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 检查梯度是否存在
            grad_exists = any(param.grad is not None and torch.sum(torch.abs(param.grad)) > 1e-12 
                            for param in [self.log_variance_pde, self.log_variance_abc, self.log_variance_ini])
            
            if grad_exists:
                # 适度的梯度裁剪
                torch.nn.utils.clip_grad_norm_([self.log_variance_pde, self.log_variance_abc, self.log_variance_ini], 
                                             max_norm=1.0)
                self.optimizer.step()
                
                if self.debug_mode and self.update_count <= 20:
                    print(f"  Step {self.update_count}: Optimizer step executed")
            else:
                if self.debug_mode and self.update_count <= 20:
                    print(f"  Step {self.update_count}: No meaningful gradients, skipping optimizer step")
            
            # 🔧 适度放宽log_variance范围限制
            with torch.no_grad():
                # 对应权重范围 [0.005, 0.3]: 合理的范围
                max_log_var = -np.log(2 * 0.005)  # 约 5.30
                min_log_var = -np.log(2 * 0.3)    # 约 -0.41
                
                self.log_variance_pde.clamp_(min_log_var, max_log_var)
                self.log_variance_abc.clamp_(min_log_var, max_log_var)
                self.log_variance_ini.clamp_(min_log_var, max_log_var)
        
        # 🔧 9. 权重更新 - 使用更小的gamma实现更明显的变化
        with torch.no_grad():
            w_pde_new = torch.exp(-self.log_variance_pde) / 2.0
            w_abc_new = torch.exp(-self.log_variance_abc) / 2.0
            w_ini_new = torch.exp(-self.log_variance_ini) / 2.0
            
            # 记录变化前的权重
            old_w_pde = self.current_w_pde
            old_w_abc = self.current_w_abc
            old_w_ini = self.current_w_ini
            
            # 🔧 使用稳定的权重更新策略
            effective_gamma = self.gamma  # 统一使用gamma，不再区分前几步
            
            # 使用有效的gamma进行权重更新
            self.current_w_pde = effective_gamma * self.current_w_pde + (1 - effective_gamma) * w_pde_new.item()
            self.current_w_abc = effective_gamma * self.current_w_abc + (1 - effective_gamma) * w_abc_new.item()
            self.current_w_ini = effective_gamma * self.current_w_ini + (1 - effective_gamma) * w_ini_new.item()
            
            # 🔧 适度放宽权重范围限制
            self.w_pde = torch.clamp(torch.tensor(self.current_w_pde, dtype=torch.float64), 0.005, 0.3)
            self.w_abc = torch.clamp(torch.tensor(self.current_w_abc, dtype=torch.float64), 0.005, 0.3)
            self.w_ini = torch.clamp(torch.tensor(self.current_w_ini, dtype=torch.float64), 0.005, 0.3)
            
            # 计算变化量
            weight_change_pde = abs(self.w_pde.item() - old_w_pde)
            weight_change_abc = abs(self.w_abc.item() - old_w_abc)
            weight_change_ini = abs(self.w_ini.item() - old_w_ini)
            total_weight_change = weight_change_pde + weight_change_abc + weight_change_ini

        # 10. 记录历史
        self.history.append({
            'step': self.update_count,
            'losses': [loss_pde, loss_abc, loss_ini],
            'weights': [self.w_pde.item(), self.w_abc.item(), self.w_ini.item()],
            'weight_changes': [weight_change_pde, weight_change_abc, weight_change_ini],
            'log_variances': [self.log_variance_pde.item(), self.log_variance_abc.item(), self.log_variance_ini.item()],
            'grad_norms': [pde_grad_norm, bc_grad_norm, ic_grad_norm],
            'gradient_balance_loss': gradient_balance_loss.item(),
            'total_loss': total_loss.item(),
            'gradient_available': gradient_available,
            'total_weight_change': total_weight_change,
            'should_update': should_update,
            'effective_gamma': effective_gamma if 'effective_gamma' in locals() else self.gamma
        })
        
        # 🔧 调整打印频率：由于更新频率降低，每次更新都打印
        if True:  # 每次更新都打印详细信息
            print(f"Dynamic Update {self.update_count}:")
            print(f"  Weights: PDE={self.w_pde.item():.6f}, BC={self.w_abc.item():.6f}, IC={self.w_ini.item():.6f}")
            print(f"  Weight Changes: PDE={weight_change_pde:.6f}, BC={weight_change_abc:.6f}, IC={weight_change_ini:.6f}")
            print(f"  Total Change: {total_weight_change:.8f}")
            print(f"  Should Update: {should_update}, Effective Gamma: {effective_gamma:.3f}")
            print(f"  Losses: PDE={loss_pde:.6e}, BC={loss_abc:.6e}, IC={loss_ini:.6e}")
            print(f"  Grad Norms: PDE={pde_grad_norm:.6e}, BC={bc_grad_norm:.6e}, IC={ic_grad_norm:.6e}")
            print(f"  Gradient Balance Loss: {gradient_balance_loss.item():.6e}")
        
        return self.w_pde, self.w_abc, self.w_ini
    
    def get_weight_change_statistics(self):
        """获取权重变化统计信息"""
        if len(self.history) < 5:
            return None
            
        recent_changes = [h['total_weight_change'] for h in self.history[-20:]]
        recent_weights = [h['weights'] for h in self.history[-10:]]
        
        # 计算权重范围
        all_weights = np.array(recent_weights)
        weight_ranges = {
            'pde_range': np.max(all_weights[:, 0]) - np.min(all_weights[:, 0]),
            'bc_range': np.max(all_weights[:, 1]) - np.min(all_weights[:, 1]),
            'ic_range': np.max(all_weights[:, 2]) - np.min(all_weights[:, 2])
        }
        
        return {
            'avg_total_weight_change': np.mean(recent_changes),
            'max_total_weight_change': np.max(recent_changes),
            'gradient_success_rate': self.gradient_extraction_success_count / max(self.update_count, 1),
            'weight_ranges': weight_ranges,
            'total_updates': self.update_count,
            'recent_changes': recent_changes[-5:]  # 最近5步的变化
        }
    
#-------------------------------------------------------------------------------#
# Enhanced RAR-D for dynamic sampling during training (flexible for different PDE types)
#-------------------------------------------------------------------------------#
class EnhancedDynamicRarDSampler:
    def __init__(self, model, geom, data, pde, bc_list=None, ic_list=None, tol=1e-3, 
                num_new_pde_points=50, num_new_bc_points=20, num_new_ic_points=20,
                enable_pde_sampling=True, enable_bc_sampling=True, enable_ic_sampling=True):
        self.model = model  
        self.geom = geom  # geomtime
        self.data = data 
        self.pde = pde  
        
        # 修改这部分，处理单个BC对象或BC列表
        if bc_list is None:
            self.bc_list = []
        elif isinstance(bc_list, list):
            self.bc_list = bc_list
        else:
            # 如果是单个BC对象，转换为列表
            self.bc_list = [bc_list]
        
        # 修改这部分，处理单个IC对象或IC列表    
        if ic_list is None:
            self.ic_list = []
        elif isinstance(ic_list, list):
            self.ic_list = ic_list
        else:
            # 如果是单个IC对象，转换为列表
            self.ic_list = [ic_list]
        
        self.tol = tol
        self.num_new_pde_points = num_new_pde_points
        self.num_new_bc_points = num_new_bc_points
        self.num_new_ic_points = num_new_ic_points
        self.sampling_count = 0
        
        # Flexible sampling control - 现在可以安全地检查长度
        self.enable_pde_sampling = enable_pde_sampling
        self.enable_bc_sampling = enable_bc_sampling and len(self.bc_list) > 0
        self.enable_ic_sampling = enable_ic_sampling and len(self.ic_list) > 0
        
        print(f"RAR-D Sampler initialized with:")
        print(f"  PDE sampling: {'Enabled' if self.enable_pde_sampling else 'Disabled'}")
        print(f"  BC sampling: {'Enabled' if self.enable_bc_sampling else 'Disabled'} ({len(self.bc_list)} BCs)")
        print(f"  IC sampling: {'Enabled' if self.enable_ic_sampling else 'Disabled'} ({len(self.ic_list)} ICs)")

    def filter_unique_points(self, existing_points, new_points):
        """
        Filters out duplicate points from new_points that are too close to existing_points based on tolerance.
        """
        if len(new_points) == 0:
            return np.array([])
            
        unique_points = []
        for p in new_points:
            dists = np.linalg.norm(existing_points - p, axis=1)
            if np.min(dists) > self.tol:
                unique_points.append(p)

        return np.array(unique_points)

    def sample_boundary_points(self):
        """
        Simplified boundary condition sampling using direct evaluation.
        Only executes if BC sampling is enabled and BC list is not empty.
        """
        if not self.enable_bc_sampling or len(self.bc_list) == 0:
            return np.array([])
            
        new_bc_points = []
        
        try:
            # Generate boundary points using the geomtime's boundary sampling
            # For GeometryXTime with Interval(-1,1) and TimeDomain(0,1):
            # Boundary points are at x=-1 or x=1 for any t in [0,1]
            
            # Sample time points
            num_candidates = 200
            t_samples = np.random.uniform(0, 1, (num_candidates, 1))
            
            # Create boundary points at x=-1 and x=1
            left_boundary = np.hstack([np.full((num_candidates, 1), -1.0), t_samples])
            right_boundary = np.hstack([np.full((num_candidates, 1), 1.0), t_samples])
            
            candidate_points = np.vstack([left_boundary, right_boundary])
            
            if len(candidate_points) == 0:
                return np.array([])
            
            # Predict values at boundary points
            bc_pred = self.model.predict(candidate_points)
            
            # Calculate boundary condition residuals
            bc_residuals = []
            valid_points_list = []
            
            for bc in self.bc_list:
                for i, point in enumerate(candidate_points):
                    try:
                        x_point = point.reshape(1, -1)
                        pred_val = bc_pred[i:i+1]
                        
                        # Calculate target value - BC function should return 0 for Dirichlet BC
                        # bc.func typically returns the target value at boundary
                        target_val = 0.0  # For Dirichlet BC with value 0
                        
                        residual = np.abs(pred_val - target_val).item()
                        if residual > 1e-10:  # Only consider significant residuals
                            bc_residuals.append(residual)
                            valid_points_list.append(point)
                            
                    except Exception as e:
                        continue
            
            if len(bc_residuals) == 0:
                return np.array([])
                
            bc_residuals = np.array(bc_residuals)
            valid_points = np.array(valid_points_list)
            
            # Create probability distribution based on residuals
            sum_res = np.sum(bc_residuals)
            if sum_res == 0:
                p_distribution = np.ones_like(bc_residuals) / len(bc_residuals)
            else:
                p_distribution = bc_residuals / sum_res
            
            # Select new boundary points
            num_select = min(self.num_new_bc_points, len(valid_points))
            if num_select > 0:
                new_indices = np.random.choice(
                    a=len(valid_points),
                    size=num_select,
                    replace=False,
                    p=p_distribution
                )
                selected_points = valid_points[new_indices]
                new_bc_points.extend(selected_points)
                
        except Exception as e:
            print(f"Error in boundary sampling: {e}")
            return np.array([])
        
        return np.array(new_bc_points) if new_bc_points else np.array([])

    def sample_initial_points(self):
        """
        Simplified initial condition sampling using direct evaluation.
        Only executes if IC sampling is enabled and IC list is not empty.
        """
        if not self.enable_ic_sampling or len(self.ic_list) == 0:
            return np.array([])
            
        new_ic_points = []
        
        try:
            # Generate initial condition points (t=0)
            num_candidates = 200
            x_samples = np.random.uniform(-1, 1, (num_candidates, 1))  # x in [-1, 1]
            t_samples = np.zeros((num_candidates, 1))  # t = 0
            
            candidate_points = np.hstack([x_samples, t_samples])
            
            if len(candidate_points) == 0:
                return np.array([])
            
            # Predict values at initial points
            ic_pred = self.model.predict(candidate_points)
            
            # Calculate initial condition residuals
            ic_residuals = []
            valid_points_list = []
            
            for ic in self.ic_list:
                for i, point in enumerate(candidate_points):
                    try:
                        x_point = point.reshape(1, -1)
                        pred_val = ic_pred[i:i+1]
                        
                        # Calculate target value using IC function
                        # ic.func should be: lambda x: -np.sin(np.pi * x[:, 0:1])
                        target_val = -np.sin(np.pi * x_point[:, 0:1])
                        
                        residual = np.abs(pred_val - target_val).item()
                        if residual > 1e-10:  # Only consider significant residuals
                            ic_residuals.append(residual)
                            valid_points_list.append(point)
                            
                    except Exception as e:
                        continue
            
            if len(ic_residuals) == 0:
                return np.array([])
                
            ic_residuals = np.array(ic_residuals)
            valid_points = np.array(valid_points_list)
            
            # Create probability distribution based on residuals
            sum_res = np.sum(ic_residuals)
            if sum_res == 0:
                p_distribution = np.ones_like(ic_residuals) / len(ic_residuals)
            else:
                p_distribution = ic_residuals / sum_res
            
            # Select new initial points
            num_select = min(self.num_new_ic_points, len(valid_points))
            if num_select > 0:
                new_indices = np.random.choice(
                    a=len(valid_points),
                    size=num_select,
                    replace=False,
                    p=p_distribution
                )
                selected_points = valid_points[new_indices]
                new_ic_points.extend(selected_points)
                
        except Exception as e:
            print(f"Error in initial condition sampling: {e}")
            return np.array([])
        
        return np.array(new_ic_points) if new_ic_points else np.array([])

    def sample_pde_points(self):
        """
        Adaptive sampling for PDE residual points.
        Only executes if PDE sampling is enabled.
        """
        if not self.enable_pde_sampling:
            return np.array([])
            
        # Generate candidate points for PDE
        candidate_points = self.geom.random_points(1000)

        try:
            residuals = self.model.predict(candidate_points, operator=self.pde)
        except Exception as e:
            print(f"Error computing PDE residuals: {e}")
            return np.array([])

        # Calculate the residual for each point
        residuals_combined = np.abs(residuals).ravel()

        # Create probability distribution
        sum_res = np.sum(residuals_combined)
        if sum_res == 0:
            p_distribution = np.ones_like(residuals_combined) / len(residuals_combined)
        else:
            p_distribution = residuals_combined / sum_res

        # Select new points based on the probability distribution
        new_indices = np.random.choice(
            a=len(candidate_points),
            size=min(self.num_new_pde_points, len(candidate_points)),
            replace=False,
            p=p_distribution
        )
        new_points = candidate_points[new_indices]
        
        return new_points

    def adaptive_sample_once(self):
        """
        Performs one iteration of enhanced RAR-D adaptive sampling for PDE, BC, and IC.
        Returns the total number of points actually added.
        """
        self.sampling_count += 1
        print(f"Enhanced RAR-D Sampling iteration {self.sampling_count}")
        
        total_points_added = 0
        existing_points = self.data.train_x_all
        sampling_summary = []
        
        # 1. Sample PDE points (if enabled)
        if self.enable_pde_sampling:
            new_pde_points = self.sample_pde_points()
            if len(new_pde_points) > 0:
                new_pde_unique = self.filter_unique_points(existing_points, new_pde_points)
                if len(new_pde_unique) > 0:
                    self.data.add_anchors(new_pde_unique)
                    total_points_added += len(new_pde_unique)
                    sampling_summary.append(f"PDE: {len(new_pde_unique)}")
                    # Update existing points for next filter
                    existing_points = self.data.train_x_all
        
        # 2. Sample boundary condition points (if enabled and BC list is not empty)
        if self.enable_bc_sampling:
            new_bc_points = self.sample_boundary_points()
            if len(new_bc_points) > 0:
                new_bc_unique = self.filter_unique_points(existing_points, new_bc_points)
                if len(new_bc_unique) > 0:
                    self.data.add_anchors(new_bc_unique)
                    total_points_added += len(new_bc_unique)
                    sampling_summary.append(f"BC: {len(new_bc_unique)}")
                    # Update existing points for next filter
                    existing_points = self.data.train_x_all
        
        # 3. Sample initial condition points (if enabled and IC list is not empty)
        if self.enable_ic_sampling:
            new_ic_points = self.sample_initial_points()
            if len(new_ic_points) > 0:
                new_ic_unique = self.filter_unique_points(existing_points, new_ic_points)
                if len(new_ic_unique) > 0:
                    self.data.add_anchors(new_ic_unique)
                    total_points_added += len(new_ic_unique)
                    sampling_summary.append(f"IC: {len(new_ic_unique)}")
        
        # Print summary
        if sampling_summary:
            print(f"  Added points - {', '.join(sampling_summary)}")
        else:
            print("  No points added this iteration")
        print(f"  Total points added: {total_points_added}")
        return total_points_added

#-------------------------------------------------------------------------------#
# Define callback class for calculating L2 relative error in training process
#-------------------------------------------------------------------------------#
class L2ErrorAndRarDCallback(dde.callbacks.Callback):
    def __init__(self, test_x, true_u, rar_sampler=None, every=1000, sampling_every=1000):
        super().__init__()
        self.test_x = test_x
        self.true_u = true_u
        self.rar_sampler = rar_sampler
        self.every = every
        self.sampling_every = sampling_every

    def on_epoch_end(self):
        current_step = self.model.train_state.step
        
        # Calculate L2 error
        if current_step % self.every == 0:
            pred = self.model.predict(self.test_x)
            l2_error = dde.metrics.l2_relative_error(self.true_u, pred)
            print(f"Step {current_step}: L2 error: {l2_error:.6e}")
        
        # Perform RAR-D sampling
        if (self.rar_sampler is not None and 
            current_step > 0 and 
            current_step % self.sampling_every == 0):
            
            print(f"Step {current_step}: Performing RAR-D adaptive sampling...")
            points_added = self.rar_sampler.adaptive_sample_once()
            total_points = len(self.rar_sampler.data.train_x_all)
            print(f"  Total training points now: {total_points}")

#-------------------------------------------------------------------------------#
# training process
#-------------------------------------------------------------------------------#
def train_and_evaluate_with_fixed_gradient_enhanced_lbpinns():
    """
    使用修复的梯度增强lbPINNs训练函数
    """
    
    # 首先运行贝叶斯优化获取最佳超参数
    best_params, best_reduction_rate, best_weights = run_improved_bayesian_optimization_with_time_trade_off(
        initial_points=10,
        total_iterations=20,
        random_seed=42,
        custom_pbounds={
            'w_pde': (0.005, 0.10),
            'w_abc': (0.01, 0.25),
            'w_ini': (0.005, 0.10),
            'lr': (0.001, 0.01),
            'num_domain_idx': (10, 100),
            'num_boundary_idx': (2, 60),
            'num_initial_idx': (2, 30),
            'num_layers': (2, 8),
            'num_neurons': (20, 80),
        },
        embedding_dim=3,
        ei_weight=0.4,
        ucb_weight=0.6,
        reduction_weight=0.5,
        time_weight=0.5
    )

    # 获取优化后的超参数
    w_pde = best_params['w_pde']
    w_abc = best_params['w_abc']
    w_ini = best_params['w_ini']
    lr = best_params['lr']
    num_domain = int(round(best_params['num_domain_idx'])) * 50
    num_boundary = int(round(best_params['num_boundary_idx'])) * 50
    num_initial = int(round(best_params['num_initial_idx'])) * 50
    num_layers = int(round(best_params['num_layers']))
    num_neurons = int(round(best_params['num_neurons']))

    print(f"\n{'='*60}")
    print("STARTING FINAL TRAINING WITH FIXED GRADIENT-ENHANCED lbPINNs")
    print(f"{'='*60}")
    
    # 设置初始权重
    loss_weights = [w_pde, w_abc, w_ini]

    # 定义PDE数据
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=2500
    )

    # 创建网络和模型
    layer_sizes = [2] + [num_neurons] * num_layers + [1]
    net = dde.nn.FNN(layer_sizes, wavelet_tanh_gaussian, "Glorot uniform")
    model = dde.Model(data, net)
    
    # 加载最佳权重（如果有）
    if best_weights is not None:
        parameters = list(model.net.parameters())
        layer_names = []
        for i in range(len(parameters)):
            if i % 2 == 0:
                layer_names.append(f"linears.{i//2}.weight")
            else:
                layer_names.append(f"linears.{i//2}.bias")
        
        best_weights_dict = {}
        for i in range(min(len(best_weights), len(layer_names))):
            param_name = layer_names[i]
            param_tensor = best_weights[i]
            current_param = parameters[i]
            
            if param_tensor is not None and param_tensor.shape == current_param.shape:
                best_weights_dict[param_name] = param_tensor
            else:
                print(f"Skipping parameter {param_name} due to shape mismatch or None")
                best_weights_dict[param_name] = current_param
        
        model.net.load_state_dict(best_weights_dict, strict=False)
    
    # 🔧 创建梯度提取器
    gradient_extractor = DeepXDEGradientExtractor(
        model=model,
        data=data,
        pde_func=pde,
        enable_logging=False  # 减少日志输出
    )
    
    # 准备测试数据
    test_points = data.test_x
    test_data = np.load("Burgers.npz")
    t_test, x_test, usol_test = test_data["t"], test_data["x"], test_data["usol"].T
    xx_test, tt_test = np.meshgrid(x_test, t_test)
    points = np.vstack((np.ravel(xx_test), np.ravel(tt_test))).T
    values = usol_test.flatten()
    u_test = griddata(points, values, test_points)
    u_test = np.nan_to_num(u_test)
    
    # 创建增强的RAR采样器
    enhanced_rar_sampler = EnhancedDynamicRarDSampler(
        model=model, 
        geom=geomtime, 
        data=data, 
        pde=pde,
        bc_list=[bc],
        ic_list=[ic],
        tol=1e-3, 
        num_new_pde_points=60,
        num_new_bc_points=20,
        num_new_ic_points=20
    )
    
    # 创建回调函数
    combined_callback = L2ErrorAndRarDCallback(
        test_points, 
        u_test, 
        rar_sampler=enhanced_rar_sampler,
        every=500,  # 每500步计算一次L2误差
        sampling_every=1000
    )
    
    # 🔧 第一阶段：初始训练（500步）
    print("Starting initial training phase (500 iterations)...")
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    losshistory, train_state = model.train(iterations=500, display_every=500, callbacks=[combined_callback])

    # 获取初始损失
    loss_pde = train_state.loss_train[0]
    loss_abc = train_state.loss_train[1]
    loss_ini = train_state.loss_train[2]
    
    print(f"Initial losses after 500 steps: PDE={loss_pde:.6e}, BC={loss_abc:.6e}, IC={loss_ini:.6e}")
    
    # 🚀 使用修复的动态更新器，参数适应500步更新策略
    fixed_dynamic_updater = FixedTrulyDynamicGradientLbPINNsUpdater(
        gradient_extractor=gradient_extractor,
        beta=0.8,           # 稳定的平滑参数
        gamma=0.7,          # 稳定的权重更新率
        init_pde=float(loss_pde),
        init_abc=float(loss_abc),
        init_ini=float(loss_ini),
        init_w_pde=w_pde,
        init_w_abc=w_abc,
        init_w_ini=w_ini,
        learning_rate=0.015,  # 稳定的学习率
        gradient_weight=1.0  # 适中的梯度权重
    )
    
    # 训练循环：每500步更新一次权重，总共3000步ADAM，最后不更新权重
    total_adam_steps = 3000  # 总ADAM训练步数
    update_interval = 500    # 每500步更新一次权重
    stop_update_at = 3000    # 在3000步时停止权重更新
    
    current_step = 500  # 已经训练了500步
    update_count = 0    # 权重更新次数
    
    print(f"\nStarting dynamic weight training phase...")
    print(f"Will train for {total_adam_steps - current_step} more ADAM steps")
    print(f"Weight update every {update_interval} steps, stop updating at step {stop_update_at}")
    
    while current_step < total_adam_steps:
        # 计算这一轮要训练的步数
        steps_this_round = min(update_interval, total_adam_steps - current_step)
        
        print(f"\nTraining step {current_step} to {current_step + steps_this_round}...")
        
        model.compile("adam", lr=lr, loss_weights=loss_weights, decay=("step", 1000, 0.9))
        losshistory, train_state = model.train(iterations=steps_this_round, display_every=steps_this_round, callbacks=[combined_callback])
        
        current_step += steps_this_round
        
        # 只在未达到停止更新步数时才更新权重
        if current_step < stop_update_at:
            # 获取当前损失
            loss_pde = train_state.loss_train[0]
            loss_abc = train_state.loss_train[1]
            loss_ini = train_state.loss_train[2]

            # 记录更新前的权重
            old_weights = [fixed_dynamic_updater.current_w_pde, 
                          fixed_dynamic_updater.current_w_abc, 
                          fixed_dynamic_updater.current_w_ini]
            
            # 更新权重
            updated_w_pde, updated_w_abc, updated_w_ini = fixed_dynamic_updater.update(loss_pde, loss_abc, loss_ini)
            update_count += 1
            
            # 显示权重变化
            new_weights = [updated_w_pde.item(), updated_w_abc.item(), updated_w_ini.item()]
            weight_changes = [abs(new - old) for new, old in zip(new_weights, old_weights)]
            
            print(f"  Weight Update #{update_count} Summary:")
            print(f"    Before: PDE={old_weights[0]:.6f}, BC={old_weights[1]:.6f}, IC={old_weights[2]:.6f}")
            print(f"    After:  PDE={new_weights[0]:.6f}, BC={new_weights[1]:.6f}, IC={new_weights[2]:.6f}")
            print(f"    Changes: PDE={weight_changes[0]:.6f}, BC={weight_changes[1]:.6f}, IC={weight_changes[2]:.6f}")
            print(f"    Total Change: {sum(weight_changes):.8f}")
            
            # 更新loss_weights为下一轮训练使用
            loss_weights = [updated_w_pde.item(), updated_w_abc.item(), updated_w_ini.item()]
            
        else:
            print(f"  Step {current_step}: Weight updates stopped, keeping current weights stable")
            print(f"  Current weights: PDE={loss_weights[0]:.6f}, BC={loss_weights[1]:.6f}, IC={loss_weights[2]:.6f}")
        
        # 每2次更新显示权重变化统计
        if update_count > 0 and update_count % 2 == 0:
            change_stats = fixed_dynamic_updater.get_weight_change_statistics()
            if change_stats:
                print(f"  Weight Change Statistics (last 10 updates):")
                print(f"    Average Total Change: {change_stats['avg_total_weight_change']:.8f}")
                print(f"    Max Total Change: {change_stats['max_total_weight_change']:.8f}")
                print(f"    Gradient Success Rate: {change_stats['gradient_success_rate']:.2%}")

    # 第三阶段：L-BFGS最终优化（使用最后稳定的权重）
    print(f"\nStarting L-BFGS optimization phase...")
    print(f"Total weight updates performed: {update_count}")
    dde.optimizers.set_LBFGS_options(maxiter=3000)
    
    # 使用最后的稳定权重
    final_loss_weights = loss_weights
    print(f"Final stable weights for L-BFGS: PDE={final_loss_weights[0]:.6f}, BC={final_loss_weights[1]:.6f}, IC={final_loss_weights[2]:.6f}")
    
    model.compile("L-BFGS", loss_weights=final_loss_weights)
    losshistory, train_state = model.train(callbacks=[combined_callback])

    return model, losshistory, train_state, updated_w_pde, updated_w_abc, updated_w_ini, fixed_dynamic_updater

# 使用修复的训练函数
print("Starting training with fixed dynamic updater...")
model, losshistory, train_state, updated_w_pde, updated_w_abc, updated_w_ini, fixed_dynamic_updater = train_and_evaluate_with_fixed_gradient_enhanced_lbpinns()

#-------------------------------------------------------------------------------#
# Caculate L2 relative error and plot
#-------------------------------------------------------------------------------#
# 绘制训练损失历史
dde.utils.external.plot_loss_history(losshistory, fname='loss_history.png')

X, y_true = gen_testdata()
y_pred = model.predict(X)

# 4. 计算相对 L2 误差
l2_error = dde.metrics.l2_relative_error(y_true, y_pred)

print("L2 relative error:", l2_error)

# Modified plotting code with larger titles and legends
X, y_true = gen_testdata()
y_pred = model.predict(X)

# Get original grid data from test data
data = np.load("Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T
xx, tt = np.meshgrid(x, t)

# Reshape predictions and true values to grid shape
y_pred_grid = y_pred.reshape(xx.shape)
y_true_grid = y_true.reshape(xx.shape)
error_grid = np.abs(y_pred_grid - y_true_grid)

# Create a figure with three subplots
plt.figure(figsize=(20, 7))

# Set font sizes
title_size = 18
label_size = 16
tick_size = 14
cbar_title_size = 14

# Plot predicted solution
plt.subplot(131)
c1 = plt.contourf(xx, tt, y_pred_grid, levels=50, cmap='viridis')
cbar1 = plt.colorbar(c1)
cbar1.set_label('Magnitude', fontsize=cbar_title_size)
cbar1.ax.tick_params(labelsize=tick_size)
plt.title('Predicted Solution', fontsize=title_size, fontweight='bold')
plt.xlabel('x', fontsize=label_size)
plt.ylabel('t', fontsize=label_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

# Plot exact solution
plt.subplot(132)
c2 = plt.contourf(xx, tt, y_true_grid, levels=50, cmap='viridis')
cbar2 = plt.colorbar(c2)
cbar2.set_label('Magnitude', fontsize=cbar_title_size)
cbar2.ax.tick_params(labelsize=tick_size)
plt.title('Exact Solution', fontsize=title_size, fontweight='bold')
plt.xlabel('x', fontsize=label_size)
plt.ylabel('t', fontsize=label_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

# Plot absolute error
plt.subplot(133)
c3 = plt.contourf(xx, tt, error_grid, levels=50, cmap='viridis')
cbar3 = plt.colorbar(c3)
cbar3.set_label('Error Magnitude', fontsize=cbar_title_size)
cbar3.ax.tick_params(labelsize=tick_size)
plt.title('Absolute Error', fontsize=title_size, fontweight='bold')
plt.xlabel('x', fontsize=label_size)
plt.ylabel('t', fontsize=label_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

# Add a super title with L2 error information
plt.suptitle(f'Burgers Equation Solutions (L2 Relative Error: {l2_error:.4e})', 
             fontsize=20, fontweight='bold', y=0.98)

# Adjust layout and save as PNG
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('burgers_solution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("L2 relative error:", l2_error)
print("Plot saved as 'burgers_solution_comparison.png'")