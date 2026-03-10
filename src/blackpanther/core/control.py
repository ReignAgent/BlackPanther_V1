"""Hamilton-Jacobi-Bellman Optimal Controller

Implements the HJB equation:
0 = min_u { r(x,u) + ∇V·f(x,u) + ½Tr(σ²∇²V) }

This finds the optimal policy balancing attack intensity and stealth.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable
from scipy.optimize import minimize

from .base import DifferentialEquation, EquationState


@dataclass
class SystemState:
    """Full system state for HJB optimization
    
    Attributes:
        knowledge: Knowledge level K
        suspicion: Suspicion level S
        access: Access level A
        time: Current time
        episode: Episode number
    """
    knowledge: float
    suspicion: float
    access: float
    time: float
    episode: int


@dataclass
class Control:
    """Control action from the agent
    
    Attributes:
        attack_intensity: u1 ∈ [0,1] - how aggressively to attack
        stealth: u2 ∈ [0,1] - how carefully to hide
    """
    attack_intensity: float
    stealth: float


class HJBController:
    """Hamilton-Jacobi-Bellman optimal controller
    
    Solves the optimal control problem:
    0 = min_u [ r(x,u) + ∇V·f(x,u) + ½Tr(σ²∇²V) ]
    
    Where:
    - x = [K, S, A] is the system state
    - u = [u1, u2] is the control (attack intensity, stealth)
    - r(x,u) is the running cost
    - f(x,u) is the system dynamics
    - V(x) is the value function
    
    Example:
        >>> controller = HJBController(grid_points=20)
        >>> controller.solve(max_iterations=100)
        >>> optimal = controller.get_optimal_action(
        ...     knowledge=50.0, suspicion=0.3, access=0.4
        ... )
        >>> print(f"Attack: {optimal.attack_intensity:.2f}")
    """
    
    def __init__(
        self,
        grid_points: int = 20,
        gamma: float = 0.95,
        dt: float = 0.1,
        noise_scale: float = 0.01
    ):
        """Initialize HJB controller
        
        Args:
            grid_points: Points per dimension for value function grid
            gamma: Discount factor
            dt: Time step
            noise_scale: Noise amplitude
        """
        self.grid_points = grid_points
        self.gamma = gamma
        self.dt = dt
        self.noise_scale = noise_scale
        
        # State space bounds
        self.k_range = (0.0, 100.0)  # Knowledge
        self.s_range = (0.0, 1.0)     # Suspicion
        self.a_range = (0.0, 1.0)     # Access
        
        # Discretize state space
        self.k_grid = np.linspace(*self.k_range, grid_points)
        self.s_grid = np.linspace(*self.s_range, grid_points)
        self.a_grid = np.linspace(*self.a_range, grid_points)
        
        # Value function grid
        self.value_grid = np.zeros((grid_points, grid_points, grid_points))
        
        # Optimal policy lookup table
        self.policy_table: Dict[Tuple[float, float, float], Control] = {}
        
        # Cost weights
        self.R_A = 10.0   # Access reward
        self.R_K = 5.0    # Knowledge reward
        self.P_S = 20.0   # Suspicion penalty
        self.C1 = 2.0     # Attack cost
        self.C2 = 5.0     # Stealth cost
    
    def running_cost(self, state: Tuple[float, float, float], control: Control) -> float:
        """Calculate running cost r(x,u)
        
        r(x,u) = -R_A·A - R_K·K + P_S·S + C1·u1² + C2·u2²
        
        Args:
            state: Tuple (K, S, A)
            control: Current control action
            
        Returns:
            Cost value (negative = reward)
        """
        K, S, A = state
        
        # Rewards (negative cost)
        access_reward = -self.R_A * A
        knowledge_reward = -self.R_K * K
        
        # Penalties (positive cost)
        suspicion_penalty = self.P_S * S
        attack_risk = self.C1 * control.attack_intensity**2
        stealth_overhead = self.C2 * control.stealth**2
        
        return (access_reward + knowledge_reward + 
                suspicion_penalty + attack_risk + stealth_overhead)
    
    def system_dynamics(
        self,
        state: Tuple[float, float, float],
        control: Control
    ) -> Tuple[float, float, float]:
        """Compute system dynamics f(x,u)
        
        Returns (dK/dt, dS/dt, dA/dt) using simplified models
        for HJB optimization.
        
        Args:
            state: Current state (K, S, A)
            control: Current control
            
        Returns:
            Derivatives (dK, dS, dA)
        """
        K, S, A = state
        u1, u2 = control.attack_intensity, control.stealth
        
        # Knowledge dynamics (simplified)
        alpha, beta, gamma = 0.1, 0.01, 0.05
        k_max = 100.0
        dK = (alpha * K * (1 - K/k_max) - beta * K + gamma * S) * u1
        
        # Suspicion dynamics (simplified)
        D, r, delta = 0.1, 0.05, 0.01
        # Approximate Laplacian with constant for HJB
        laplacian_approx = 0.1
        dS = (D * laplacian_approx + r * S * (1 - S) - delta * K * A) * (1 - u2)
        
        # Access dynamics (simplified)
        eta, mu = 0.2, 0.01
        dA = (eta * K * A * (1 - A) - mu * A) * u1
        
        return (dK, dS, dA)
    
    def _gradient_K(self, i: int, j: int, k: int) -> float:
        """∂V/∂K using finite differences"""
        if i == 0:
            return (self.value_grid[i+1, j, k] - self.value_grid[i, j, k]) / (self.k_grid[1] - self.k_grid[0])
        elif i == self.grid_points - 1:
            return (self.value_grid[i, j, k] - self.value_grid[i-1, j, k]) / (self.k_grid[1] - self.k_grid[0])
        else:
            return (self.value_grid[i+1, j, k] - self.value_grid[i-1, j, k]) / (2 * (self.k_grid[1] - self.k_grid[0]))
    
    def _gradient_S(self, i: int, j: int, k: int) -> float:
        """∂V/∂S using finite differences"""
        if j == 0:
            return (self.value_grid[i, j+1, k] - self.value_grid[i, j, k]) / (self.s_grid[1] - self.s_grid[0])
        elif j == self.grid_points - 1:
            return (self.value_grid[i, j, k] - self.value_grid[i, j-1, k]) / (self.s_grid[1] - self.s_grid[0])
        else:
            return (self.value_grid[i, j+1, k] - self.value_grid[i, j-1, k]) / (2 * (self.s_grid[1] - self.s_grid[0]))
    
    def _gradient_A(self, i: int, j: int, k: int) -> float:
        """∂V/∂A using finite differences"""
        if k == 0:
            return (self.value_grid[i, j, k+1] - self.value_grid[i, j, k]) / (self.a_grid[1] - self.a_grid[0])
        elif k == self.grid_points - 1:
            return (self.value_grid[i, j, k] - self.value_grid[i, j, k-1]) / (self.a_grid[1] - self.a_grid[0])
        else:
            return (self.value_grid[i, j, k+1] - self.value_grid[i, j, k-1]) / (2 * (self.a_grid[1] - self.a_grid[0]))
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-3) -> None:
        """Solve HJB equation using value iteration
        
        This approximates the solution to:
        0 = min_u { r(x,u) + ∇V·f(x,u) + ½Tr(σ²∇²V) }
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold
        """
        print("Solving HJB equation...")
        
        for iteration in range(max_iterations):
            value_old = self.value_grid.copy()
            max_diff = 0.0
            
            # Iterate over all states
            for i, K in enumerate(self.k_grid):
                for j, S in enumerate(self.s_grid):
                    for k, A in enumerate(self.a_grid):
                        state = (K, S, A)
                        
                        def objective(u):
                            control = Control(attack_intensity=u[0], stealth=u[1])
                            
                            # Get gradients
                            grad_K = self._gradient_K(i, j, k)
                            grad_S = self._gradient_S(i, j, k)
                            grad_A = self._gradient_A(i, j, k)
                            
                            # Compute dynamics
                            dK, dS, dA = self.system_dynamics(state, control)
                            
                            # Hamiltonian: H = r + ∇V·f
                            cost = self.running_cost(state, control)
                            drift = grad_K * dK + grad_S * dS + grad_A * dA
                            
                            return cost + drift
                        
                        # Minimize over control
                        result = minimize(
                            objective,
                            [0.5, 0.5],
                            bounds=[(0, 1), (0, 1)],
                            method='L-BFGS-B'
                        )
                        
                        if result.success:
                            self.value_grid[i, j, k] = -result.fun
                            self.policy_table[(K, S, A)] = Control(
                                attack_intensity=result.x[0],
                                stealth=result.x[1]
                            )
                            
                            diff = abs(self.value_grid[i, j, k] - value_old[i, j, k])
                            max_diff = max(max_diff, diff)
            
            if max_diff < tolerance:
                print(f"Converged after {iteration} iterations")
                break
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, max diff={max_diff:.6f}")
    
    def get_optimal_action(
        self,
        knowledge: float,
        suspicion: float,
        access: float
    ) -> Control:
        """Get optimal control for given state
        
        Args:
            knowledge: Current knowledge level
            suspicion: Current suspicion level
            access: Current access level
            
        Returns:
            Optimal control action
        """
        # Find nearest grid points
        i = np.argmin(np.abs(self.k_grid - knowledge))
        j = np.argmin(np.abs(self.s_grid - suspicion))
        k = np.argmin(np.abs(self.a_grid - access))
        
        K = self.k_grid[i]
        S = self.s_grid[j]
        A = self.a_grid[k]
        
        # Look up policy
        if (K, S, A) in self.policy_table:
            return self.policy_table[(K, S, A)]
        else:
            # Default: balanced approach
            return Control(attack_intensity=0.5, stealth=0.5)