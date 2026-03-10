"""Base abstractions for mathematical models

This module defines the foundation for all differential equation models
used in BlackPanther. Following the Open/Closed principle, these base
classes can be extended without modification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class EquationState:
    """Immutable state container for equation solvers
    
    Attributes:
        timestamp: Simulation time
        episode: Training episode number
        values: Dictionary of state variables
        metadata: Additional information about the state
    """
    timestamp: float
    episode: int
    values: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DifferentialEquation(ABC):
    """Abstract base for all differential equation models
    
    This class defines the interface that all mathematical models must implement.
    It handles common functionality like state history and noise generation.
    
    Example:
        class KnowledgeEvolution(DifferentialEquation):
            def step(self, *args, **kwargs):
                # Implementation here
                pass
    """
    
    def __init__(self, dt: float = 0.1, noise_scale: float = 0.01):
        """Initialize the equation solver
        
        Args:
            dt: Time step for numerical integration
            noise_scale: Standard deviation of Wiener process noise
        """
        self.dt = dt
        self.noise_scale = noise_scale
        self._history: List[EquationState] = []
        self._validate_parameters()
    
    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate model parameters
        
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def step(self, **kwargs) -> EquationState:
        """Evolve the system one time step
        
        Args:
            **kwargs: Model-specific inputs
            
        Returns:
            New state after evolution
        """
        pass
    
    @abstractmethod
    def reset(self) -> EquationState:
        """Reset the model to initial state
        
        Returns:
            Initial state
        """
        pass
    
    def _wiener_noise(self) -> float:
        """Generate Wiener process noise
        
        Returns:
            Random value scaled by √dt
        """
        return self.noise_scale * np.random.randn() * np.sqrt(self.dt)
    
    @property
    def history(self) -> List[EquationState]:
        """Get immutable copy of history"""
        return self._history.copy()
    
    def save_state(self, path: str) -> None:
        """Persist model state to disk
        
        Args:
            path: File path for saving
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'history': self._history,
                'params': {k: v for k, v in self.__dict__.items() 
                          if not k.startswith('_')}
            }, f)