"""Knowledge Evolution Model

Implements the differential equation:
dK/dt = αK(1 - K/K_max) - βK + γS + σξ(t)

This models how the attacker's knowledge grows through learning,
decays through forgetting, and benefits from defender reactions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .base import DifferentialEquation, EquationState


@dataclass
class KnowledgeState(EquationState):
    """Specialized state for knowledge evolution
    
    Attributes:
        knowledge: Current knowledge level K
        suspicion: Current suspicion level S (input)
        learning_action: Learning effort from RL agent
        gradient: Rate of change dK/dt
    """
    knowledge: float
    suspicion: float
    learning_action: float
    gradient: float


class KnowledgeEvolution(DifferentialEquation):
    """Knowledge evolution model for attacker learning
    
    The model captures:
    - Logistic growth from learning (αK(1-K/K_max))
    - Linear decay from forgetting (-βK)
    - Learning from defender reactions (+γS)
    - Stochastic effects (σξ)
    
    The learning rate α can be modulated by the RL agent's learning_action.
    
    Example:
        >>> model = KnowledgeEvolution(alpha=0.1, beta=0.01, gamma=0.05)
        >>> state = model.reset()
        >>> for episode in range(100):
        ...     state = model.step(suspicion=0.3, learning_action=0.5)
        ...     print(f"Knowledge: {state.knowledge:.2f}")
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 0.05,
        k_max: float = 100.0,
        dt: float = 0.1,
        noise_scale: float = 0.01
    ):
        """Initialize knowledge evolution model
        
        Args:
            alpha: Base learning rate [0, 2]
            beta: Forgetting rate [0, 1]
            gamma: Suspicion conversion rate [0, 1]
            k_max: Maximum knowledge capacity
            dt: Time step for integration
            noise_scale: Noise amplitude
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_max = k_max
        self._knowledge = 0.0
        super().__init__(dt, noise_scale)
    
    def _validate_parameters(self) -> None:
        """Validate model parameters"""
        if not 0 <= self.alpha <= 2:
            raise ValueError(f"alpha must be in [0,2], got {self.alpha}")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta must be in [0,1], got {self.beta}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0,1], got {self.gamma}")
        if self.k_max <= 0:
            raise ValueError(f"k_max must be positive, got {self.k_max}")
    
    def reset(self, initial_knowledge: float = 0.0) -> KnowledgeState:
        """Reset knowledge to initial value
        
        Args:
            initial_knowledge: Starting knowledge level
            
        Returns:
            Initial state
        """
        self._knowledge = initial_knowledge
        self._history = []
        
        state = KnowledgeState(
            knowledge=self._knowledge,
            suspicion=0.0,
            learning_action=0.0,
            gradient=0.0,
            timestamp=0.0,
            episode=0,
            values={'knowledge': self._knowledge}
        )
        self._history.append(state)
        return state
    
    def step(
        self,
        suspicion: float,
        learning_action: float = 0.5,
        episode: Optional[int] = None
    ) -> KnowledgeState:
        """Evolve knowledge one time step
        
        The evolution follows:
        dK/dt = αK(1-K/K_max) - βK + γS + σξ
        
        Args:
            suspicion: Current suspicion level S(t)
            learning_action: Agent's learning effort (modulates α)
            episode: Current episode number
            
        Returns:
            New knowledge state
        """
        # Term 1: Logistic growth with modulated learning
        effective_alpha = self.alpha * (1.0 + learning_action)
        saturation = 1.0 - self._knowledge / self.k_max
        growth = effective_alpha * self._knowledge * saturation
        
        # Term 2: Forgetting
        forgetting = self.beta * self._knowledge
        
        # Term 3: Learning from defense
        from_defense = self.gamma * suspicion
        
        # Term 4: Stochastic noise
        noise = self._wiener_noise()
        
        # Total change
        dK = (growth - forgetting + from_defense) * self.dt + noise
        self._knowledge = max(0.0, min(self.k_max, self._knowledge + dK))
        
        # Calculate gradient for analysis
        gradient = dK / self.dt if self.dt > 0 else 0
        
        # Create state
        episode_num = episode if episode is not None else len(self._history)
        state = KnowledgeState(
            knowledge=self._knowledge,
            suspicion=suspicion,
            learning_action=learning_action,
            gradient=gradient,
            timestamp=len(self._history) * self.dt,
            episode=episode_num,
            values={'knowledge': self._knowledge},
            metadata={
                'growth': float(growth),
                'forgetting': float(forgetting),
                'from_defense': float(from_defense),
                'noise': float(noise)
            }
        )
        
        self._history.append(state)
        return state
    
    @property
    def knowledge(self) -> float:
        """Current knowledge level"""
        return self._knowledge