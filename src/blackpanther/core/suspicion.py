"""Suspicion Field Model

Implements the partial differential equation:
∂S/∂t = D∇²S + rS(1 - S) - δKA + σξ(t)

This 2D reaction-diffusion equation models how defender awareness
spreads through the network like heat.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .base import DifferentialEquation, EquationState


@dataclass
class SuspicionState(EquationState):
    """Specialized state for suspicion field
    
    Attributes:
        field: 2D numpy array of suspicion values
        mean_suspicion: Average suspicion across field
        max_suspicion: Maximum suspicion value
        hotspots: List of (x, y, value) for S > 0.7
        gradient_x: X-component of gradient
        gradient_y: Y-component of gradient
    """
    field: np.ndarray
    mean_suspicion: float
    max_suspicion: float
    hotspots: List[Tuple[int, int, float]]
    gradient_x: np.ndarray
    gradient_y: np.ndarray


class SuspicionField(DifferentialEquation):
    """2D reaction-diffusion model for defender suspicion
    
    The field evolves according to:
    ∂S/∂t = D∇²S + rS(1 - S) - δKA + σξ
    
    Where:
    - Diffusion (D∇²S): Suspicion spreads to neighbors
    - Reaction (rS(1-S)): Defender vigilance grows logistically
    - Suppression (-δKA): Attacks can reduce suspicion if successful
    - Noise (σξ): Random events (false alarms, missed detections)
    
    Example:
        >>> field = SuspicionField(width=50, height=50, D=0.1, r=0.05)
        >>> state = field.reset()
        >>> for step in range(100):
        ...     attacks = [(0.5, 0.5, 0.8)]  # Attack at center
        ...     state = field.step(attacks, knowledge=0.7, access=0.3)
        ...     print(f"Mean suspicion: {state.mean_suspicion:.3f}")
    """
    
    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        D: float = 0.1,
        r: float = 0.05,
        delta: float = 0.01,
        dt: float = 0.1,
        noise_scale: float = 0.001
    ):
        """Initialize suspicion field
        
        Args:
            width: Grid width in cells
            height: Grid height in cells
            D: Diffusion coefficient [0, 1]
            r: Reaction rate [0, 1]
            delta: Detection coefficient [0, 1]
            dt: Time step
            noise_scale: Noise amplitude
        """
        self.width = width
        self.height = height
        self.D = D
        self.r = r
        self.delta = delta
        self.dx = 1.0  # Grid spacing
        
        self._field = np.zeros((height, width))
        self._gradient_x = np.zeros((height, width))
        self._gradient_y = np.zeros((height, width))
        
        super().__init__(dt, noise_scale)
    
    def _validate_parameters(self) -> None:
        """Validate model parameters"""
        if not 0 <= self.D <= 1:
            raise ValueError(f"D must be in [0,1], got {self.D}")
        if not 0 <= self.r <= 1:
            raise ValueError(f"r must be in [0,1], got {self.r}")
        if not 0 <= self.delta <= 1:
            raise ValueError(f"delta must be in [0,1], got {self.delta}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive")
    
    def reset(self) -> SuspicionState:
        """Reset suspicion field to zero
        
        Returns:
            Initial state with zero field
        """
        self._field = np.zeros((self.height, self.width))
        self._gradient_x = np.zeros((self.height, self.width))
        self._gradient_y = np.zeros((self.height, self.width))
        self._history = []
        
        state = self._create_state(timestamp=0.0, episode=0)
        self._history.append(state)
        return state
    
    def _laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """Calculate 2D Laplacian using finite differences
        
        ∇²S = ∂²S/∂x² + ∂²S/∂y²
        
        Discrete approximation:
        ∇²S[i,j] = S[i+1,j] + S[i-1,j] + S[i,j+1] + S[i,j-1] - 4S[i,j]
        
        Args:
            field: 2D array to compute Laplacian for
            
        Returns:
            Laplacian of the field
        """
        laplacian = np.zeros_like(field)
        
        # Interior points using central difference
        laplacian[1:-1, 1:-1] = (
            field[1:-1, 2:] +    # right
            field[1:-1, :-2] +   # left
            field[2:, 1:-1] +    # down
            field[:-2, 1:-1] -   # up
            4 * field[1:-1, 1:-1]
        ) / (self.dx ** 2)
        
        # Neumann boundaries (zero gradient - no flow)
        # This keeps suspicion contained
        return laplacian
    
    def step(
        self,
        attack_positions: List[Tuple[float, float, float]],
        knowledge: float,
        access: float,
        episode: Optional[int] = None
    ) -> SuspicionState:
        """Evolve suspicion field one time step
        
        Args:
            attack_positions: List of (x, y, intensity) in normalized [0,1]
            knowledge: Current knowledge level K
            access: Current access level A
            episode: Episode number
            
        Returns:
            Updated suspicion state
        """
        # Term 1: Diffusion - suspicion spreads
        laplacian = self._laplacian_2d(self._field)
        diffusion = self.D * laplacian
        
        # Term 2: Reaction - defender vigilance grows logistically
        reaction = self.r * self._field * (1 - self._field)
        
        # Term 3: Attack suppression - covering tracks
        suppression = np.zeros_like(self._field)
        for x, y, intensity in attack_positions:
            ix = int(x * (self.width - 1))
            iy = int(y * (self.height - 1))
            
            if 0 <= ix < self.width and 0 <= iy < self.height:
                # Suppression requires both knowledge AND access
                suppression[iy, ix] += intensity * self.delta * knowledge * access
        
        # Term 4: Spatiotemporal noise
        noise = self.noise_scale * np.random.randn(self.height, self.width) * np.sqrt(self.dt)
        
        # Total change
        dS = (diffusion + reaction - suppression) * self.dt + noise
        self._field += dS
        
        # Physical constraints: suspicion between 0 and 1
        self._field = np.clip(self._field, 0.0, 1.0)
        
        # Update gradients
        self._gradient_x, self._gradient_y = np.gradient(self._field)
        
        # Create state
        episode_num = episode if episode is not None else len(self._history)
        state = self._create_state(
            timestamp=len(self._history) * self.dt,
            episode=episode_num,
            diffusion=diffusion,
            reaction=reaction,
            suppression=suppression
        )
        
        self._history.append(state)
        return state
    
    def _create_state(
        self,
        timestamp: float,
        episode: int,
        diffusion: Optional[np.ndarray] = None,
        reaction: Optional[np.ndarray] = None,
        suppression: Optional[np.ndarray] = None
    ) -> SuspicionState:
        """Create state object from current field"""
        hotspots = []
        y_coords, x_coords = np.where(self._field > 0.7)
        for y, x in zip(y_coords[:10], x_coords[:10]):  # Limit to 10
            hotspots.append((int(x), int(y), float(self._field[y, x])))
        
        return SuspicionState(
            field=self._field.copy(),
            mean_suspicion=float(np.mean(self._field)),
            max_suspicion=float(np.max(self._field)),
            hotspots=hotspots,
            gradient_x=self._gradient_x.copy(),
            gradient_y=self._gradient_y.copy(),
            timestamp=timestamp,
            episode=episode,
            values={'mean_suspicion': float(np.mean(self._field))},
            metadata={
                'diffusion': float(np.mean(np.abs(diffusion))) if diffusion is not None else 0,
                'reaction': float(np.mean(reaction)) if reaction is not None else 0,
                'suppression': float(np.mean(suppression)) if suppression is not None else 0
            }
        )
    
    @property
    def field(self) -> np.ndarray:
        """Current suspicion field"""
        return self._field.copy()
    
    def get_suspicion_at(self, x: float, y: float) -> float:
        """Get suspicion value at normalized coordinates
        
        Args:
            x: X-coordinate in [0, 1]
            y: Y-coordinate in [0, 1]
            
        Returns:
            Suspicion value at that point
        """
        ix = int(x * (self.width - 1))
        iy = int(y * (self.height - 1))
        ix = max(0, min(self.width - 1, ix))
        iy = max(0, min(self.height - 1, iy))
        return float(self._field[iy, ix])