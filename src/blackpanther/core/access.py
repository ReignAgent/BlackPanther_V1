"""Access Propagation Model

Implements the differential equation:
dA/dt = ηKA(1 - A) - μA + Σ w_ji A_j(1 - A_i)

This models how attacker access spreads through a network like an epidemic,
with lateral movement between connected hosts.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .base import DifferentialEquation, EquationState


@dataclass
class HostAccess:
    """Access state for a single host
    
    Attributes:
        host_id: Unique identifier
        access: Current access level [0, 1]
        compromised: Whether access > 0.5
        privileges: List of obtained privileges
        services: Vulnerable services on this host
    """
    host_id: str
    access: float
    compromised: bool
    privileges: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)


@dataclass
class AccessState(EquationState):
    """Global access state for the network
    
    Attributes:
        global_access: Maximum access across all hosts
        compromised_hosts: List of host IDs with access > 0.5
        host_accesses: Dictionary mapping host_id to access level
        access_gradient: Rate of change of global access
        lateral_rate: Current lateral movement rate
    """
    global_access: float
    compromised_hosts: List[str]
    host_accesses: Dict[str, float]
    access_gradient: float
    lateral_rate: float


class AccessPropagation(DifferentialEquation):
    """Networked epidemic model for access propagation
    
    The model captures:
    - Within-host growth: ηKA(1 - A) (privilege escalation)
    - Decay: -μA (defender cleanup)
    - Between-host spread: Σ w_ji A_j(1 - A_i) (lateral movement)
    
    Example:
        >>> model = AccessPropagation(eta=0.2, mu=0.01)
        >>> model.add_host("web", initial_access=0.3)
        >>> model.add_host("db")
        >>> G = nx.Graph()
        >>> G.add_edge("web", "db", weight=0.8)
        >>> model.set_network(G)
        >>> for step in range(100):
        ...     state = model.step(knowledge=0.7)
        ...     print(f"Compromised: {state.compromised_hosts}")
    """
    
    def __init__(
        self,
        eta: float = 0.2,
        mu: float = 0.01,
        dt: float = 0.1,
        noise_scale: float = 0.001
    ):
        """Initialize access propagation model
        
        Args:
            eta: Propagation coefficient [0, 1]
            mu: Decay rate [0, 1]
            dt: Time step
            noise_scale: Noise amplitude
        """
        self.eta = eta
        self.mu = mu
        self._hosts: Dict[str, HostAccess] = {}
        self._network: Optional[nx.Graph] = None
        self._global_access = 0.0
        
        super().__init__(dt, noise_scale)
    
    def _validate_parameters(self) -> None:
        """Validate model parameters"""
        if not 0 <= self.eta <= 1:
            raise ValueError(f"eta must be in [0,1], got {self.eta}")
        if not 0 <= self.mu <= 1:
            raise ValueError(f"mu must be in [0,1], got {self.mu}")
    
    def reset(self) -> AccessState:
        """Reset all access levels to zero
        
        Returns:
            Initial state with no access
        """
        self._hosts = {}
        self._global_access = 0.0
        self._history = []
        
        state = self._create_state(timestamp=0.0, episode=0)
        self._history.append(state)
        return state
    
    def add_host(
        self,
        host_id: str,
        initial_access: float = 0.0,
        services: Optional[List[str]] = None
    ) -> None:
        """Add a host to track
        
        Args:
            host_id: Unique identifier
            initial_access: Starting access level
            services: List of vulnerable services
        """
        self._hosts[host_id] = HostAccess(
            host_id=host_id,
            access=initial_access,
            compromised=initial_access > 0.5,
            services=services or []
        )
    
    def set_network(self, network: nx.Graph) -> None:
        """Set network topology
        
        Edge attributes can include:
        - weight: Connection strength [0, 1]
        - vulnerability: How exploitable the connection is [0, 1]
        
        Args:
            network: NetworkX graph with host_id nodes
        """
        self._network = network
        
        # Add any missing hosts from network
        for node in network.nodes():
            if node not in self._hosts:
                self.add_host(node)
    
    def step(
        self,
        knowledge: float,
        target_host: Optional[str] = None,
        attack_intensity: float = 1.0,
        episode: Optional[int] = None
    ) -> AccessState:
        """Evolve access for all hosts
        
        The evolution follows:
        dA_i/dt = ηKA_i(1 - A_i) - μA_i + Σ w_ji A_j(1 - A_i)
        
        Args:
            knowledge: Current knowledge level K
            target_host: Specific host to update (None for all)
            attack_intensity: Attack effort multiplier
            episode: Episode number
            
        Returns:
            Updated access state
        """
        if not self._hosts:
            raise RuntimeError("No hosts added. Call add_host() first.")
        
        hosts_to_update = [target_host] if target_host else self._hosts.keys()
        lateral_total = 0.0
        
        for host_id in hosts_to_update:
            if host_id not in self._hosts:
                continue
            
            host = self._hosts[host_id]
            
            # Term 1: Growth within host (privilege escalation)
            # ηKA(1 - A)
            growth = self.eta * knowledge * host.access * (1 - host.access)
            effective_growth = growth * attack_intensity
            
            # Term 2: Decay (defender cleanup)
            # -μA
            decay = self.mu * host.access
            
            # Term 3: Lateral movement from other hosts
            # Σ w_ji A_j(1 - A_i)
            lateral = 0.0
            if self._network:
                for source_id in self._hosts:
                    if source_id != host_id and self._network.has_edge(source_id, host_id):
                        source = self._hosts[source_id]
                        
                        # Get edge attributes with defaults
                        edge_data = self._network.get_edge_data(source_id, host_id)
                        weight = edge_data.get('weight', 0.5) if edge_data else 0.5
                        vuln = edge_data.get('vulnerability', 0.3) if edge_data else 0.3
                        
                        # Lateral movement contribution
                        lateral += weight * vuln * source.access * (1 - host.access)
            
            lateral_total += lateral
            
            # Noise
            noise = self._wiener_noise()
            
            # Total change
            dA = (effective_growth - decay + lateral) * self.dt + noise
            new_access = max(0.0, min(1.0, host.access + dA))
            
            # Update host
            was_compromised = host.compromised
            host.access = new_access
            host.compromised = new_access > 0.5
            
            # Log new compromises
            if host.compromised and not was_compromised:
                print(f"Host {host_id} COMPROMISED (access={new_access:.3f})")
        
        # Update global access (max across hosts)
        self._global_access = max(h.access for h in self._hosts.values())
        
        # Create state
        episode_num = episode if episode is not None else len(self._history)
        state = self._create_state(
            timestamp=len(self._history) * self.dt,
            episode=episode_num,
            lateral_total=lateral_total
        )
        
        self._history.append(state)
        return state
    
    def _create_state(
        self,
        timestamp: float,
        episode: int,
        lateral_total: float = 0.0
    ) -> AccessState:
        """Create state object from current data"""
        compromised = [
            host_id for host_id, host in self._hosts.items()
            if host.compromised
        ]
        
        host_accesses = {
            host_id: host.access for host_id, host in self._hosts.items()
        }
        
        # Calculate gradient (finite difference)
        if len(self._history) > 1:
            prev = self._history[-1]
            gradient = (self._global_access - prev.global_access) / self.dt
        else:
            gradient = 0.0
        
        return AccessState(
            global_access=self._global_access,
            compromised_hosts=compromised,
            host_accesses=host_accesses,
            access_gradient=gradient,
            lateral_rate=lateral_total,
            timestamp=timestamp,
            episode=episode,
            values={'global_access': self._global_access},
            metadata={
                'total_hosts': len(self._hosts),
                'compromised_count': len(compromised)
            }
        )
    
    @property
    def hosts(self) -> Dict[str, HostAccess]:
        """Get all hosts"""
        return self._hosts.copy()
    
    @property
    def global_access(self) -> float:
        """Current global access level"""
        return self._global_access
    
    def get_compromised_hosts(self) -> List[str]:
        """Get list of compromised host IDs"""
        return [hid for hid, h in self._hosts.items() if h.compromised]