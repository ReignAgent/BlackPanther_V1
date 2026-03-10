#!/usr/bin/env python3
"""Demonstration of all mathematical models working together

This script shows how the four core mathematical models
interact in a simulated cyber attack scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.core.access import AccessPropagation
from blackpanther.core.control import HJBController


def run_demo():
    """Run complete mathematical models demo"""
    print("=" * 60)
    print("BlackPanther V2 - Mathematical Models Demo")
    print("=" * 60)
    
    # Initialize models
    print("\n[1/5] Initializing models...")
    knowledge = KnowledgeEvolution(alpha=0.15, beta=0.02, gamma=0.08, k_max=100)
    suspicion = SuspicionField(width=30, height=30, D=0.15, r=0.08, delta=0.02)
    access = AccessPropagation(eta=0.25, mu=0.015)
    controller = HJBController(grid_points=15)
    
    # Setup network
    print("[2/5] Setting up network topology...")
    import networkx as nx
    G = nx.cycle_graph(5)  # 5 hosts in a ring
    for i, node in enumerate(G.nodes()):
        access.add_host(f"host_{node}", 
                       initial_access=0.2 if i == 0 else 0.0,
                       services=["http", "ssh"] if i == 0 else [])
    access.set_network(G)
    
    # Run simulation
    print("[3/5] Running 200-step simulation...")
    attack_positions = [(0.3, 0.3, 0.7), (0.7, 0.7, 0.5)]
    
    for step in range(200):
        # Get current states
        K = knowledge.knowledge
        S = suspicion.field.mean()
        A = access.global_access
        
        # Get optimal control (simplified for demo)
        attack_intensity = 0.5 + 0.3 * np.sin(step / 20)
        stealth = 0.5 + 0.2 * np.cos(step / 15)
        
        # Update models
        knowledge.step(suspicion=S, learning_action=attack_intensity)
        suspicion.step(attack_positions, knowledge=K, access=A)
        access.step(knowledge=K, attack_intensity=attack_intensity)
    
    # Show results
    print("[4/5] Simulation complete!")
    print("\n" + "=" * 40)
    print("FINAL STATE")
    print("=" * 40)
    print(f"Knowledge: {knowledge.knowledge:.2f}/{knowledge.k_max}")
    print(f"Suspicion (mean): {suspicion.field.mean():.3f}")
    print(f"Suspicion (max): {suspicion.field.max():.3f}")
    print(f"Global Access: {access.global_access:.3f}")
    print(f"Compromised Hosts: {access.get_compromised_hosts()}")
    
    # Create visualizations
    print("\n[5/5] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Knowledge evolution
    ax = axes[0, 0]
    times = [h.timestamp for h in knowledge.history]
    values = [h.knowledge for h in knowledge.history]
    ax.plot(times, values, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Knowledge')
    ax.set_title('Knowledge Evolution')
    ax.grid(True, alpha=0.3)
    
    # Suspicion field
    ax = axes[0, 1]
    im = ax.imshow(suspicion.field, cmap='hot', vmin=0, vmax=1)
    ax.set_title('Suspicion Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Access propagation
    ax = axes[1, 0]
    hosts = list(access.hosts.keys())
    accesses = [access.hosts[h].access for h in hosts]
    colors = ['red' if a > 0.5 else 'green' for a in accesses]
    ax.bar(hosts, accesses, color=colors)
    ax.set_xlabel('Host')
    ax.set_ylabel('Access Level')
    ax.set_title('Access per Host')
    ax.set_ylim(0, 1)
    
    # State summary
    ax = axes[1, 1]
    ax.axis('off')
    stats = f"""State Summary

Knowledge: {knowledge.knowledge:.1f}
Suspicion: {suspicion.field.mean():.2f}
Access: {access.global_access:.2f}

Compromised: {len(access.get_compromised_hosts())}/{len(access.hosts)}

Learning Rate α: {knowledge.alpha}
Forgetting β: {knowledge.beta}
Propagation η: {access.eta}
Decay μ: {access.mu}
"""
    ax.text(0.1, 0.5, stats, transform=ax.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "mathematical_demo.png", dpi=150)
    print(f"Plot saved to output/mathematical_demo.png")
    
    # Show plot (if interactive)
    plt.show()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()