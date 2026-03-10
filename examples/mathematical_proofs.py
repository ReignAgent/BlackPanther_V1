#!/usr/bin/env python3
"""Mathematical Proofs -- Visual verification of every core model.

Generates publication-quality plots proving the correctness and behavior
of the four BlackPanther differential-equation models:

  1. Knowledge Evolution   dK/dt = αK(1-K/K_max) - βK + γS + σξ
  2. Suspicion Field       ∂S/∂t = D∇²S + rS(1-S) - δKA + σξ
  3. Access Propagation    dA/dt = ηKA(1-A) - μA + Σ w_ji A_j(1-A_i)
  4. HJB Controller        0 = min_u { r(x,u) + ∇V·f(x,u) + ½Tr(σ²∇²V) }

All figures are saved into  output/proofs/  as high-res PNGs.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D proj)
import seaborn as sns
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.core.access import AccessPropagation
from blackpanther.core.control import HJBController, Control

OUT = Path("output/proofs")
OUT.mkdir(parents=True, exist_ok=True)

DPI = 150
sns.set_theme(style="whitegrid", context="notebook", palette="deep")


# =====================================================================
# 1.  KNOWLEDGE EVOLUTION
# =====================================================================

def proof_knowledge_alpha_sweep():
    """Time-series of K(t) for several learning rates α."""
    fig, ax = plt.subplots(figsize=(10, 6))
    alphas = [0.05, 0.10, 0.20, 0.40, 0.80]
    steps = 300

    for alpha in alphas:
        model = KnowledgeEvolution(alpha=alpha, beta=0.02, gamma=0.05,
                                   k_max=100, noise_scale=0.0)
        model.reset(initial_knowledge=1.0)
        for _ in range(steps):
            model.step(suspicion=0.2, learning_action=0.5)
        ts = [h.timestamp for h in model.history]
        ks = [h.knowledge for h in model.history]
        ax.plot(ts, ks, linewidth=2, label=f"α = {alpha}")

    ax.set_xlabel("Time  t")
    ax.set_ylabel("Knowledge  K(t)")
    ax.set_title("Knowledge Evolution — Learning-Rate Sweep\n"
                 r"$dK/dt = \alpha K(1-K/K_{max}) - \beta K + \gamma S$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_knowledge_alpha_sweep.png", dpi=DPI)
    plt.close(fig)
    print("  [1/12] Knowledge α-sweep saved")


def proof_knowledge_phase_portrait():
    """Phase portrait: dK/dt vs K showing equilibria and flow."""
    fig, ax = plt.subplots(figsize=(10, 6))

    K_vals = np.linspace(0, 100, 500)
    alpha, beta, gamma, S, k_max = 0.15, 0.02, 0.05, 0.2, 100.0
    learning_actions = [0.0, 0.5, 1.0, 2.0]

    for la in learning_actions:
        eff_alpha = alpha * (1.0 + la)
        dKdt = eff_alpha * K_vals * (1 - K_vals / k_max) - beta * K_vals + gamma * S
        ax.plot(K_vals, dKdt, linewidth=2, label=f"learning_action = {la}")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Knowledge  K")
    ax.set_ylabel("dK/dt")
    ax.set_title("Phase Portrait — Knowledge Evolution\n"
                 "Equilibria where dK/dt = 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "02_knowledge_phase_portrait.png", dpi=DPI)
    plt.close(fig)
    print("  [2/12] Knowledge phase portrait saved")


def proof_knowledge_sensitivity_heatmap():
    """Heatmap: final K as function of (α, β)."""
    alphas = np.linspace(0.02, 0.5, 30)
    betas = np.linspace(0.001, 0.15, 30)
    Z = np.zeros((len(betas), len(alphas)))
    steps = 200

    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            model = KnowledgeEvolution(alpha=alpha, beta=beta, gamma=0.05,
                                       k_max=100, noise_scale=0.0)
            model.reset(initial_knowledge=1.0)
            for _ in range(steps):
                model.step(suspicion=0.2, learning_action=0.5)
            Z[i, j] = model.knowledge

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
                   cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final Knowledge  K(T)")
    ax.set_xlabel("Learning rate  α")
    ax.set_ylabel("Forgetting rate  β")
    ax.set_title("Parameter Sensitivity — Knowledge at T=20\n"
                 r"Higher $\alpha$ and lower $\beta$ $\Rightarrow$ greater knowledge")
    fig.tight_layout()
    fig.savefig(OUT / "03_knowledge_sensitivity.png", dpi=DPI)
    plt.close(fig)
    print("  [3/12] Knowledge sensitivity heatmap saved")


# =====================================================================
# 2.  SUSPICION FIELD
# =====================================================================

def proof_suspicion_heatmap_snapshots():
    """Heatmap snapshots of S(x,y) at four time points."""
    field = SuspicionField(width=50, height=50, D=0.15, r=0.08,
                           delta=0.02, noise_scale=0.0)
    field.reset()

    field._field[25, 25] = 1.0
    field._field[10, 10] = 0.8
    field._field[40, 35] = 0.6

    snap_at = {0: field.field}
    for step in range(1, 201):
        field.step(attack_positions=[(0.5, 0.5, 0.3)],
                   knowledge=0.5, access=0.2)
        if step in (50, 100, 200):
            snap_at[step] = field.field

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (t, data) in zip(axes, sorted(snap_at.items())):
        im = ax.imshow(data, cmap="inferno", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Suspicion  S")
    fig.suptitle("Suspicion Field Diffusion — Snapshots over Time\n"
                 r"$\partial S/\partial t = D\nabla^2 S + rS(1-S) - \delta KA$",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "04_suspicion_snapshots.png", dpi=DPI)
    plt.close(fig)
    print("  [4/12] Suspicion snapshots saved")


def proof_suspicion_contour_gradient():
    """Contour plot of S with gradient vector field overlaid."""
    field = SuspicionField(width=50, height=50, D=0.15, r=0.08,
                           delta=0.01, noise_scale=0.0)
    field.reset()
    field._field[25, 25] = 1.0
    field._field[10, 40] = 0.7

    for _ in range(80):
        field.step(attack_positions=[], knowledge=0.3, access=0.1)

    S = field.field
    gy, gx = np.gradient(S)

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(S, levels=20, cmap="magma", origin="lower")
    fig.colorbar(cs, ax=ax, label="Suspicion  S")

    step = 3
    Y, X = np.mgrid[0:S.shape[0]:step, 0:S.shape[1]:step]
    ax.quiver(X, Y, gx[::step, ::step], gy[::step, ::step],
              color="white", alpha=0.7, scale=5)

    ax.set_title("Suspicion Contour with Gradient Vectors\n"
                 "Arrows show direction of steepest suspicion increase")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUT / "05_suspicion_contour_gradient.png", dpi=DPI)
    plt.close(fig)
    print("  [5/12] Suspicion contour + gradient saved")


def proof_suspicion_3d_surface():
    """3-D surface plot of the suspicion field."""
    field = SuspicionField(width=50, height=50, D=0.15, r=0.08,
                           delta=0.01, noise_scale=0.0)
    field.reset()
    field._field[25, 25] = 1.0
    field._field[15, 35] = 0.8

    for _ in range(60):
        field.step(attack_positions=[], knowledge=0.4, access=0.2)

    S = field.field
    X, Y = np.meshgrid(range(S.shape[1]), range(S.shape[0]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, S, cmap="coolwarm", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Suspicion  S")
    ax.set_title("Suspicion Field — 3-D Surface\n"
                 "Peaks show defender-awareness hotspots")
    ax.view_init(elev=35, azim=135)
    fig.tight_layout()
    fig.savefig(OUT / "06_suspicion_3d_surface.png", dpi=DPI)
    plt.close(fig)
    print("  [6/12] Suspicion 3-D surface saved")


# =====================================================================
# 3.  ACCESS PROPAGATION
# =====================================================================

def proof_access_epidemic_curves():
    """Epidemic-style curves: access level per host over time."""
    G_raw = nx.barabasi_albert_graph(8, 2, seed=42)
    G = nx.relabel_nodes(G_raw, {n: f"h{n}" for n in G_raw.nodes()})
    access = AccessPropagation(eta=0.25, mu=0.015, noise_scale=0.0)

    for n in G.nodes():
        init = 0.4 if n == "h0" else 0.0
        access.add_host(n, initial_access=init,
                        services=["ssh"] if int(n[1:]) % 2 == 0 else ["http"])
    access.set_network(G)

    steps = 250
    traces = {n: [] for n in G.nodes()}

    for _ in range(steps):
        state = access.step(knowledge=0.7, attack_intensity=0.8)
        for hid, val in state.host_accesses.items():
            traces[hid].append(val)

    fig, ax = plt.subplots(figsize=(12, 6))
    for hid, vals in traces.items():
        ax.plot(vals, linewidth=1.8, label=hid)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Compromise threshold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Access level  A")
    ax.set_title("Access Propagation — Epidemic Curves\n"
                 r"$dA/dt = \eta K A(1-A) - \mu A + \Sigma w_{ji} A_j (1-A_i)$")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "07_access_epidemic_curves.png", dpi=DPI)
    plt.close(fig)
    print("  [7/12] Access epidemic curves saved")


def proof_access_network_viz():
    """Network graph with node colour = access level at final step."""
    G_raw = nx.barabasi_albert_graph(12, 2, seed=7)
    G = nx.relabel_nodes(G_raw, {n: f"h{n}" for n in G_raw.nodes()})
    access = AccessPropagation(eta=0.25, mu=0.015, noise_scale=0.0)

    for n in G.nodes():
        access.add_host(n, initial_access=0.5 if n == "h0" else 0.0)
    access.set_network(G)

    for _ in range(150):
        access.step(knowledge=0.7, attack_intensity=0.8)

    vals = [access.hosts[n].access for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500,
                                   node_color=vals, cmap=cm.RdYlGn_r,
                                   vmin=0, vmax=1)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    fig.colorbar(nodes, ax=ax, label="Access level")
    ax.set_title("Network Topology — Access Level per Host\n"
                 "Red = compromised, Green = safe")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "08_access_network_graph.png", dpi=DPI)
    plt.close(fig)
    print("  [8/12] Access network graph saved")


def proof_access_host_time_heatmap():
    """Heatmap: host (y) vs time (x), colour = access."""
    G_raw = nx.cycle_graph(6)
    G = nx.relabel_nodes(G_raw, {n: f"h{n}" for n in G_raw.nodes()})
    access = AccessPropagation(eta=0.25, mu=0.015, noise_scale=0.0)
    for n in G.nodes():
        access.add_host(n, initial_access=0.3 if n == "h0" else 0.0)
    access.set_network(G)

    steps = 200
    host_ids = sorted(access.hosts.keys())
    matrix = np.zeros((len(host_ids), steps))

    for t in range(steps):
        state = access.step(knowledge=0.6, attack_intensity=0.7)
        for i, hid in enumerate(host_ids):
            matrix[i, t] = state.host_accesses.get(hid, 0)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   origin="lower")
    ax.set_yticks(range(len(host_ids)))
    ax.set_yticklabels(host_ids)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Host")
    fig.colorbar(im, ax=ax, label="Access level")
    ax.set_title("Host × Time Access Heatmap — Lateral Movement Wavefront")
    fig.tight_layout()
    fig.savefig(OUT / "09_access_host_time_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  [9/12] Access host-time heatmap saved")


# =====================================================================
# 4.  HJB CONTROLLER
# =====================================================================

def proof_hjb_value_surface():
    """3-D surface of V(K, S) at a fixed access level."""
    ctrl = HJBController(grid_points=15, gamma=0.95)
    ctrl.solve(max_iterations=30, tolerance=1e-2)

    a_idx = ctrl.grid_points // 2
    V_slice = ctrl.value_grid[:, :, a_idx]

    K, S = np.meshgrid(ctrl.k_grid, ctrl.s_grid)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, S, V_slice.T, cmap="plasma", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Knowledge  K")
    ax.set_ylabel("Suspicion  S")
    ax.set_zlabel("Value  V(K, S)")
    ax.set_title(f"HJB Value Function — V(K, S) at A = {ctrl.a_grid[a_idx]:.2f}\n"
                 r"$0 = \min_u \{r(x,u) + \nabla V \cdot f(x,u)\}$")
    ax.view_init(elev=30, azim=225)
    fig.tight_layout()
    fig.savefig(OUT / "10_hjb_value_surface.png", dpi=DPI)
    plt.close(fig)
    print("  [10/12] HJB value surface saved")


def proof_hjb_policy_quiver():
    """Quiver plot of optimal (attack, stealth) over (K, S)."""
    ctrl = HJBController(grid_points=15, gamma=0.95)
    ctrl.solve(max_iterations=30, tolerance=1e-2)

    a_idx = ctrl.grid_points // 2
    A_fixed = ctrl.a_grid[a_idx]

    n = ctrl.grid_points
    U1 = np.zeros((n, n))
    U2 = np.zeros((n, n))

    for i, K in enumerate(ctrl.k_grid):
        for j, S in enumerate(ctrl.s_grid):
            action = ctrl.get_optimal_action(K, S, A_fixed)
            U1[j, i] = action.attack_intensity
            U2[j, i] = action.stealth

    K_mesh, S_mesh = np.meshgrid(ctrl.k_grid, ctrl.s_grid)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    q = ax.quiver(K_mesh, S_mesh, U1, np.zeros_like(U1),
                  U1, cmap="Reds", scale=15)
    ax.set_xlabel("Knowledge  K")
    ax.set_ylabel("Suspicion  S")
    ax.set_title(f"Optimal Attack Intensity  $u_1(K,S)$\nat A = {A_fixed:.2f}")
    fig.colorbar(q, ax=ax, label="Attack intensity")

    ax = axes[1]
    q = ax.quiver(K_mesh, S_mesh, np.zeros_like(U2), U2,
                  U2, cmap="Blues", scale=15)
    ax.set_xlabel("Knowledge  K")
    ax.set_ylabel("Suspicion  S")
    ax.set_title(f"Optimal Stealth  $u_2(K,S)$\nat A = {A_fixed:.2f}")
    fig.colorbar(q, ax=ax, label="Stealth")

    fig.suptitle("HJB Optimal Policy Quiver Plots", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "11_hjb_policy_quiver.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [11/12] HJB policy quiver saved")


def proof_hjb_cost_contour():
    """Running cost contour r(x, u) over (K, S)."""
    ctrl = HJBController(grid_points=40)

    K_vals = np.linspace(0, 100, 60)
    S_vals = np.linspace(0, 1, 60)
    K_mesh, S_mesh = np.meshgrid(K_vals, S_vals)
    Cost = np.zeros_like(K_mesh)

    A_fixed = 0.5
    control = Control(attack_intensity=0.5, stealth=0.5)

    for i in range(K_mesh.shape[0]):
        for j in range(K_mesh.shape[1]):
            Cost[i, j] = ctrl.running_cost(
                (K_mesh[i, j], S_mesh[i, j], A_fixed), control
            )

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(K_mesh, S_mesh, Cost, levels=30, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax, label="Running cost  r(x, u)")
    ax.contour(K_mesh, S_mesh, Cost, levels=15, colors="black",
               linewidths=0.5, alpha=0.4)
    ax.set_xlabel("Knowledge  K")
    ax.set_ylabel("Suspicion  S")
    ax.set_title("Running Cost Landscape  r(x, u)\n"
                 r"$r = -R_A A - R_K K + P_S S + C_1 u_1^2 + C_2 u_2^2$"
                 f"\nA = {A_fixed}, u = (0.5, 0.5)")
    fig.tight_layout()
    fig.savefig(OUT / "12_hjb_cost_contour.png", dpi=DPI)
    plt.close(fig)
    print("  [12/12] HJB cost contour saved")


# =====================================================================
# 5.  COUPLED SYSTEM DYNAMICS
# =====================================================================

def proof_coupled_dashboard():
    """Multi-panel dashboard: K, S, A evolving together."""
    knowledge_model = KnowledgeEvolution(alpha=0.15, beta=0.02, gamma=0.08,
                                         k_max=100, noise_scale=0.0)
    suspicion_model = SuspicionField(width=30, height=30, D=0.15, r=0.08,
                                     delta=0.02, noise_scale=0.0)
    access_model = AccessPropagation(eta=0.25, mu=0.015, noise_scale=0.0)

    G = nx.cycle_graph(5)
    for n in G.nodes():
        access_model.add_host(f"h{n}", initial_access=0.2 if n == 0 else 0.0)
    access_model.set_network(G)

    knowledge_model.reset(initial_knowledge=1.0)
    suspicion_model.reset()
    suspicion_model._field[15, 15] = 0.5

    steps = 300
    K_trace, S_trace, A_trace = [], [], []

    for step in range(steps):
        K = knowledge_model.knowledge
        S_mean = suspicion_model.field.mean()
        A = access_model.global_access

        K_trace.append(K)
        S_trace.append(S_mean)
        A_trace.append(A)

        attack_intensity = 0.5 + 0.3 * np.sin(step / 20)
        knowledge_model.step(suspicion=S_mean, learning_action=attack_intensity)
        suspicion_model.step(
            attack_positions=[(0.5, 0.5, attack_intensity * 0.5)],
            knowledge=K / 100, access=A)
        access_model.step(knowledge=K / 100, attack_intensity=attack_intensity)

    t = np.arange(steps)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, K_trace, "b-", linewidth=2)
    axes[0].set_ylabel("Knowledge  K")
    axes[0].set_title("Coupled System Dynamics — 300-Step Simulation")

    axes[1].plot(t, S_trace, "r-", linewidth=2)
    axes[1].set_ylabel("Mean Suspicion  S̄")

    axes[2].plot(t, A_trace, "g-", linewidth=2)
    axes[2].set_ylabel("Global Access  A")
    axes[2].set_xlabel("Time step")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "13_coupled_dashboard.png", dpi=DPI)
    plt.close(fig)
    print("  [13/14] Coupled dashboard saved")

    return K_trace, S_trace, A_trace


def proof_coupled_correlation(K_trace, S_trace, A_trace):
    """Correlation heatmap between K, S, A."""
    data = np.column_stack([K_trace, S_trace, A_trace])
    corr = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=["K", "S̄", "A"],
                yticklabels=["K", "S̄", "A"],
                vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=1, linecolor="white")
    ax.set_title("Cross-Correlation between State Variables\n"
                 "K = Knowledge,  S̄ = Mean Suspicion,  A = Access")
    fig.tight_layout()
    fig.savefig(OUT / "14_coupled_correlation.png", dpi=DPI)
    plt.close(fig)
    print("  [14/14] Coupled correlation heatmap saved")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("BlackPanther V2 — Mathematical Proofs & Visualizations")
    print("=" * 60)
    print(f"Output directory: {OUT.resolve()}\n")

    print("[Part 1] Knowledge Evolution")
    proof_knowledge_alpha_sweep()
    proof_knowledge_phase_portrait()
    proof_knowledge_sensitivity_heatmap()

    print("\n[Part 2] Suspicion Field")
    proof_suspicion_heatmap_snapshots()
    proof_suspicion_contour_gradient()
    proof_suspicion_3d_surface()

    print("\n[Part 3] Access Propagation")
    proof_access_epidemic_curves()
    proof_access_network_viz()
    proof_access_host_time_heatmap()

    print("\n[Part 4] HJB Controller")
    proof_hjb_value_surface()
    proof_hjb_policy_quiver()
    proof_hjb_cost_contour()

    print("\n[Part 5] Coupled System")
    K, S, A = proof_coupled_dashboard()
    proof_coupled_correlation(K, S, A)

    print("\n" + "=" * 60)
    print(f"All 14 figures saved to {OUT.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
