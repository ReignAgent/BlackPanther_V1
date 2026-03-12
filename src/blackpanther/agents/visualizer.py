"""Real-Time Mathematical Proof Visualizer

Generates publication-grade plots for the five coupled models.
All plots are saved to ``output/proofs/`` after every agent action
so the mathematical proof evolves in real time.

Two rendering modes:
  realtime=True  — persistent figure windows that update live via
                   plt.ion() + plt.pause(); also saves PNGs to disk.
  realtime=False — headless Agg backend, create-save-close per call
                   (suitable for API / Celery / CI).

Plots produced:
  knowledge.png      -- K(t) with growth/decay decomposition
  suspicion.png      -- 2-D heatmap with gradient quiver + hotspots
  suspicion_3d.png   -- 3-D surface of the suspicion field
  access.png         -- Horizontal bar chart per host
  hjb_policy.png     -- Quiver plot of optimal (attack, stealth) policy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# Backend selection must happen before importing pyplot.
# We defer the import so the caller can set ``realtime`` first.
_backend_configured = False


def _configure_backend(realtime: bool) -> None:
    global _backend_configured
    if _backend_configured:
        return
    import matplotlib
    if realtime:
        try:
            if sys.platform == "darwin":
                matplotlib.use("macosx")
            else:
                matplotlib.use("TkAgg")
        except ImportError:
            matplotlib.use("Agg")
            logger.warning("[viz] interactive backend unavailable, falling back to Agg")
    else:
        matplotlib.use("Agg")
    _backend_configured = True


# We need these imports after _configure_backend, but they are
# always available as module-level names after __init__ calls
# _configure_backend.  Lazy-import pattern used below.
import matplotlib
import matplotlib.pyplot as plt

from blackpanther.core.access import AccessPropagation
from blackpanther.core.control import HJBController
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField


class Visualizer:
    """Generates and saves proof plots for the five math models.

    Args:
        k_model: Knowledge evolution instance.
        s_model: Suspicion field instance.
        a_model: Access propagation instance.
        hjb: HJB controller (may be ``None`` if not yet solved).
        output_dir: Where to write PNGs.
        dpi: Resolution for saved figures.
        realtime: If True, open persistent live windows AND save to disk.
    """

    def __init__(
        self,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
        hjb: Optional[HJBController] = None,
        output_dir: str = "output/proofs",
        dpi: int = 150,
        realtime: bool = False,
    ) -> None:
        _configure_backend(realtime)

        self.k = k_model
        self.s = s_model
        self.a = a_model
        self.hjb = hjb
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._realtime = realtime
        self._event_markers: List[Tuple[float, str, str]] = []
        self._figs: Dict[str, plt.Figure] = {}

        if self._realtime:
            plt.ion()

    # ------------------------------------------------------------------
    # Figure lifecycle helpers
    # ------------------------------------------------------------------

    def _get_or_create_fig(
        self,
        name: str,
        figsize: Tuple[float, float] = (12, 8),
        projection: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """Return a figure (persistent in real-time mode, ephemeral otherwise).

        In real-time mode the same named window is reused and cleared
        so the user sees a live-updating plot.
        """
        if self._realtime:
            if name not in self._figs or not plt.fignum_exists(self._figs[name].number):
                self._figs[name] = plt.figure(num=name, figsize=figsize)
            fig = self._figs[name]
            fig.clf()
        else:
            fig = plt.figure(figsize=figsize)

        if projection:
            ax = fig.add_subplot(111, projection=projection)
        else:
            ax = None
        return fig, ax

    def _refresh(self, fig: plt.Figure, path: Path) -> None:
        """Save to disk and, if real-time, push update to screen."""
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        if self._realtime:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)
        else:
            plt.close(fig)
        logger.debug("[viz] saved {}", path)

    def add_event(self, time: float, label: str, agent: str) -> None:
        """Record an event (recon / scan / exploit) for annotation."""
        self._event_markers.append((time, label, agent))

    # ------------------------------------------------------------------
    # 1. Knowledge Curve
    # ------------------------------------------------------------------

    def plot_knowledge_curve(self) -> Path:
        """K(t) line plot with growth/decay decomposition."""
        history = self.k.history
        path = self.out / "knowledge.png"
        if len(history) < 2:
            logger.debug("[viz] not enough K history to plot")
            return path

        ts = [s.timestamp for s in history]
        ks = [s.knowledge for s in history]
        growths = [s.metadata.get("growth", 0) for s in history]
        forgettings = [s.metadata.get("forgetting", 0) for s in history]
        from_def = [s.metadata.get("from_defense", 0) for s in history]

        with plt.style.context("dark_background"):
            fig, _ = self._get_or_create_fig("knowledge", (12, 8))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)

            ax1.plot(ts, ks, linewidth=2.5, color="#00e5ff", label="K(t)")
            ax1.fill_between(ts, 0, ks, alpha=0.15, color="#00e5ff")
            ax1.set_ylabel("Knowledge  K", fontsize=13)
            ax1.set_title(
                f"Knowledge Evolution    K = {ks[-1]:.2f}  |  dK/dt = αK(1−K/K_max) − βK + γS + σξ",
                fontsize=14, fontweight="bold",
            )

            colors = {"recon": "#76ff03", "scanner": "#ffea00", "exploit": "#ff1744"}
            for t_ev, lbl, agent in self._event_markers:
                if t_ev <= ts[-1]:
                    c = colors.get(agent, "#ffffff")
                    ax1.axvline(t_ev, color=c, linestyle="--", alpha=0.5, linewidth=0.8)
                    ax1.annotate(lbl, (t_ev, max(ks) * 0.95), fontsize=7, color=c, rotation=45)
            ax1.legend(fontsize=11)
            ax1.grid(alpha=0.2)

            ax2.plot(ts, growths, color="#76ff03", label="growth", linewidth=1.2)
            ax2.plot(ts, forgettings, color="#ff1744", label="forget", linewidth=1.2)
            ax2.plot(ts, from_def, color="#ffea00", label="from_defense", linewidth=1.2)
            ax2.set_xlabel("Time", fontsize=12)
            ax2.set_ylabel("Term value", fontsize=11)
            ax2.legend(fontsize=9, ncol=3)
            ax2.grid(alpha=0.2)

            fig.tight_layout()
            self._refresh(fig, path)
        return path

    # ------------------------------------------------------------------
    # 2. Suspicion Heatmap (2-D)
    # ------------------------------------------------------------------

    def plot_suspicion_heatmap(self) -> Path:
        """2-D suspicion field with gradient quiver and hotspots."""
        field = self.s.field
        mean_s = float(np.mean(field))
        max_s = float(np.max(field))

        with plt.style.context("dark_background"):
            fig, _ = self._get_or_create_fig("suspicion_2d", (12, 10))
            ax = fig.add_subplot(111)

            im = ax.imshow(
                field, cmap="inferno", origin="lower",
                vmin=0.0, vmax=1.0, aspect="auto",
                interpolation="bicubic",
            )
            cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cbar.set_label("Suspicion Level  S(x,y)", fontsize=12)
            cbar.ax.tick_params(labelsize=10)

            gy, gx = np.gradient(field)
            step = max(1, field.shape[0] // 20)
            Y, X = np.mgrid[0:field.shape[0]:step, 0:field.shape[1]:step]
            U = gx[::step, ::step]
            V = gy[::step, ::step]
            mag = np.sqrt(U**2 + V**2)
            mag[mag == 0] = 1
            ax.quiver(X, Y, U / mag, V / mag, mag, cmap="cool", alpha=0.6, scale=30, width=0.003)

            hotspot_y, hotspot_x = np.where(field > 0.7)
            if len(hotspot_x) > 0:
                ax.scatter(
                    hotspot_x, hotspot_y, s=60, facecolors="none",
                    edgecolors="#ff1744", linewidths=1.5,
                    label=f"Hotspots (S>0.7): {len(hotspot_x)}",
                )
                ax.legend(fontsize=10, loc="upper right")

            ax.set_title(
                f"Suspicion Field    mean={mean_s:.4f}   max={max_s:.4f}  |  "
                f"∂S/∂t = D∇²S + rS(1−S) − δKA + σξ",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlabel("Network X-coordinate", fontsize=12)
            ax.set_ylabel("Network Y-coordinate", fontsize=12)

            stats_text = (
                f"Grid: {field.shape[0]}x{field.shape[1]}\n"
                f"Mean: {mean_s:.6f}\n"
                f"Max:  {max_s:.6f}\n"
                f"Std:  {float(np.std(field)):.6f}\n"
                f"Hotspots: {len(hotspot_x)}"
            )
            ax.text(
                0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
                family="monospace",
            )

            fig.tight_layout()
            path = self.out / "suspicion.png"
            self._refresh(fig, path)
        return path

    # ------------------------------------------------------------------
    # 3. Suspicion 3-D Surface  (NEW)
    # ------------------------------------------------------------------

    def plot_suspicion_3d(self) -> Path:
        """3-D surface of the suspicion field with threshold plane."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

        field = self.s.field
        mean_s = float(np.mean(field))
        max_s = float(np.max(field))

        x = np.arange(field.shape[1])
        y = np.arange(field.shape[0])
        X, Y = np.meshgrid(x, y)

        with plt.style.context("dark_background"):
            fig, _ = self._get_or_create_fig("suspicion_3d", (14, 10))
            ax = fig.add_subplot(111, projection="3d")

            surf = ax.plot_surface(
                X, Y, field,
                cmap="inferno",
                vmin=0.0, vmax=1.0,
                alpha=0.92,
                rstride=max(1, field.shape[0] // 40),
                cstride=max(1, field.shape[1] // 40),
                edgecolor="none",
                antialiased=True,
            )

            threshold_z = np.full_like(field, 0.7)
            ax.plot_surface(
                X, Y, threshold_z,
                color="#ff1744",
                alpha=0.15,
                rstride=field.shape[0],
                cstride=field.shape[1],
            )

            cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, aspect=20)
            cbar.set_label("Suspicion Level  S", fontsize=11)
            cbar.ax.tick_params(labelsize=9)

            hotspot_y, hotspot_x = np.where(field > 0.7)
            if len(hotspot_x) > 0:
                ax.scatter(
                    hotspot_x, hotspot_y,
                    field[hotspot_y, hotspot_x],
                    c="#ff1744", s=25, alpha=0.9,
                    depthshade=True, label=f"Hotspots: {len(hotspot_x)}",
                )
                ax.legend(fontsize=9, loc="upper left")

            ax.set_xlabel("Network X", fontsize=11, labelpad=8)
            ax.set_ylabel("Network Y", fontsize=11, labelpad=8)
            ax.set_zlabel("Suspicion S", fontsize=11, labelpad=8)
            ax.set_zlim(0, 1.0)

            ax.set_title(
                f"3D Suspicion Surface    mean={mean_s:.4f}   max={max_s:.4f}\n"
                f"∂S/∂t = D∇²S + rS(1−S) − δKA + σξ",
                fontsize=13, fontweight="bold", pad=15,
            )

            ax.view_init(elev=35, azim=-60)

            stats_text = (
                f"Grid: {field.shape[0]}x{field.shape[1]}\n"
                f"Mean: {mean_s:.6f}\n"
                f"Max:  {max_s:.6f}\n"
                f"Std:  {float(np.std(field)):.6f}\n"
                f"Hotspots(>0.7): {len(hotspot_x)}"
            )
            ax.text2D(
                0.02, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
                family="monospace",
            )

            fig.tight_layout()
            path = self.out / "suspicion_3d.png"
            self._refresh(fig, path)
        return path

    # ------------------------------------------------------------------
    # 4. Access per Host
    # ------------------------------------------------------------------

    def plot_access_bars(self) -> Path:
        """Horizontal bar chart of access level per host."""
        hosts = self.a.hosts
        path = self.out / "access.png"
        if not hosts:
            logger.debug("[viz] no hosts for access plot")
            return path

        names = list(hosts.keys())
        values = [hosts[n].access for n in names]
        compromised = [hosts[n].compromised for n in names]
        bar_colors = ["#ff1744" if c else "#00e5ff" for c in compromised]

        with plt.style.context("dark_background"):
            fig, _ = self._get_or_create_fig(
                "access", (10, max(4, len(names) * 0.5 + 2)),
            )
            ax = fig.add_subplot(111)

            bars = ax.barh(names, values, color=bar_colors, edgecolor="white", linewidth=0.3)
            ax.axvline(0.5, color="#ffea00", linestyle="--", linewidth=1.2, alpha=0.7, label="Compromise threshold")

            for bar, val, comp in zip(bars, values, compromised):
                label = f"{val:.3f} {'[PWNED]' if comp else ''}"
                ax.text(
                    min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=9, color="white",
                )

            n_compromised = sum(compromised)
            ax.set_title(
                f"Access Propagation    Global={self.a.global_access:.3f}   "
                f"Compromised={n_compromised}/{len(names)}  |  "
                f"dA/dt = ηKA(1−A) − μA + Σw_jiA_j(1−A_i)",
                fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("Access Level  A", fontsize=12)
            ax.set_xlim(0, 1.05)
            ax.legend(fontsize=10)
            ax.grid(axis="x", alpha=0.2)

            fig.tight_layout()
            self._refresh(fig, path)
        return path

    # ------------------------------------------------------------------
    # 5. HJB Policy Quiver
    # ------------------------------------------------------------------

    def plot_hjb_policy(self, fixed_access: float = 0.3) -> Path:
        """Quiver plot of optimal (attack, stealth) vs (K, S)."""
        if self.hjb is None or not self.hjb.policy_table:
            logger.debug("[viz] HJB not solved yet — generating from model defaults")
            return self._plot_hjb_analytical(fixed_access)

        k_vals = self.hjb.k_grid
        s_vals = self.hjb.s_grid
        a_idx = int(np.argmin(np.abs(self.hjb.a_grid - fixed_access)))

        attack = np.zeros((len(s_vals), len(k_vals)))
        stealth = np.zeros_like(attack)

        for i, K in enumerate(k_vals):
            for j, S in enumerate(s_vals):
                A = self.hjb.a_grid[a_idx]
                ctrl = self.hjb.policy_table.get((K, S, A))
                if ctrl:
                    attack[j, i] = ctrl.attack_intensity
                    stealth[j, i] = ctrl.stealth
                else:
                    attack[j, i] = 0.5
                    stealth[j, i] = 0.5

        with plt.style.context("dark_background"):
            fig, _ = self._get_or_create_fig("hjb_policy", (16, 7))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            KK, SS = np.meshgrid(k_vals, s_vals)

            c1 = ax1.contourf(KK, SS, attack, levels=20, cmap="viridis")
            ax1.quiver(
                KK[::2, ::2], SS[::2, ::2],
                attack[::2, ::2], np.zeros_like(attack[::2, ::2]),
                color="white", alpha=0.6, scale=15,
            )
            fig.colorbar(c1, ax=ax1, label="Attack Intensity u₁")
            ax1.axhline(0.7, color="#ff1744", linestyle="--", linewidth=1.5, label="Suspicion threshold")
            ax1.set_xlabel("Knowledge  K", fontsize=12)
            ax1.set_ylabel("Suspicion  S", fontsize=12)
            ax1.set_title("Optimal Attack Intensity", fontsize=13, fontweight="bold")
            ax1.legend(fontsize=9)

            c2 = ax2.contourf(KK, SS, stealth, levels=20, cmap="plasma")
            ax2.quiver(
                KK[::2, ::2], SS[::2, ::2],
                np.zeros_like(stealth[::2, ::2]), stealth[::2, ::2],
                color="white", alpha=0.6, scale=15,
            )
            fig.colorbar(c2, ax=ax2, label="Stealth Level u₂")
            ax2.axhline(0.7, color="#ff1744", linestyle="--", linewidth=1.5, label="Suspicion threshold")
            ax2.set_xlabel("Knowledge  K", fontsize=12)
            ax2.set_ylabel("Suspicion  S", fontsize=12)
            ax2.set_title("Optimal Stealth Level", fontsize=13, fontweight="bold")
            ax2.legend(fontsize=9)

            fig.suptitle(
                f"HJB Policy    A={fixed_access:.1f}  |  "
                "0 = min_u {{ r(x,u) + ∇V·f + ½Tr(σ²∇²V) }}",
                fontsize=14, fontweight="bold", y=1.02,
            )
            fig.tight_layout()
            path = self.out / "hjb_policy.png"
            self._refresh(fig, path)
        return path

    def _plot_hjb_analytical(self, fixed_access: float) -> Path:
        """Fallback when HJB hasn't been solved: show analytical heuristic."""
        k_vals = np.linspace(0, 100, 40)
        s_vals = np.linspace(0, 1, 40)
        KK, SS = np.meshgrid(k_vals, s_vals)

        attack = np.clip(0.8 * (KK / 100) * (1 - SS), 0, 1)
        stealth = np.clip(SS ** 0.5, 0, 1)

        with plt.style.context("dark_background"):
            fig, _ = self._get_or_create_fig("hjb_policy", (16, 7))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            step = 3
            c1 = ax1.contourf(KK, SS, attack, levels=20, cmap="viridis")
            ax1.quiver(
                KK[::step, ::step], SS[::step, ::step],
                attack[::step, ::step], np.zeros_like(attack[::step, ::step]),
                color="white", alpha=0.5, scale=15,
            )
            fig.colorbar(c1, ax=ax1, label="Attack Intensity u₁")
            ax1.axhline(0.7, color="#ff1744", linestyle="--", linewidth=1.5, label="S threshold=0.7")
            ax1.set_xlabel("Knowledge  K", fontsize=12)
            ax1.set_ylabel("Suspicion  S", fontsize=12)
            ax1.set_title("Attack Intensity (heuristic)", fontsize=13, fontweight="bold")
            ax1.legend(fontsize=9)

            c2 = ax2.contourf(KK, SS, stealth, levels=20, cmap="plasma")
            ax2.quiver(
                KK[::step, ::step], SS[::step, ::step],
                np.zeros_like(stealth[::step, ::step]), stealth[::step, ::step],
                color="white", alpha=0.5, scale=15,
            )
            fig.colorbar(c2, ax=ax2, label="Stealth Level u₂")
            ax2.axhline(0.7, color="#ff1744", linestyle="--", linewidth=1.5, label="S threshold=0.7")
            ax2.set_xlabel("Knowledge  K", fontsize=12)
            ax2.set_ylabel("Suspicion  S", fontsize=12)
            ax2.set_title("Stealth Level (heuristic)", fontsize=13, fontweight="bold")
            ax2.legend(fontsize=9)

            fig.suptitle(
                f"HJB Policy (analytical fallback)    A={fixed_access:.1f}  |  "
                "0 = min_u {{ r(x,u) + ∇V·f + ½Tr(σ²∇²V) }}",
                fontsize=14, fontweight="bold", y=1.02,
            )
            fig.tight_layout()
            path = self.out / "hjb_policy.png"
            self._refresh(fig, path)
        return path

    # ------------------------------------------------------------------
    # Convenience — plot everything
    # ------------------------------------------------------------------

    def plot_all(self, fixed_access: float = 0.3) -> Dict[str, Path]:
        """Generate all five plots and return their paths."""
        paths = {
            "knowledge": self.plot_knowledge_curve(),
            "suspicion": self.plot_suspicion_heatmap(),
            "suspicion_3d": self.plot_suspicion_3d(),
            "access": self.plot_access_bars(),
            "hjb_policy": self.plot_hjb_policy(fixed_access),
        }
        logger.info("[viz] all plots saved to {}", self.out)
        return paths
