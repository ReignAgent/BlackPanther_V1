"""Interactive Plotly-based visualizations for real-time streaming.

Generates Plotly JSON that can be rendered by React Native frontend
via react-plotly.js or WebView.

Chart types:
  - knowledge: Real-time knowledge evolution curve
  - suspicion_2d: Interactive 2D heatmap with zoom/pan
  - suspicion_3d: Interactive 3D surface (only for suspicion)
  - access: Animated bar chart per host
  - hjb_policy: Contour plot with quiver overlay
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from blackpanther.core.access import AccessPropagation
from blackpanther.core.control import HJBController
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField


class InteractiveVisualizer:
    """Generates Plotly-based interactive visualizations.
    
    All outputs are JSON-serializable dictionaries that can be
    sent over WebSocket to the React Native frontend.
    """

    def __init__(
        self,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
        hjb: Optional[HJBController] = None,
    ) -> None:
        self.k = k_model
        self.s = s_model
        self.a = a_model
        self.hjb = hjb

    def generate_all_charts(self) -> Dict[str, Dict[str, Any]]:
        """Generate all chart types and return as JSON-serializable dict."""
        return {
            "knowledge": self.plot_knowledge_curve(),
            "suspicion_2d": self.plot_suspicion_heatmap_2d(),
            "suspicion_3d": self.plot_suspicion_surface_3d(),
            "access": self.plot_access_bars(),
            "hjb_policy": self.plot_hjb_policy(),
            "hjb_contour": self.plot_hjb_contour(),
        }

    def plot_knowledge_curve(self) -> Dict[str, Any]:
        """Generate knowledge evolution curve with Plotly."""
        history = self.k.history
        
        if len(history) < 2:
            ts = [0, 1]
            ks = [self.k.knowledge, self.k.knowledge]
        else:
            ts = [s.timestamp for s in history]
            ks = [s.knowledge for s in history]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ts,
            y=ks,
            mode="lines+markers",
            name="K(t)",
            line=dict(color="#00e5ff", width=3),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(0, 229, 255, 0.15)",
        ))

        fig.update_layout(
            title=dict(
                text=f"Knowledge Evolution | K = {ks[-1]:.2f}",
                font=dict(size=16, color="white"),
            ),
            xaxis=dict(
                title="Time",
                gridcolor="rgba(255,255,255,0.1)",
                color="white",
            ),
            yaxis=dict(
                title="Knowledge K",
                gridcolor="rgba(255,255,255,0.1)",
                color="white",
            ),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="white"),
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
            ),
        )

        return {
            "data": fig.to_dict()["data"],
            "layout": fig.to_dict()["layout"],
            "config": {"responsive": True, "displayModeBar": True},
        }

    def plot_suspicion_heatmap_2d(self) -> Dict[str, Any]:
        """Generate 2D suspicion heatmap with Plotly."""
        field = self.s.field
        mean_s = float(np.mean(field))
        max_s = float(np.max(field))

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=field,
            colorscale="Inferno",
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Suspicion S",
                titlefont=dict(color="white"),
                tickfont=dict(color="white"),
            ),
            hovertemplate="X: %{x}<br>Y: %{y}<br>S: %{z:.4f}<extra></extra>",
        ))

        gy, gx = np.gradient(field)
        step = max(1, field.shape[0] // 15)
        x_coords = list(range(0, field.shape[1], step))
        y_coords = list(range(0, field.shape[0], step))

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if y < field.shape[0] and x < field.shape[1]:
                    dx = gx[y, x]
                    dy = gy[y, x]
                    mag = np.sqrt(dx**2 + dy**2)
                    if mag > 0.01:
                        fig.add_annotation(
                            x=x, y=y,
                            ax=x + dx * 3 / max(mag, 0.1),
                            ay=y + dy * 3 / max(mag, 0.1),
                            xref="x", yref="y",
                            axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor="rgba(255,255,255,0.5)",
                        )

        fig.update_layout(
            title=dict(
                text=f"Suspicion Field | Mean={mean_s:.4f} Max={max_s:.4f}",
                font=dict(size=16, color="white"),
            ),
            xaxis=dict(
                title="Network X",
                color="white",
                showgrid=False,
            ),
            yaxis=dict(
                title="Network Y",
                color="white",
                showgrid=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="white"),
        )

        return {
            "data": fig.to_dict()["data"],
            "layout": fig.to_dict()["layout"],
            "config": {"responsive": True, "displayModeBar": True, "scrollZoom": True},
        }

    def plot_suspicion_surface_3d(self) -> Dict[str, Any]:
        """Generate 3D interactive surface for suspicion field."""
        field = self.s.field
        mean_s = float(np.mean(field))
        
        x = np.arange(field.shape[1])
        y = np.arange(field.shape[0])

        fig = go.Figure()

        fig.add_trace(go.Surface(
            z=field,
            x=x,
            y=y,
            colorscale="Viridis",
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="Suspicion S",
                titlefont=dict(color="white"),
                tickfont=dict(color="white"),
                x=1.02,
            ),
            hovertemplate="X: %{x}<br>Y: %{y}<br>S: %{z:.4f}<extra></extra>",
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
            ),
            lightposition=dict(x=100, y=200, z=0),
        ))

        threshold_z = np.full_like(field, 0.7)
        fig.add_trace(go.Surface(
            z=threshold_z,
            x=x,
            y=y,
            colorscale=[[0, "rgba(255,23,68,0.3)"], [1, "rgba(255,23,68,0.3)"]],
            showscale=False,
            name="Threshold (0.7)",
            hoverinfo="skip",
        ))

        fig.update_layout(
            title=dict(
                text=f"3D Suspicion Surface | Mean={mean_s:.4f}",
                font=dict(size=16, color="white"),
            ),
            scene=dict(
                xaxis=dict(
                    title="Network X",
                    backgroundcolor="#16213e",
                    gridcolor="rgba(255,255,255,0.1)",
                    color="white",
                ),
                yaxis=dict(
                    title="Network Y",
                    backgroundcolor="#16213e",
                    gridcolor="rgba(255,255,255,0.1)",
                    color="white",
                ),
                zaxis=dict(
                    title="Suspicion S",
                    backgroundcolor="#16213e",
                    gridcolor="rgba(255,255,255,0.1)",
                    color="white",
                    range=[0, 1],
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            paper_bgcolor="#1a1a2e",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        return {
            "data": fig.to_dict()["data"],
            "layout": fig.to_dict()["layout"],
            "config": {"responsive": True, "displayModeBar": True},
        }

    def plot_access_bars(self) -> Dict[str, Any]:
        """Generate animated bar chart for access levels."""
        hosts = self.a.hosts
        
        if not hosts:
            names = ["No hosts"]
            values = [0.0]
            colors = ["#00e5ff"]
        else:
            names = list(hosts.keys())
            values = [hosts[n].access for n in names]
            colors = [
                "#ff1744" if hosts[n].compromised else "#00e5ff"
                for n in names
            ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="white", width=1),
            ),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(color="white"),
            hovertemplate="%{y}<br>Access: %{x:.4f}<extra></extra>",
        ))

        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_color="#ffea00",
            annotation_text="Compromise threshold",
            annotation_position="top",
            annotation_font_color="white",
        )

        n_compromised = sum(1 for h in hosts.values() if h.compromised) if hosts else 0
        
        fig.update_layout(
            title=dict(
                text=f"Access Propagation | Global={self.a.global_access:.3f} | Compromised={n_compromised}/{len(names)}",
                font=dict(size=16, color="white"),
            ),
            xaxis=dict(
                title="Access Level A",
                range=[0, 1.1],
                gridcolor="rgba(255,255,255,0.1)",
                color="white",
            ),
            yaxis=dict(
                title="Host",
                color="white",
            ),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="white"),
            bargap=0.3,
        )

        return {
            "data": fig.to_dict()["data"],
            "layout": fig.to_dict()["layout"],
            "config": {"responsive": True, "displayModeBar": True},
        }

    def plot_hjb_policy(self) -> Dict[str, Any]:
        """Generate HJB policy visualization with quiver overlay."""
        k_vals = np.linspace(0, 100, 30)
        s_vals = np.linspace(0, 1, 30)
        KK, SS = np.meshgrid(k_vals, s_vals)

        attack = np.clip(0.8 * (KK / 100) * (1 - SS), 0, 1)
        stealth = np.clip(SS ** 0.5, 0, 1)

        if self.hjb and self.hjb.policy_table:
            for i, K in enumerate(k_vals):
                for j, S in enumerate(s_vals):
                    ctrl = self.hjb.policy_table.get((K, S, 0.3))
                    if ctrl:
                        attack[j, i] = ctrl.attack_intensity
                        stealth[j, i] = ctrl.stealth

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Optimal Attack Intensity u₁", "Optimal Stealth Level u₂"),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Contour(
                z=attack,
                x=k_vals,
                y=s_vals,
                colorscale="Viridis",
                contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
                colorbar=dict(title="u₁", x=0.45, len=0.9),
                hovertemplate="K: %{x:.1f}<br>S: %{y:.2f}<br>Attack: %{z:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Contour(
                z=stealth,
                x=k_vals,
                y=s_vals,
                colorscale="Plasma",
                contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
                colorbar=dict(title="u₂", x=1.02, len=0.9),
                hovertemplate="K: %{x:.1f}<br>S: %{y:.2f}<br>Stealth: %{z:.2f}<extra></extra>",
            ),
            row=1, col=2
        )

        for col in [1, 2]:
            fig.add_hline(
                y=0.7, line_dash="dash", line_color="#ff1744",
                annotation_text="S threshold" if col == 1 else None,
                row=1, col=col
            )

        fig.update_xaxes(title_text="Knowledge K", color="white", gridcolor="rgba(255,255,255,0.1)")
        fig.update_yaxes(title_text="Suspicion S", color="white", gridcolor="rgba(255,255,255,0.1)")

        fig.update_layout(
            title=dict(
                text="HJB Optimal Policy | 0 = min_u { r(x,u) + ∇V·f }",
                font=dict(size=16, color="white"),
            ),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="white"),
            height=500,
        )

        return {
            "data": fig.to_dict()["data"],
            "layout": fig.to_dict()["layout"],
            "config": {"responsive": True, "displayModeBar": True},
        }

    def plot_hjb_contour(self) -> Dict[str, Any]:
        """Generate HJB value function contour plot."""
        k_vals = np.linspace(0, 100, 40)
        s_vals = np.linspace(0, 1, 40)
        KK, SS = np.meshgrid(k_vals, s_vals)

        V = -KK / 100 + SS ** 2 + 0.1 * np.sin(KK / 10) * np.cos(SS * np.pi)

        fig = go.Figure()

        fig.add_trace(go.Contour(
            z=V,
            x=k_vals,
            y=s_vals,
            colorscale="RdBu_r",
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            colorbar=dict(
                title="V(K,S)",
                titlefont=dict(color="white"),
                tickfont=dict(color="white"),
            ),
            hovertemplate="K: %{x:.1f}<br>S: %{y:.2f}<br>V: %{z:.3f}<extra></extra>",
        ))

        step = 5
        for i in range(0, len(k_vals), step):
            for j in range(0, len(s_vals), step):
                dV_dK = (V[j, min(i+1, len(k_vals)-1)] - V[j, max(i-1, 0)]) / 2
                dV_dS = (V[min(j+1, len(s_vals)-1), i] - V[max(j-1, 0), i]) / 2
                mag = np.sqrt(dV_dK**2 + dV_dS**2)
                if mag > 0.01:
                    fig.add_annotation(
                        x=k_vals[i], y=s_vals[j],
                        ax=k_vals[i] - dV_dK * 5 / mag,
                        ay=s_vals[j] - dV_dS * 0.05 / mag,
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=0.8,
                        arrowwidth=1.5,
                        arrowcolor="white",
                    )

        fig.update_layout(
            title=dict(
                text="HJB Value Function V(K,S) with Gradient Flow",
                font=dict(size=16, color="white"),
            ),
            xaxis=dict(
                title="Knowledge K",
                color="white",
                gridcolor="rgba(255,255,255,0.1)",
            ),
            yaxis=dict(
                title="Suspicion S",
                color="white",
                gridcolor="rgba(255,255,255,0.1)",
            ),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="white"),
        )

        return {
            "data": fig.to_dict()["data"],
            "layout": fig.to_dict()["layout"],
            "config": {"responsive": True, "displayModeBar": True},
        }


def to_plotly_json(fig: go.Figure) -> Dict[str, Any]:
    """Convert a Plotly figure to JSON-serializable dict."""
    return {
        "data": fig.to_dict()["data"],
        "layout": fig.to_dict()["layout"],
    }
