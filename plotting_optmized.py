"""
PlottingManager Module
======================

Handles all plotting operations for the CommHEAT analysis pipeline.
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class PlottingManager:
    """Manages all plotting operations with consistent styling"""

    # Define color scheme once
    COLORS = {
        'predicted': '#9370DB',      # Purple
        'hobo_max': '#000000',       # Black
        'hobo_avg': '#00AA00',       # Green
        'archetype_1': '#DC143C',    # Crimson Red
        'archetype_2': '#0066FF',    # Blue
        'ac': '#0066CC',             # Blue
        'noac': '#CC0000'            # Red
    }

    def __init__(self, output_dir: Path, plot_style: Dict):
        self.output_dir = output_dir
        self.plot_dir = output_dir / "plots"
        self.comparison_dir = output_dir / "comparisons"
        plt.rcParams.update(plot_style)

    def _create_base_plot(self, figsize: Tuple[int, int] = (14, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """Create base figure with common formatting"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        return fig, ax

    def _finalize_plot(self, fig: plt.Figure, ax: plt.Axes, output_path: Path, 
                       xlabel: str = "Time", ylabel: str = "Temperature (°C)"):
        """Apply common formatting and save"""
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.95, edgecolor="black")
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot: {output_path.name}")
        plt.close(fig)

    def _plot_temperature_lines(self, ax: plt.Axes, data: pd.DataFrame, 
                                 plot_config: List[Tuple[str, str, Dict]]):
        """Plot multiple temperature series with configuration
        
        Args:
            ax: Matplotlib axes
            data: DataFrame containing temperature data
            plot_config: List of (column_name, label, style_kwargs) tuples
        """
        for col_name, label, style in plot_config:
            if col_name in data.columns:
                ax.plot(data.index, data[col_name], label=label, **style)

    def plot_comparison(
        self,
        combined_data: pd.DataFrame,
        title: str,
        output_path: Path,
        archetypes: Optional[str] = None,
        mse_avg: Optional[float] = None,
        mse_max: Optional[float] = None,
    ) -> None:
        """Unified plotting for all comparison types (intersection, heat events, period intersection)"""
        
        fig, ax = self._create_base_plot()
        
        try:
            # Build plot configuration
            plot_config = []
            
            # Individual archetypes (if present)
            if archetypes:
                archetype_list = [a.strip() for a in archetypes.split(",")]
                colors = [self.COLORS['archetype_1'], self.COLORS['archetype_2']]
                
                for i, arch in enumerate(archetype_list):
                    col_name = f"{arch}_temp"
                    plot_config.append((
                        col_name,
                        f"{arch} (individual)",
                        {'color': colors[i % len(colors)], 'linewidth': 1.8, 
                         'linestyle': '--', 'alpha': 0.7, 'zorder': 2}
                    ))
            
            # Averaged prediction or single archetype
            if 'T_predicted' in combined_data.columns:
                label = f"T_predicted (avg: {archetypes})" if archetypes else "T_sim"
                plot_config.append((
                    'T_predicted',
                    label,
                    {'color': self.COLORS['predicted'], 'linewidth': 2.5, 
                     'alpha': 0.85, 'zorder': 3}
                ))
            elif 'archetype_internal_temperature' in combined_data.columns:
                plot_config.append((
                    'archetype_internal_temperature',
                    'T_sim',
                    {'color': self.COLORS['predicted'], 'linewidth': 2.0, 
                     'alpha': 0.7, 'zorder': 3}
                ))
            
            # HOBO data
            plot_config.extend([
                ('actual_max_temperature', 'Hobologger_max_temp',
                 {'color': self.COLORS['hobo_max'], 'linewidth': 2.0, 
                  'alpha': 0.9, 'zorder': 4}),
                ('actual_average_temperature', 'Hobologger_avg_temp',
                 {'color': self.COLORS['hobo_avg'], 'linewidth': 2.0, 
                  'linestyle': ':', 'alpha': 1.0, 'zorder': 5})
            ])
            
            # Plot all configured lines
            self._plot_temperature_lines(ax, combined_data, plot_config)
            
            # Add MSE to title if provided
            if mse_avg is not None and mse_max is not None:
                title += f"\nMSE(T_predicted vs Hobologger_avg)={mse_avg:.4f} | " \
                         f"MSE(T_predicted vs Hobologger_max)={mse_max:.4f}"
            
            ax.set_title(title, fontsize=12, fontweight="bold")
            self._finalize_plot(fig, ax, output_path)
            
        finally:
            plt.close(fig)

    # ============================================================
    # SIMPLIFIED PUBLIC METHODS (delegate to unified plotter)
    # ============================================================

    def plot_intersection_comparison_averaged(
        self, combined_data: pd.DataFrame, row_info: Dict, archetypes: str,
        mse_predicted_avg: float, mse_predicted_max: float,
        actual_start: pd.Timestamp, actual_end: pd.Timestamp, output_path: Path
    ) -> None:
        """Plot averaged archetype comparison for intersection period"""
        title = f"{row_info['address']}\nAveraged Archetypes: {archetypes}"
        self.plot_comparison(combined_data, title, output_path, archetypes, 
                           mse_predicted_avg, mse_predicted_max)

    def plot_heat_event(
        self, combined_data: pd.DataFrame, address: str, archetype: str,
        event_name: str, event_id: str, output_path: Path,
        mse_avg: Optional[float] = None, mse_max: Optional[float] = None
    ) -> None:
        """Plot single archetype heat event with MSE"""
        title = f"{address}\n{event_name} ({event_id}) - Archetype: {archetype}"
        
        # Add MSE to title if provided
        if mse_avg is not None and mse_max is not None:
            title += f"\nMSE(T_sim vs Avg)={mse_avg:.4f} | MSE(T_sim vs Max)={mse_max:.4f}"
        
        self.plot_comparison(combined_data, title, output_path)

    def plot_heat_event_averaged(
        self, combined_data: pd.DataFrame, address: str, archetypes: str,
        event_name: str, event_id: str, output_path: Path,
        mse_avg: Optional[float] = None, mse_max: Optional[float] = None
    ) -> None:
        """Plot averaged archetype heat event with MSE"""
        title = f"{address}\n{event_name} ({event_id}) - Averaged Simulation\n" \
                f"Archetypes: {archetypes}"
        
        # Add MSE to title if provided
        if mse_avg is not None and mse_max is not None:
            title += f"\nMSE(T_predicted vs Avg)={mse_avg:.4f} | MSE(T_predicted vs Max)={mse_max:.4f}"
        
        self.plot_comparison(combined_data, title, output_path, archetypes)

    def plot_period_intersection(
        self, combined_data: pd.DataFrame, address: str, archetypes: str,
        period_start: pd.Timestamp, period_end: pd.Timestamp, output_path: Path
    ) -> None:
        """Plot full period intersection"""
        title = f"{address}\nPeriod of Intersection: " \
                f"{period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}\n" \
                f"Averaged Archetypes: {archetypes}"
        self.plot_comparison(combined_data, title, output_path, archetypes)

    # ============================================================
    # AC VS NO-AC (Unique logic - keep separate)
    # ============================================================
    def plot_ac_vs_noac(
        self, ac_series: pd.Series, noac_series: pd.Series,
        title: str, output_path: Path
    ) -> None:
        """Plot AC vs No-AC comparison with statistics"""
        
        fig, ax = self._create_base_plot()
        
        try:
            # Plot series
            if ac_series is not None and not ac_series.empty:
                ax.plot(ac_series.index, ac_series.values, label="With AC",
                       color=self.COLORS['ac'], linewidth=2.0, alpha=0.8, zorder=3)
            
            if noac_series is not None and not noac_series.empty:
                ax.plot(noac_series.index, noac_series.values, label="Without AC",
                       color=self.COLORS['noac'], linewidth=2.0, alpha=0.8, 
                       linestyle='--', zorder=4)
            
            # Add statistics textbox
            if ac_series is not None and noac_series is not None:
                aligned = pd.DataFrame({"AC": ac_series, "NoAC": noac_series}).dropna()
                if not aligned.empty:
                    mean_diff = aligned["NoAC"].mean() - aligned["AC"].mean()
                    max_diff = aligned["NoAC"].max() - aligned["AC"].max()
                    textstr = f'Mean Diff: {mean_diff:.2f}°C\nMax Diff: {max_diff:.2f}°C'
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f"{title}\nAC vs No-AC Temperature Comparison", 
                        fontsize=12, fontweight="bold")
            self._finalize_plot(fig, ax, output_path)
            
        finally:
            plt.close(fig)
