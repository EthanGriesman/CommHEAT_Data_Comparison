"""
PlottingManager Module
======================

Handles all plotting operations for the CommHEAT analysis pipeline.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PlottingManager:
    """Manages all plotting operations with consistent styling"""

    def __init__(self, output_dir: Path, plot_style: Dict):
        self.output_dir = output_dir
        self.plot_dir = output_dir / "plots"
        self.comparison_dir = output_dir / "comparisons"
        self.plot_style = plot_style

        plt.rcParams.update(plot_style)

    # ============================================================
    # INTERSECTION – SINGLE ARCHETYPE
    # ============================================================
    def plot_intersection_comparison(
        self,
        combined_data: pd.DataFrame,
        row_info: Dict,
        archetype: str,
        mse_avg_sim: float,
        mse_max_sim: float,
        actual_start: pd.Timestamp,
        actual_end: pd.Timestamp,
        output_label: str
    ) -> None:

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            combined_data.index,
            combined_data["archetype_internal_temperature"],
            label="T_sim",
            color="#00AA00",
            linewidth=2.0,
            alpha=0.7,
            linestyle="--",
            zorder=3,
        )

        ax.plot(
            combined_data.index,
            combined_data["actual_max_temperature"],
            label="Hobologger_max",
            color="#CC0000",
            linewidth=2.0,
            alpha=0.8,
            zorder=4,
        )

        ax.plot(
            combined_data.index,
            combined_data["actual_average_temperature"],
            label="Hobologger_avg",
            color="#0066CC",
            linewidth=1.8,
            linestyle=":",
            alpha=1.0,
            zorder=5,
        )

        ax.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{row_info['address']}\nArchetype: {archetype}\n"
            f"MSE(avg vs sim)={mse_avg_sim:.4f} | MSE(max vs sim)={mse_max_sim:.4f}",
            fontsize=12,
            fontweight="bold",
        )

        ax.legend(loc="best", fontsize=10, framealpha=0.95, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        output_path = self.comparison_dir / f"{output_label}_{archetype}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved comparison plot: {output_path.name}")

    # ============================================================
    # INTERSECTION – AVERAGED ARCHETYPES
    # ============================================================
    def plot_intersection_comparison_averaged(
        self,
        combined_data: pd.DataFrame,
        row_info: Dict,
        archetypes: str,
        mse_predicted_avg: float,
        mse_predicted_max: float,
        actual_start: pd.Timestamp,
        actual_end: pd.Timestamp,
        output_path: Path = None,
    ) -> None:

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            combined_data.index,
            combined_data["T_predicted"],
            label="T_predicted",
            color="#9370DB",
            linewidth=2.5,
            alpha=0.75,
            linestyle="--",
            zorder=3,
        )

        ax.plot(
            combined_data.index,
            combined_data["actual_max_temperature"],
            label="Hobologger_max",
            color="#CC0000",
            linewidth=2.0,
            alpha=0.8,
            zorder=4,
        )

        ax.plot(
            combined_data.index,
            combined_data["actual_average_temperature"],
            label="Hobologger_avg",
            color="#0066CC",
            linewidth=2.0,
            linestyle=":",
            alpha=1.0,
            zorder=5,
        )

        ax.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{row_info['address']}\nAveraged Archetypes: {archetypes}\n"
            f"MSE(T_predicted vs Hobologger_avg)={mse_predicted_avg:.4f} | "
            f"MSE(T_predicted vs Hobologger_max)={mse_predicted_max:.4f}",
            fontsize=12,
            fontweight="bold",
        )

        ax.legend(loc="best", fontsize=10, framealpha=0.95, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()

        if output_path is None:
            output_label = row_info.get("clean_address", "unknown")
            output_path = self.comparison_dir / f"{output_label}_averaged_prediction.png"

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved averaged prediction plot: {output_path.name}")

    # ============================================================
    # HEAT EVENT – SINGLE ARCHETYPE
    # ============================================================
    def plot_heat_event(
        self,
        combined_data: pd.DataFrame,
        address: str,
        archetype: str,
        event_name: str,
        event_id: str,
        output_path: Path,
    ) -> None:

        fig, ax = plt.subplots(figsize=(14, 6))

        if "archetype_internal_temperature" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["archetype_internal_temperature"],
                label="T_sim",
                color="#00AA00",
                linewidth=2.0,
                alpha=0.7,
                linestyle="--",
                zorder=3,
            )

        if "actual_max_temperature" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["actual_max_temperature"],
                label="Hobologger_max",
                color="#CC0000",
                linewidth=2.0,
                alpha=0.8,
                zorder=4,
            )

        if "actual_average_temperature" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["actual_average_temperature"],
                label="Hobologger_avg",
                color="#0066CC",
                linewidth=1.8,
                linestyle=":",
                alpha=1.0,
                zorder=5,
            )

        ax.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{address}\n{event_name} ({event_id}) - Archetype: {archetype}",
            fontsize=12,
            fontweight="bold",
        )

        ax.legend(loc="best", fontsize=10, framealpha=0.95, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ============================================================
    # HEAT EVENT – AVERAGED ARCHETYPES
    # ============================================================
    def plot_heat_event_averaged(
        self,
        combined_data: pd.DataFrame,
        address: str,
        archetypes: str,
        event_name: str,
        event_id: str,
        output_path: Path,
    ) -> None:

        fig, ax = plt.subplots(figsize=(14, 6))

        archetype_list = [a.strip() for a in archetypes.split(",")]
        colors = ["#90EE90", "#FFD700", "#87CEEB", "#FFB6C1"]

        for i, arch in enumerate(archetype_list):
            col_name = f"{arch}_temp"
            if col_name in combined_data.columns:
                ax.plot(
                    combined_data.index,
                    combined_data[col_name],
                    label=f"{arch} (individual)",
                    color=colors[i % len(colors)],
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.4,
                    zorder=1,
                )

        if "T_predicted" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["T_predicted"],
                label=f"T_predicted (avg: {archetypes})",
                color="#9370DB",
                linewidth=2.5,
                alpha=0.75,
                linestyle="--",
                marker="^",
                markersize=2,
                markevery=8,
                zorder=3,
            )

        if "actual_max_temperature" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["actual_max_temperature"],
                label="Hobologger_max",
                color="#CC0000",
                linewidth=2.2,
                alpha=0.85,
                linestyle="-",
                marker="s",
                markersize=2,
                markevery=8,
                zorder=4,
            )

        if "actual_average_temperature" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["actual_average_temperature"],
                label="Hobologger_avg",
                color="#0066CC",
                linewidth=2.0,
                linestyle=":",
                marker="o",
                markersize=2.0,
                markevery=8,
                alpha=1.0,
                zorder=5,
            )

        ax.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{address}\n{event_name} ({event_id}) - Averaged Simulation\n"
            f"Archetypes: {archetypes}",
            fontsize=12,
            fontweight="bold",
        )

        ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.95, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ============================================================
    # PERIOD INTERSECTION – AVERAGED ARCHETYPES
    # ============================================================
    def plot_period_intersection(
        self,
        combined_data: pd.DataFrame,
        address: str,
        archetypes: str,
        period_start: pd.Timestamp,
        period_end: pd.Timestamp,
        output_path: Path,
    ) -> None:
        """
        Plot full period intersection showing HOBO data vs averaged archetype predictions.
        
        Args:
            combined_data: DataFrame with HOBO and prediction data
            address: House address for title
            archetypes: Comma-separated archetype names
            period_start: Start of intersection period
            period_end: End of intersection period
            output_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot individual archetype temperatures with light colors
        archetype_list = [a.strip() for a in archetypes.split(",")]
        colors = ["#90EE90", "#FFD700", "#87CEEB", "#FFB6C1"]

        for i, arch in enumerate(archetype_list):
            col_name = f"{arch}_temp"
            if col_name in combined_data.columns:
                ax.plot(
                    combined_data.index,
                    combined_data[col_name],
                    label=f"{arch} (individual)",
                    color=colors[i % len(colors)],
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.4,
                    zorder=1,
                )

        # Plot averaged prediction
        if "T_predicted" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["T_predicted"],
                label=f"T_predicted (avg: {archetypes})",
                color="#9370DB",
                linewidth=2.5,
                alpha=0.75,
                linestyle="--",
                zorder=3,
            )

        # Plot HOBO max temperature
        if "actual_max_temperature" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["actual_max_temperature"],
                label="Hobologger_max",
                color="#CC0000",
                linewidth=2.0,
                alpha=0.8,
                zorder=4,
            )

        # Plot HOBO average temperature
        if "actual_average_temperature" in combined_data.columns:
            ax.plot(
                combined_data.index,
                combined_data["actual_average_temperature"],
                label="Hobologger_avg",
                color="#0066CC",
                linewidth=2.0,
                linestyle=":",
                alpha=1.0,
                zorder=5,
            )

        ax.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{address}\nPeriod Intersection: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}\n"
            f"Averaged Archetypes: {archetypes}",
            fontsize=12,
            fontweight="bold",
        )

        ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.95, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved period intersection plot: {output_path.name}")

    # ============================================================
    # AC VS NO-AC COMPARISON
    # ============================================================
    def plot_ac_vs_noac(
        self,
        ac_series: pd.Series,
        noac_series: pd.Series,
        title: str,
        output_path: Path,
    ) -> None:
        """
        Plot comparison between AC and No-AC temperature scenarios.
        
        Args:
            ac_series: Temperature series with AC
            noac_series: Temperature series without AC
            title: Plot title (usually address - archetype)
            output_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot AC series
        if ac_series is not None and not ac_series.empty:
            ax.plot(
                ac_series.index,
                ac_series.values,
                label="With AC",
                color="#0066CC",
                linewidth=2.0,
                alpha=0.8,
                linestyle="-",
                zorder=3,
            )

        # Plot No-AC series
        if noac_series is not None and not noac_series.empty:
            ax.plot(
                noac_series.index,
                noac_series.values,
                label="Without AC",
                color="#CC0000",
                linewidth=2.0,
                alpha=0.8,
                linestyle="--",
                zorder=4,
            )

        # Calculate and display statistics if both series exist
        if ac_series is not None and noac_series is not None and not ac_series.empty and not noac_series.empty:
            # Align series for comparison
            aligned = pd.DataFrame({"AC": ac_series, "NoAC": noac_series}).dropna()
            if not aligned.empty:
                mean_diff = aligned["NoAC"].mean() - aligned["AC"].mean()
                max_diff = aligned["NoAC"].max() - aligned["AC"].max()
                
                # Add text box with statistics
                textstr = f'Mean Diff: {mean_diff:.2f}°C\nMax Diff: {max_diff:.2f}°C'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)

        ax.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{title}\nAC vs No-AC Temperature Comparison",
            fontsize=12,
            fontweight="bold",
        )

        ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved AC vs No-AC plot: {output_path.name}")
