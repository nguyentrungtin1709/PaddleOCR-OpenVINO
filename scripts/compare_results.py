"""
Compare Results Script.

This script compares benchmark results between OpenVINO and PaddleOCR pipelines,
generates summary statistics and visualization charts.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --openvino-dir output/openvino --paddle-dir output/paddle
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare benchmark results between OpenVINO and PaddleOCR pipelines"
    )
    
    parser.add_argument(
        "--openvino-dir",
        type=str,
        default="output/openvino",
        help="Directory containing OpenVINO results (default: output/openvino)"
    )
    
    parser.add_argument(
        "--paddle-dir",
        type=str,
        default="output/paddle",
        help="Directory containing PaddleOCR results (default: output/paddle)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save comparison results (default: output)"
    )
    
    return parser.parse_args()


def load_openvino_details(details_path: Path) -> pd.DataFrame:
    """
    Load OpenVINO pipeline details.csv.
    
    Expected columns: filename, image_width, image_height, num_regions,
                      detection_ms, recognition_ms, total_ms, results
    """
    if not details_path.exists():
        raise FileNotFoundError(f"OpenVINO details not found: {details_path}")
    
    df = pd.read_csv(details_path)
    df["pipeline"] = "OpenVINO"
    
    # Ensure numeric columns
    df["total_ms"] = pd.to_numeric(df["total_ms"], errors="coerce")
    if "detection_ms" in df.columns:
        df["detection_ms"] = pd.to_numeric(df["detection_ms"], errors="coerce")
    if "recognition_ms" in df.columns:
        df["recognition_ms"] = pd.to_numeric(df["recognition_ms"], errors="coerce")
    
    return df


def load_paddle_details(details_path: Path) -> pd.DataFrame:
    """
    Load PaddleOCR pipeline details.csv.
    
    Expected columns: filename, image_width, image_height, num_regions, total_ms, results
    """
    if not details_path.exists():
        raise FileNotFoundError(f"PaddleOCR details not found: {details_path}")
    
    df = pd.read_csv(details_path)
    df["pipeline"] = "PaddleOCR"
    
    # Ensure numeric columns
    df["total_ms"] = pd.to_numeric(df["total_ms"], errors="coerce")
    
    return df


def calculate_statistics(df: pd.DataFrame, pipeline_name: str) -> dict:
    """Calculate statistics for a pipeline."""
    total_times = df["total_ms"].dropna()
    
    stats = {
        "pipeline": pipeline_name,
        "num_images": len(df),
        "avg_ms": total_times.mean(),
        "min_ms": total_times.min(),
        "max_ms": total_times.max(),
        "std_ms": total_times.std(),
        "median_ms": total_times.median(),
        "p95_ms": total_times.quantile(0.95),
        "p99_ms": total_times.quantile(0.99),
        "throughput_fps": 1000 / total_times.mean() if total_times.mean() > 0 else 0,
        "total_time_s": total_times.sum() / 1000
    }
    
    # Add detection/recognition breakdown for OpenVINO
    if "detection_ms" in df.columns and "recognition_ms" in df.columns:
        det_times = df["detection_ms"].dropna()
        rec_times = df["recognition_ms"].dropna()
        stats["avg_detection_ms"] = det_times.mean()
        stats["avg_recognition_ms"] = rec_times.mean()
    
    return stats


def generate_summary_csv(
    openvino_stats: dict,
    paddle_stats: dict,
    output_path: Path
) -> None:
    """Generate summary.csv with comparison statistics."""
    
    # Calculate speedup
    speedup = paddle_stats["avg_ms"] / openvino_stats["avg_ms"] if openvino_stats["avg_ms"] > 0 else 0
    
    summary_data = [
        ["Metric", "OpenVINO", "PaddleOCR (MKL-DNN)", "Speedup"],
        ["Number of Images", openvino_stats["num_images"], paddle_stats["num_images"], "-"],
        ["Average Time (ms)", f"{openvino_stats['avg_ms']:.2f}", f"{paddle_stats['avg_ms']:.2f}", f"{speedup:.2f}x"],
        ["Median Time (ms)", f"{openvino_stats['median_ms']:.2f}", f"{paddle_stats['median_ms']:.2f}", "-"],
        ["Min Time (ms)", f"{openvino_stats['min_ms']:.2f}", f"{paddle_stats['min_ms']:.2f}", "-"],
        ["Max Time (ms)", f"{openvino_stats['max_ms']:.2f}", f"{paddle_stats['max_ms']:.2f}", "-"],
        ["Std Dev (ms)", f"{openvino_stats['std_ms']:.2f}", f"{paddle_stats['std_ms']:.2f}", "-"],
        ["P95 Time (ms)", f"{openvino_stats['p95_ms']:.2f}", f"{paddle_stats['p95_ms']:.2f}", "-"],
        ["P99 Time (ms)", f"{openvino_stats['p99_ms']:.2f}", f"{paddle_stats['p99_ms']:.2f}", "-"],
        ["Throughput (FPS)", f"{openvino_stats['throughput_fps']:.2f}", f"{paddle_stats['throughput_fps']:.2f}", f"{speedup:.2f}x"],
        ["Total Time (s)", f"{openvino_stats['total_time_s']:.2f}", f"{paddle_stats['total_time_s']:.2f}", "-"],
    ]
    
    # Add detection/recognition breakdown for OpenVINO
    if "avg_detection_ms" in openvino_stats:
        summary_data.append([
            "Avg Detection (ms)", 
            f"{openvino_stats['avg_detection_ms']:.2f}", 
            "-", 
            "-"
        ])
        summary_data.append([
            "Avg Recognition (ms)", 
            f"{openvino_stats['avg_recognition_ms']:.2f}", 
            "-", 
            "-"
        ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(summary_data)
    
    logger.info(f"Summary saved to: {output_path}")


def generate_bar_chart(
    openvino_stats: dict,
    paddle_stats: dict,
    output_path: Path
) -> None:
    """Generate bar chart comparing pipeline performance."""
    
    # Set up the figure with a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors
    colors = ['#2E86AB', '#A23B72']  # Blue for OpenVINO, Pink/Purple for PaddleOCR
    labels = ['OpenVINO', 'PaddleOCR\n(MKL-DNN)']
    
    # Chart 1: Average Processing Time
    ax1 = axes[0]
    times = [openvino_stats['avg_ms'], paddle_stats['avg_ms']]
    bars1 = ax1.bar(labels, times, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Average Processing Time per Image', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(times) * 1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                 f'{val:.1f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Chart 2: Throughput (FPS)
    ax2 = axes[1]
    fps = [openvino_stats['throughput_fps'], paddle_stats['throughput_fps']]
    bars2 = ax2.bar(labels, fps, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Images per Second (FPS)', fontsize=12)
    ax2.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(fps) * 1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars2, fps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps)*0.02,
                 f'{val:.2f} FPS', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Chart 3: Time Distribution (Box-like representation with min/avg/max)
    ax3 = axes[2]
    
    # Create grouped bar chart for min/avg/max
    x = np.arange(3)  # min, avg, max
    width = 0.35
    
    openvino_vals = [openvino_stats['min_ms'], openvino_stats['avg_ms'], openvino_stats['max_ms']]
    paddle_vals = [paddle_stats['min_ms'], paddle_stats['avg_ms'], paddle_stats['max_ms']]
    
    bars3a = ax3.bar(x - width/2, openvino_vals, width, label='OpenVINO', color=colors[0], edgecolor='black', linewidth=1.2)
    bars3b = ax3.bar(x + width/2, paddle_vals, width, label='PaddleOCR', color=colors[1], edgecolor='black', linewidth=1.2)
    
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_title('Time Distribution (Min/Avg/Max)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Min', 'Average', 'Max'])
    ax3.legend(loc='upper left')
    
    # Calculate and display speedup
    speedup = paddle_stats['avg_ms'] / openvino_stats['avg_ms'] if openvino_stats['avg_ms'] > 0 else 0
    
    # Add speedup annotation
    fig.text(0.5, 0.02, 
             f'OpenVINO is {speedup:.2f}x faster than PaddleOCR (MKL-DNN) on average',
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Chart saved to: {output_path}")


def generate_detailed_chart(
    openvino_df: pd.DataFrame,
    paddle_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate detailed per-image comparison chart."""
    
    # Merge dataframes on filename
    merged = pd.merge(
        openvino_df[['filename', 'total_ms']].rename(columns={'total_ms': 'openvino_ms'}),
        paddle_df[['filename', 'total_ms']].rename(columns={'total_ms': 'paddle_ms'}),
        on='filename',
        how='inner'
    )
    
    if len(merged) == 0:
        logger.warning("No matching filenames found for detailed comparison")
        return
    
    # Sort by filename for consistent ordering
    merged = merged.sort_values('filename').reset_index(drop=True)
    
    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sample if too many images
    if len(merged) > 50:
        # Take every nth sample
        step = len(merged) // 50
        merged_sample = merged.iloc[::step].reset_index(drop=True)
    else:
        merged_sample = merged
    
    x = np.arange(len(merged_sample))
    width = 0.4
    
    colors = ['#2E86AB', '#A23B72']
    
    ax.bar(x - width/2, merged_sample['openvino_ms'], width, label='OpenVINO', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, merged_sample['paddle_ms'], width, label='PaddleOCR', color=colors[1], alpha=0.8)
    
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.set_xlabel('Image Index', fontsize=12)
    ax.set_title('Per-Image Processing Time Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add horizontal lines for averages
    ax.axhline(y=merged['openvino_ms'].mean(), color=colors[0], linestyle='--', linewidth=2, label='OpenVINO Avg')
    ax.axhline(y=merged['paddle_ms'].mean(), color=colors[1], linestyle='--', linewidth=2, label='PaddleOCR Avg')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Detailed chart saved to: {output_path}")


def print_summary(openvino_stats: dict, paddle_stats: dict) -> None:
    """Print summary to console."""
    speedup = paddle_stats['avg_ms'] / openvino_stats['avg_ms'] if openvino_stats['avg_ms'] > 0 else 0
    
    print("\n" + "=" * 70)
    print(" BENCHMARK COMPARISON SUMMARY ".center(70, "="))
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'OpenVINO':>15} {'PaddleOCR':>15} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Number of Images':<30} {openvino_stats['num_images']:>15} {paddle_stats['num_images']:>15} {'-':>10}")
    print(f"{'Average Time (ms)':<30} {openvino_stats['avg_ms']:>15.2f} {paddle_stats['avg_ms']:>15.2f} {speedup:>9.2f}x")
    print(f"{'Median Time (ms)':<30} {openvino_stats['median_ms']:>15.2f} {paddle_stats['median_ms']:>15.2f} {'-':>10}")
    print(f"{'Min Time (ms)':<30} {openvino_stats['min_ms']:>15.2f} {paddle_stats['min_ms']:>15.2f} {'-':>10}")
    print(f"{'Max Time (ms)':<30} {openvino_stats['max_ms']:>15.2f} {paddle_stats['max_ms']:>15.2f} {'-':>10}")
    print(f"{'Std Dev (ms)':<30} {openvino_stats['std_ms']:>15.2f} {paddle_stats['std_ms']:>15.2f} {'-':>10}")
    print(f"{'Throughput (FPS)':<30} {openvino_stats['throughput_fps']:>15.2f} {paddle_stats['throughput_fps']:>15.2f} {speedup:>9.2f}x")
    print(f"{'Total Time (s)':<30} {openvino_stats['total_time_s']:>15.2f} {paddle_stats['total_time_s']:>15.2f} {'-':>10}")
    print("-" * 70)
    print()
    
    if speedup > 1:
        print(f"OpenVINO is {speedup:.2f}x FASTER than PaddleOCR (MKL-DNN)")
    else:
        print(f"PaddleOCR is {1/speedup:.2f}x faster than OpenVINO")
    
    print("=" * 70)
    print()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Resolve paths
    openvino_dir = Path(args.openvino_dir)
    paddle_dir = Path(args.paddle_dir)
    output_dir = Path(args.output_dir)
    
    if not openvino_dir.is_absolute():
        openvino_dir = PROJECT_ROOT / openvino_dir
    if not paddle_dir.is_absolute():
        paddle_dir = PROJECT_ROOT / paddle_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    logger.info(f"OpenVINO results: {openvino_dir}")
    logger.info(f"PaddleOCR results: {paddle_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load data
        openvino_df = load_openvino_details(openvino_dir / "details.csv")
        paddle_df = load_paddle_details(paddle_dir / "details.csv")
        
        logger.info(f"Loaded {len(openvino_df)} OpenVINO results")
        logger.info(f"Loaded {len(paddle_df)} PaddleOCR results")
        
        # Calculate statistics
        openvino_stats = calculate_statistics(openvino_df, "OpenVINO")
        paddle_stats = calculate_statistics(paddle_df, "PaddleOCR")
        
        # Generate outputs
        generate_summary_csv(openvino_stats, paddle_stats, output_dir / "summary.csv")
        generate_bar_chart(openvino_stats, paddle_stats, output_dir / "summary_chart.png")
        generate_detailed_chart(openvino_df, paddle_df, output_dir / "detailed_chart.png")
        
        # Print summary
        print_summary(openvino_stats, paddle_stats)
        
        logger.info("Comparison completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
