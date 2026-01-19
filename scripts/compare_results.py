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
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.accuracy_evaluator import AccuracyEvaluator

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
    
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="samples",
        help="Directory containing ground truth samples (default: samples)"
    )
    
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip accuracy evaluation (only compare speed)"
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


def evaluate_accuracy(
    samples_dir: Path,
    openvino_dir: Path,
    paddle_dir: Path,
    output_dir: Path
) -> tuple:
    """
    Evaluate accuracy for both pipelines against ground truth.
    
    Args:
        samples_dir: Directory containing ground truth JSON files
        openvino_dir: Directory containing OpenVINO results
        paddle_dir: Directory containing PaddleOCR results
        output_dir: Directory to save accuracy results
        
    Returns:
        Tuple of (openvino_accuracy_df, paddle_accuracy_df, openvino_stats, paddle_stats)
    """
    evaluator = AccuracyEvaluator()
    
    # Evaluate both pipelines
    logger.info("Evaluating OpenVINO accuracy...")
    openvino_accuracy = evaluator.evaluatePipeline(samples_dir, openvino_dir, "OpenVINO")
    
    logger.info("Evaluating PaddleOCR accuracy...")
    paddle_accuracy = evaluator.evaluatePipeline(samples_dir, paddle_dir, "PaddleOCR")
    
    # Calculate summary statistics
    openvino_stats = evaluator.calculateSummaryStatistics(openvino_accuracy)
    paddle_stats = evaluator.calculateSummaryStatistics(paddle_accuracy)
    
    return openvino_accuracy, paddle_accuracy, openvino_stats, paddle_stats


def merge_accuracy_to_details(
    details_df: pd.DataFrame,
    accuracy_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge accuracy results into existing details DataFrame.
    
    Args:
        details_df: Original details DataFrame with timing info
        accuracy_df: Accuracy evaluation DataFrame
        
    Returns:
        Merged DataFrame with both timing and accuracy columns
    """
    if accuracy_df.empty:
        return details_df
    
    # Select accuracy columns (exclude duplicates like filename, pipeline)
    accuracy_cols = [col for col in accuracy_df.columns 
                     if col not in ['filename', 'pipeline']]
    
    # Merge on filename
    merged = details_df.merge(
        accuracy_df[['filename'] + accuracy_cols],
        on='filename',
        how='left'
    )
    
    return merged


def generate_accuracy_summary_csv(
    openvino_stats: Dict,
    paddle_stats: Dict,
    output_path: Path
) -> None:
    """
    Generate accuracy_summary.csv with comparison statistics.
    
    Args:
        openvino_stats: OpenVINO accuracy statistics
        paddle_stats: PaddleOCR accuracy statistics
        output_path: Path to save the CSV file
    """
    rows = [
        ["Metric", "OpenVINO", "PaddleOCR", "Winner"]
    ]
    
    # Number of images
    ov_num = openvino_stats.get('num_images', 0)
    pd_num = paddle_stats.get('num_images', 0)
    rows.append(["Number of Images", ov_num, pd_num, "-"])
    rows.append(["", "", "", ""])  # Empty row
    
    # Field-level statistics
    fields = ['color', 'productCode', 'size', 'positionQuantity']
    field_labels = {
        'color': 'COLOR',
        'productCode': 'PRODUCT CODE',
        'size': 'SIZE',
        'positionQuantity': 'POSITION/QUANTITY'
    }
    
    for field in fields:
        ov_field = openvino_stats.get('fields', {}).get(field, {})
        pd_field = paddle_stats.get('fields', {}).get(field, {})
        
        ov_exact = ov_field.get('exact_pct', 0)
        pd_exact = pd_field.get('exact_pct', 0)
        exact_winner = "OpenVINO" if ov_exact > pd_exact else ("PaddleOCR" if pd_exact > ov_exact else "Tie")
        
        ov_accept = ov_field.get('acceptable_pct', 0)
        pd_accept = pd_field.get('acceptable_pct', 0)
        accept_winner = "OpenVINO" if ov_accept > pd_accept else ("PaddleOCR" if pd_accept > ov_accept else "Tie")
        
        ov_score = ov_field.get('avg_score', 0)
        pd_score = pd_field.get('avg_score', 0)
        score_winner = "OpenVINO" if ov_score > pd_score else ("PaddleOCR" if pd_score > ov_score else "Tie")
        
        rows.append([f"=== {field_labels[field]} ===", "", "", ""])
        rows.append([f"  Exact Match (%)", f"{ov_exact:.1f}%", f"{pd_exact:.1f}%", exact_winner])
        rows.append([f"  Acceptable (>=0.90) (%)", f"{ov_accept:.1f}%", f"{pd_accept:.1f}%", accept_winner])
        rows.append([f"  Avg Score", f"{ov_score:.3f}", f"{pd_score:.3f}", score_winner])
        rows.append(["", "", "", ""])
    
    # Overall statistics
    ov_overall = openvino_stats.get('overall', {})
    pd_overall = paddle_stats.get('overall', {})
    
    ov_avg = ov_overall.get('avg_score', 0)
    pd_avg = pd_overall.get('avg_score', 0)
    avg_winner = "OpenVINO" if ov_avg > pd_avg else ("PaddleOCR" if pd_avg > ov_avg else "Tie")
    
    ov_all_exact = ov_overall.get('all_exact_pct', 0)
    pd_all_exact = pd_overall.get('all_exact_pct', 0)
    all_exact_winner = "OpenVINO" if ov_all_exact > pd_all_exact else ("PaddleOCR" if pd_all_exact > ov_all_exact else "Tie")
    
    ov_all_accept = ov_overall.get('all_acceptable_pct', 0)
    pd_all_accept = pd_overall.get('all_acceptable_pct', 0)
    all_accept_winner = "OpenVINO" if ov_all_accept > pd_all_accept else ("PaddleOCR" if pd_all_accept > ov_all_accept else "Tie")
    
    rows.append(["=== OVERALL ===", "", "", ""])
    rows.append(["  Average Score", f"{ov_avg:.3f}", f"{pd_avg:.3f}", avg_winner])
    rows.append(["  All Fields Exact (%)", f"{ov_all_exact:.1f}%", f"{pd_all_exact:.1f}%", all_exact_winner])
    rows.append(["  All Fields Acceptable (%)", f"{ov_all_accept:.1f}%", f"{pd_all_accept:.1f}%", all_accept_winner])
    
    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    logger.info(f"Accuracy summary saved to: {output_path}")


def print_accuracy_summary(openvino_stats: Dict, paddle_stats: Dict) -> None:
    """
    Print accuracy comparison summary to console.
    
    Args:
        openvino_stats: OpenVINO accuracy statistics
        paddle_stats: PaddleOCR accuracy statistics
    """
    print()
    print("=" * 75)
    print(" ACCURACY COMPARISON SUMMARY ".center(75, "="))
    print("=" * 75)
    print()
    print(f"{'Metric':<35} {'OpenVINO':>12} {'PaddleOCR':>12} {'Winner':>12}")
    print("-" * 75)
    
    # Number of images
    ov_num = openvino_stats.get('num_images', 0)
    pd_num = paddle_stats.get('num_images', 0)
    print(f"{'Images Evaluated':<35} {ov_num:>12} {pd_num:>12} {'-':>12}")
    print("=" * 75)
    
    # Field-level statistics
    fields = ['color', 'productCode', 'size', 'positionQuantity']
    field_labels = {
        'color': 'COLOR',
        'productCode': 'PRODUCT CODE',
        'size': 'SIZE',
        'positionQuantity': 'POSITION/QUANTITY'
    }
    
    for field in fields:
        ov_field = openvino_stats.get('fields', {}).get(field, {})
        pd_field = paddle_stats.get('fields', {}).get(field, {})
        
        print(f"{field_labels[field]}:")
        
        # Exact match
        ov_exact = ov_field.get('exact_pct', 0)
        pd_exact = pd_field.get('exact_pct', 0)
        winner = "OpenVINO" if ov_exact > pd_exact else ("PaddleOCR" if pd_exact > ov_exact else "Tie")
        print(f"{'  - Exact Match (%)':<35} {ov_exact:>11.1f}% {pd_exact:>11.1f}% {winner:>12}")
        
        # Acceptable
        ov_accept = ov_field.get('acceptable_pct', 0)
        pd_accept = pd_field.get('acceptable_pct', 0)
        winner = "OpenVINO" if ov_accept > pd_accept else ("PaddleOCR" if pd_accept > ov_accept else "Tie")
        print(f"{'  - Acceptable (>=0.90)':<35} {ov_accept:>11.1f}% {pd_accept:>11.1f}% {winner:>12}")
        
        # Average score
        ov_score = ov_field.get('avg_score', 0)
        pd_score = pd_field.get('avg_score', 0)
        winner = "OpenVINO" if ov_score > pd_score else ("PaddleOCR" if pd_score > ov_score else "Tie")
        print(f"{'  - Average Score':<35} {ov_score:>12.3f} {pd_score:>12.3f} {winner:>12}")
        
        print("-" * 75)
    
    # Overall
    ov_overall = openvino_stats.get('overall', {})
    pd_overall = paddle_stats.get('overall', {})
    
    print("OVERALL:")
    
    ov_avg = ov_overall.get('avg_score', 0)
    pd_avg = pd_overall.get('avg_score', 0)
    winner = "OpenVINO" if ov_avg > pd_avg else ("PaddleOCR" if pd_avg > ov_avg else "Tie")
    print(f"{'  - Average Score':<35} {ov_avg:>12.3f} {pd_avg:>12.3f} {winner:>12}")
    
    ov_all_exact = ov_overall.get('all_exact_pct', 0)
    pd_all_exact = pd_overall.get('all_exact_pct', 0)
    winner = "OpenVINO" if ov_all_exact > pd_all_exact else ("PaddleOCR" if pd_all_exact > ov_all_exact else "Tie")
    print(f"{'  - All Fields Exact':<35} {ov_all_exact:>11.1f}% {pd_all_exact:>11.1f}% {winner:>12}")
    
    ov_all_accept = ov_overall.get('all_acceptable_pct', 0)
    pd_all_accept = pd_overall.get('all_acceptable_pct', 0)
    winner = "OpenVINO" if ov_all_accept > pd_all_accept else ("PaddleOCR" if pd_all_accept > ov_all_accept else "Tie")
    print(f"{'  - All Fields Acceptable':<35} {ov_all_accept:>11.1f}% {pd_all_accept:>11.1f}% {winner:>12}")
    
    print("=" * 75)
    
    # Winner announcement
    if ov_avg > pd_avg:
        diff = (ov_avg - pd_avg) * 100
        print(f"\n>>> OpenVINO has {diff:.1f}% higher accuracy than PaddleOCR <<<")
    elif pd_avg > ov_avg:
        diff = (pd_avg - ov_avg) * 100
        print(f"\n>>> PaddleOCR has {diff:.1f}% higher accuracy than OpenVINO <<<")
    else:
        print(f"\n>>> Both pipelines have equal accuracy <<<")
    
    print()


def generate_accuracy_chart(
    openvino_stats: Dict,
    paddle_stats: Dict,
    output_path: Path
) -> None:
    """
    Generate accuracy comparison bar chart.
    
    Args:
        openvino_stats: OpenVINO accuracy statistics
        paddle_stats: PaddleOCR accuracy statistics
        output_path: Path to save the chart
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#2E86AB', '#A23B72']  # Blue for OpenVINO, Pink for PaddleOCR
    
    # Chart 1: Average Score by Field
    ax1 = axes[0]
    fields = ['color', 'productCode', 'size', 'positionQuantity']
    field_labels = ['Color', 'Product\nCode', 'Size', 'Position/\nQuantity']
    
    x = np.arange(len(fields))
    width = 0.35
    
    ov_scores = [openvino_stats.get('fields', {}).get(f, {}).get('avg_score', 0) for f in fields]
    pd_scores = [paddle_stats.get('fields', {}).get(f, {}).get('avg_score', 0) for f in fields]
    
    bars1 = ax1.bar(x - width/2, ov_scores, width, label='OpenVINO', color=colors[0], edgecolor='black')
    bars2 = ax1.bar(x + width/2, pd_scores, width, label='PaddleOCR', color=colors[1], edgecolor='black')
    
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_title('Accuracy by Field', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(field_labels)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax1.set_ylim(0, 1.2)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Overall Metrics
    ax2 = axes[1]
    metrics = ['Avg Score', 'All Exact (%)', 'All Acceptable (%)']
    
    ov_overall = openvino_stats.get('overall', {})
    pd_overall = paddle_stats.get('overall', {})
    
    # Normalize percentages to 0-1 scale for comparison
    ov_values = [
        ov_overall.get('avg_score', 0),
        ov_overall.get('all_exact_pct', 0) / 100,
        ov_overall.get('all_acceptable_pct', 0) / 100
    ]
    pd_values = [
        pd_overall.get('avg_score', 0),
        pd_overall.get('all_exact_pct', 0) / 100,
        pd_overall.get('all_acceptable_pct', 0) / 100
    ]
    
    x2 = np.arange(len(metrics))
    
    bars3 = ax2.bar(x2 - width/2, ov_values, width, label='OpenVINO', color=colors[0], edgecolor='black')
    bars4 = ax2.bar(x2 + width/2, pd_values, width, label='PaddleOCR', color=colors[1], edgecolor='black')
    
    ax2.set_ylabel('Score / Percentage', fontsize=12)
    ax2.set_title('Overall Accuracy Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax2.set_ylim(0, 1.2)
    
    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        label = f'{height:.2f}' if i == 0 else f'{height*100:.1f}%'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                label, ha='center', va='bottom', fontsize=9)
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        label = f'{height:.2f}' if i == 0 else f'{height*100:.1f}%'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                label, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Accuracy chart saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Resolve paths
    openvino_dir = Path(args.openvino_dir)
    paddle_dir = Path(args.paddle_dir)
    output_dir = Path(args.output_dir)
    samples_dir = Path(args.samples_dir)
    
    if not openvino_dir.is_absolute():
        openvino_dir = PROJECT_ROOT / openvino_dir
    if not paddle_dir.is_absolute():
        paddle_dir = PROJECT_ROOT / paddle_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not samples_dir.is_absolute():
        samples_dir = PROJECT_ROOT / samples_dir
    
    logger.info(f"OpenVINO results: {openvino_dir}")
    logger.info(f"PaddleOCR results: {paddle_dir}")
    logger.info(f"Samples directory: {samples_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load data from pipeline output (details.csv - original name)
        openvino_df = load_openvino_details(openvino_dir / "details.csv")
        paddle_df = load_paddle_details(paddle_dir / "details.csv")
        
        logger.info(f"Loaded {len(openvino_df)} OpenVINO results")
        logger.info(f"Loaded {len(paddle_df)} PaddleOCR results")
        
        # Calculate speed statistics
        openvino_stats = calculate_statistics(openvino_df, "OpenVINO")
        paddle_stats = calculate_statistics(paddle_df, "PaddleOCR")
        
        # Generate speed comparison outputs
        generate_summary_csv(openvino_stats, paddle_stats, output_dir / "speed_summary.csv")
        generate_bar_chart(openvino_stats, paddle_stats, output_dir / "speed_summary_chart.png")
        generate_detailed_chart(openvino_df, paddle_df, output_dir / "speed_detailed_chart.png")
        
        # Print speed summary
        print_summary(openvino_stats, paddle_stats)
        
        # Evaluate accuracy (unless skipped)
        if not args.skip_accuracy:
            if samples_dir.exists():
                logger.info("Starting accuracy evaluation...")
                
                # Evaluate accuracy
                ov_accuracy, pd_accuracy, ov_acc_stats, pd_acc_stats = evaluate_accuracy(
                    samples_dir, openvino_dir, paddle_dir, output_dir
                )
                
                # Merge accuracy into details and save to output dir with prefix
                if not ov_accuracy.empty:
                    openvino_merged = merge_accuracy_to_details(openvino_df, ov_accuracy)
                    openvino_merged.to_csv(output_dir / "openvino_details.csv", index=False)
                    logger.info(f"Saved openvino_details.csv to output directory")
                
                if not pd_accuracy.empty:
                    paddle_merged = merge_accuracy_to_details(paddle_df, pd_accuracy)
                    paddle_merged.to_csv(output_dir / "paddle_details.csv", index=False)
                    logger.info(f"Saved paddle_details.csv to output directory")
                
                # Generate accuracy summary
                generate_accuracy_summary_csv(ov_acc_stats, pd_acc_stats, output_dir / "accuracy_summary.csv")
                generate_accuracy_chart(ov_acc_stats, pd_acc_stats, output_dir / "accuracy_summary_chart.png")
                
                # Print accuracy summary
                print_accuracy_summary(ov_acc_stats, pd_acc_stats)
            else:
                logger.warning(f"Samples directory not found: {samples_dir}")
                logger.warning("Skipping accuracy evaluation")
        else:
            logger.info("Skipping accuracy evaluation (--skip-accuracy flag)")
        
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
