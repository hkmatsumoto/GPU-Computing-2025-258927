#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def setup_results_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path("results")
    eval_dir = base_dir / f"evaluation_{timestamp}"
    plots_dir = eval_dir / "plots"
    analysis_dir = eval_dir / "analysis"
    
    for dir_path in [base_dir, eval_dir, plots_dir, analysis_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return eval_dir, plots_dir, analysis_dir

def load_benchmark_data():
    csv_path = Path("results/consolidated_benchmark_results.csv")
    json_path = Path("results/consolidated_benchmark_results.json")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    return df, json_data

def plot_performance_comparison(df, plots_dir):
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    
    matrices = df['Matrix'].unique()
    selected_kernels = ['Basic (D1)', 'Shared Memory', 'Warp Reduce', 'cuSPARSE']
    plt.figure(figsize=(14, 8))
    for kernel in selected_kernels:
        kernel_data = df[df['Kernel'] == kernel]
        
        if not kernel_data.empty:
            kernel_data_sorted = kernel_data.sort_values('NNZ')
            plt.plot(kernel_data_sorted['NNZ'], kernel_data_sorted['GFLOPS'], 
                    label=kernel, marker='o', markersize=8, linewidth=2, alpha=0.8)
    
    plt.xlabel('Number of Non-zeros (NNZ)')
    plt.ylabel('GFLOPS')
    plt.title('SpMV Performance: GFLOPS vs Matrix Size')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'gflops_vs_nnz.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 8))
    for kernel in selected_kernels:
        kernel_data = df[df['Kernel'] == kernel]
        
        if not kernel_data.empty:
            kernel_data_sorted = kernel_data.sort_values('NNZ')
            plt.plot(kernel_data_sorted['NNZ'], kernel_data_sorted['Bandwidth_GB_s'], 
                    label=kernel, marker='o', markersize=8, linewidth=2, alpha=0.8)
    
    plt.xlabel('Number of Non-zeros (NNZ)')
    plt.ylabel('Memory Bandwidth (GB/s)')
    plt.title('SpMV Performance: Memory Bandwidth vs Matrix Size')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'bandwidth_vs_nnz.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 8))
    for kernel in selected_kernels:
        kernel_data = df[df['Kernel'] == kernel]
        
        if not kernel_data.empty:
            kernel_data_sorted = kernel_data.sort_values('NNZ')
            plt.plot(kernel_data_sorted['NNZ'], kernel_data_sorted['Time_ms'], 
                    label=kernel, marker='o', markersize=8, linewidth=2, alpha=0.8)
    
    plt.xlabel('Number of Non-zeros (NNZ)')
    plt.ylabel('Execution Time (ms)')
    plt.title('SpMV Performance: Execution Time vs Matrix Size')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'time_vs_nnz.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_speedup_comparison(df, plots_dir):
    selected_kernels = ['Basic (D1)', 'Shared Memory', 'Warp Reduce', 'cuSPARSE']
    matrices = df['Matrix'].unique()
    
    # Calculate speedups relative to Basic (D1) kernel
    speedup_data = []
    for matrix in matrices:
        matrix_data = df[df['Matrix'] == matrix]
        basic_time = matrix_data[matrix_data['Kernel'] == 'Basic (D1)']['Time_ms'].iloc[0]
        
        for _, row in matrix_data.iterrows():
            speedup = basic_time / row['Time_ms']
            speedup_data.append({
                'Matrix': matrix,
                'Kernel': row['Kernel'],
                'Speedup': speedup,
                'Verification': row['Verification'],
                'GFLOPS': row['GFLOPS']
            })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Create speedup bar plot
    plt.figure(figsize=(16, 10))
    
    # Select specific kernels for cleaner visualization
    selected_kernels = ['Basic (D1)', 'Shared Memory', 'Warp Reduce', 'cuSPARSE']
    
    # Set up bar positions
    x = np.arange(len(matrices))
    width = 0.15
    offset = -(len(selected_kernels) - 1) * width / 2
    
    for i, kernel in enumerate(selected_kernels):
        kernel_speedups = []
        for matrix in matrices:
            kernel_data = speedup_df[(speedup_df['Matrix'] == matrix) & 
                                   (speedup_df['Kernel'] == kernel)]
            if not kernel_data.empty:
                speedup = kernel_data['Speedup'].iloc[0]
                kernel_speedups.append(speedup)
            else:
                kernel_speedups.append(0)
        
        bars = plt.bar(x + offset + i * width, kernel_speedups, width, 
                      label=kernel, alpha=0.8)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}x', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Matrix')
    plt.ylabel('Speedup (relative to Basic kernel)')
    plt.title('SpMV Kernel Performance: Speedup Comparison')
    plt.xticks(x, matrices, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Vector Warp comparison function removed - focusing on 4 main kernels only

def plot_verification_heatmap(df, plots_dir):
    verification_matrix = df.pivot(index='Kernel', columns='Matrix', values='Verification')
    
    # Convert to numeric (1 for PASS, 0 for FAIL)
    verification_numeric = verification_matrix.replace({'PASS': 1, 'FAIL': 0})
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(verification_numeric, annot=True, cmap='RdYlGn', 
                cbar_kws={'label': 'Verification Status'}, 
                fmt='d', linewidths=0.5)
    plt.title('Kernel Verification Status Across Matrices')
    plt.xlabel('Matrix')
    plt.ylabel('Kernel')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / 'verification_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_vs_cusparse(df, plots_dir):
    matrices = df['Matrix'].unique()
    kernels = df['Kernel'].unique()
    relative_performance = []
    for matrix in matrices:
        matrix_data = df[df['Matrix'] == matrix]
        cusparse_gflops = matrix_data[matrix_data['Kernel'] == 'cuSPARSE']['GFLOPS'].iloc[0]
        
        for _, row in matrix_data.iterrows():
            if row['Kernel'] != 'cuSPARSE':
                relative_perf = row['GFLOPS'] / cusparse_gflops
                relative_performance.append({
                    'Matrix': matrix,
                    'Kernel': row['Kernel'],
                    'Relative_Performance': relative_perf,
                    'Verification': row['Verification']
                })
    
    rel_perf_df = pd.DataFrame(relative_performance)
    
    # Plot performance gap to cuSPARSE
    plt.figure(figsize=(14, 8))
    
    # Select specific kernels for cleaner visualization
    top_kernels = ['Basic (D1)', 'Shared Memory', 'Warp Reduce']
    
    for kernel in top_kernels:
        kernel_data = rel_perf_df[rel_perf_df['Kernel'] == kernel]
        
        if not kernel_data.empty:
            matrix_order = [matrices.tolist().index(m) for m in kernel_data['Matrix']]
            plt.scatter(matrix_order, kernel_data['Relative_Performance'], 
                       label=kernel, s=100, alpha=0.8)
    
    plt.axhline(y=1.0, linestyle='--', alpha=0.7, label='cuSPARSE baseline')
    plt.xlabel('Matrix')
    plt.ylabel('Performance relative to cuSPARSE')
    plt.title('Custom Kernels Performance vs cuSPARSE')
    plt.xticks(range(len(matrices)), matrices, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_vs_cusparse.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(df, json_data, analysis_dir):
    with open(analysis_dir / 'performance_analysis.txt', 'w') as f:
        f.write("SpMV Kernel Performance Analysis - Deliverable 2\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # GPU Information
        gpu_info = json_data['benchmark_metadata']
        f.write("Hardware Configuration:\n")
        f.write(f"- GPU: {gpu_info['gpu']}\n")
        f.write(f"- Memory: {gpu_info['memory']}\n")
        f.write(f"- Multiprocessors: {gpu_info['multiprocessors']}\n")
        f.write(f"- Max Threads/Block: {gpu_info['max_threads_per_block']}\n")
        f.write(f"- Max Shared Memory/Block: {gpu_info['max_shared_memory_per_block']}\n\n")
        
        # Matrix statistics
        f.write("Matrix Characteristics:\n")
        matrices = df['Matrix'].unique()
        for matrix in matrices:
            matrix_data = df[df['Matrix'] == matrix].iloc[0]
            f.write(f"- {matrix}: {matrix_data['Rows']:,} x {matrix_data['Cols']:,}, ")
            f.write(f"NNZ: {matrix_data['NNZ']:,}, ")
            f.write(f"Density: {matrix_data['Density_Percent']:.3f}%\n")
        f.write("\n")
        
        # Top performers by matrix
        f.write("Top Performing Kernels by Matrix:\n")
        for matrix in matrices:
            matrix_data = df[df['Matrix'] == matrix]
            passed_data = matrix_data[matrix_data['Verification'] == 'PASS']
            if not passed_data.empty:
                best = passed_data.loc[passed_data['GFLOPS'].idxmax()]
                f.write(f"- {matrix}: {best['Kernel']} ({best['GFLOPS']:.2f} GFLOPS)\n")
        f.write("\n")
        
        # Kernel Performance Summary
        f.write("Performance Summary by Kernel:\n")
        for kernel in ['Basic (D1)', 'Shared Memory', 'Warp Reduce']:
            kernel_data = df[df['Kernel'] == kernel]
            if not kernel_data.empty:
                avg_gflops = kernel_data['GFLOPS'].mean()
                max_gflops = kernel_data['GFLOPS'].max()
                f.write(f"- {kernel}: Avg={avg_gflops:.2f}, Max={max_gflops:.2f} GFLOPS\n")
        f.write("\n")
        
        # Verification issues
        f.write("Verification Issues:\n")
        failed_data = df[df['Verification'] == 'FAIL']
        if not failed_data.empty:
            for matrix in matrices:
                matrix_failures = failed_data[failed_data['Matrix'] == matrix]
                if not matrix_failures.empty:
                    f.write(f"- {matrix}: {list(matrix_failures['Kernel'])}\n")
        else:
            f.write("- No verification failures detected\n")

def main():
    # Setup results directory
    eval_dir, plots_dir, analysis_dir = setup_results_dir()
    print(f"Evaluation results will be saved in: {eval_dir}")
    
    try:
        # Load benchmark data
        print("Loading benchmark data...")
        df, json_data = load_benchmark_data()
        
        # Filter to only the 4 main kernels
        selected_kernels = ['Basic (D1)', 'Shared Memory', 'Warp Reduce', 'cuSPARSE']
        original_count = len(df)
        df = df[df['Kernel'].isin(selected_kernels)]
        print(f"Filtered to 4 main kernels: {len(df)} combinations (was {original_count})")
        
        # Generate plots
        print("Generating performance comparison plots...")
        plot_performance_comparison(df, plots_dir)
        
        print("Generating speedup comparison plots...")
        plot_speedup_comparison(df, plots_dir)
        
        print("Generating verification status heatmap...")
        plot_verification_heatmap(df, plots_dir)
        
        print("Generating performance vs cuSPARSE comparison...")
        plot_performance_vs_cusparse(df, plots_dir)
        
        # Generate analysis report
        print("Generating analysis report...")
        generate_analysis_report(df, json_data, analysis_dir)
        
        # Create summary
        with open(eval_dir / 'summary.txt', 'w') as f:
            f.write(f"SpMV Kernel Performance Evaluation - Deliverable 2\n")
            f.write(f"===================================================\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Data source: results/consolidated_benchmark_results.csv\n")
            f.write(f"Matrices tested: {', '.join(df['Matrix'].unique())}\n")
            f.write(f"Kernels evaluated: {len(df['Kernel'].unique())}\n\n")
            f.write(f"Generated files:\n")
            f.write(f"- Plots: {plots_dir}\n")
            f.write(f"- Analysis: {analysis_dir}\n")
        
        print(f"\nEvaluation complete! Results saved in: {eval_dir}")
        print(f"View plots in: {plots_dir}")
        print(f"View analysis in: {analysis_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 