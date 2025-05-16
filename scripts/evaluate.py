#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def setup_results_dir():
    """
    Create results directory structure
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    base_dir = Path("results")
    run_dir = base_dir / timestamp
    plots_dir = run_dir / "plots"
    data_dir = run_dir / "data"
    profiling_dir = run_dir / "profiling"
    
    # Create all directories
    for dir_path in [base_dir, run_dir, plots_dir, data_dir, profiling_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return run_dir, plots_dir, data_dir, profiling_dir

def run_spmv(matrix_path, valgrind=False):
    """
    Run SpMV program with the given matrix and return performance metrics
    """
    cmd = ['./build/spmv', matrix_path]
    if valgrind:
        cmd = ['valgrind', '--tool=cachegrind'] + cmd
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running SpMV: {e}")
        return None

def parse_output(output):
    """
    Parse the JSON output to extract performance metrics
    """
    try:
        # Extract JSON part between markers
        json_start = output.find('JSON_START\n') + len('JSON_START\n')
        json_end = output.find('JSON_END')
        if json_start == -1 or json_end == -1:
            print("Could not find JSON markers in output")
            print(f"Raw output: {output}")
            return None
            
        json_str = output[json_start:json_end]
        data = json.loads(json_str)
        
        metrics = {
            'rows': data['matrix']['rows'],
            'cols': data['matrix']['cols'],
            'nnz': data['matrix']['nnz'],
            'cpu_time': data['cpu']['time_us'],
            'cpu_gflops': data['cpu']['gflops'],
            'gpu_time': data['gpu']['time_us'],
            'gpu_gflops': data['gpu']['gflops'],
            'results_match': data['validation']['results_match']
        }
        
        # Calculate memory throughput (GB/s)
        # For CSR format: values (8 bytes) + col_indices (4 bytes) + row_offsets (4 bytes)
        bytes_accessed = metrics['nnz'] * (8 + 4) + (metrics['rows'] + 1) * 4
        metrics['cpu_bandwidth'] = bytes_accessed / (metrics['cpu_time'] * 1e-6) / 1e9
        metrics['gpu_bandwidth'] = bytes_accessed / (metrics['gpu_time'] * 1e-6) / 1e9
        
        return metrics
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        print(f"Raw output: {output}")
        return None
    except KeyError as e:
        print(f"Missing key in JSON output: {e}")
        print(f"Raw output: {output}")
        return None

def run_valgrind_analysis(matrix_path, profiling_dir):
    """
    Run valgrind analysis on CPU implementation
    """
    cachegrind_out = profiling_dir / "cachegrind.out"
    cmd = ['valgrind', '--tool=cachegrind', f'--cachegrind-out-file={cachegrind_out}', './build/spmv', matrix_path]
    subprocess.run(cmd)
    # Parse cachegrind output
    cmd = ['cg_annotate', str(cachegrind_out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def plot_results(results_df, plots_dir):
    """
    Generate performance plots
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [10, 6]
    
    # Plot execution time
    plt.figure()
    plt.plot(results_df['nnz'], results_df['cpu_time'], 'o-', label='CPU')
    plt.plot(results_df['nnz'], results_df['gpu_time'], 'o-', label='GPU')
    plt.xlabel('Number of Non-zeros')
    plt.ylabel('Execution Time (μs)')
    plt.title('SpMV Performance: CPU vs GPU')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(plots_dir / 'execution_time.png')
    plt.close()
    
    # Plot GFLOPS
    plt.figure()
    plt.plot(results_df['nnz'], results_df['cpu_gflops'], 'o-', label='CPU')
    plt.plot(results_df['nnz'], results_df['gpu_gflops'], 'o-', label='GPU')
    plt.xlabel('Number of Non-zeros')
    plt.ylabel('GFLOPS')
    plt.title('SpMV Performance: GFLOPS')
    plt.legend()
    plt.xscale('log')
    plt.savefig(plots_dir / 'gflops.png')
    plt.close()
    
    # Plot Memory Bandwidth
    plt.figure()
    plt.plot(results_df['nnz'], results_df['cpu_bandwidth'], 'o-', label='CPU')
    plt.plot(results_df['nnz'], results_df['gpu_bandwidth'], 'o-', label='GPU')
    plt.xlabel('Number of Non-zeros')
    plt.ylabel('Memory Bandwidth (GB/s)')
    plt.title('SpMV Performance: Memory Bandwidth')
    plt.legend()
    plt.xscale('log')
    plt.savefig(plots_dir / 'memory_bandwidth.png')
    plt.close()

def main():
    # Setup results directory
    run_dir, plots_dir, data_dir, profiling_dir = setup_results_dir()
    print(f"Results will be saved in: {run_dir}")
    
    # List of test matrices
    matrices = [
        'matrices/rdist1.mtx',
        'matrices/ss.mtx'
    ]
    
    # Collect results
    results = []
    for matrix in matrices:
        print(f"Testing matrix: {matrix}")
        metrics = run_spmv(matrix)
        if metrics:
            # Add matrix name to metrics
            metrics['matrix_name'] = Path(matrix).name
            results.append(metrics)
            if not metrics['results_match']:
                print(f"Warning: Results do not match for {matrix}")
    
    if not results:
        print("No results collected. Exiting.")
        return
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(data_dir / 'results.csv', index=False)
    
    # Save raw results as JSON for future reference
    with open(data_dir / 'raw_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_results(df, plots_dir)
    
    # Run Valgrind analysis on a smaller matrix
    print("\nRunning Valgrind analysis...")
    valgrind_output = run_valgrind_analysis(matrices[0], profiling_dir)  # Use smallest matrix
    with open(profiling_dir / 'valgrind_analysis.txt', 'w') as f:
        f.write(valgrind_output)
    
    # Create a summary file
    with open(run_dir / 'summary.txt', 'w') as f:
        f.write(f"SpMV Performance Analysis\n")
        f.write(f"=======================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Matrices tested:\n")
        for matrix in matrices:
            f.write(f"- {matrix}\n")
        f.write(f"\nResults saved in:\n")
        f.write(f"- CSV data: {data_dir/'results.csv'}\n")
        f.write(f"- Plots: {plots_dir}\n")
        f.write(f"- Profiling: {profiling_dir}\n")

if __name__ == '__main__':
    main() 