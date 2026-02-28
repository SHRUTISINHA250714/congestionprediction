"""
============================================================================
Final Year Project: ML-Based Predictive Congestion Control Using NS2
File: run_pipeline.py
Description: Master script orchestrating the execution of the entire project
============================================================================
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")

def run_script(script_name, cwd=None):
    print(f"\n[EXEC] Running {script_name}...")
    cmd = f"{sys.executable} {script_name}"
    
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Executing {script_name} failed. Aborting pipeline.")
        sys.exit(1)
    else:
        print(f"[SUCCESS] {script_name} completed successfully.")

def run_ns2(topology_name, ns2_dir):
    print(f"\n[EXEC] Running NS2 Simulation for {topology_name}...")
    ns_cmd = 'ns'
    if os.name == 'nt':
        ns_cmd = 'wsl ns'
        
    cmd = f"{ns_cmd} {topology_name}"
    result = subprocess.run(cmd, shell=True, cwd=ns2_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] NS2 simulation {topology_name} failed. Aborting pipeline.")
        sys.exit(1)
    else:
        print(f"[SUCCESS] NS2 simulation {topology_name} completed.")

def main():
    print_header("MACHINE LEARNING PREDICTIVE CONGESTION CONTROL - AUTOMATED PIPELINE")
    
    start_time = time.time()
    project_root = Path(__file__).parent.absolute()
    scripts_dir = os.path.join(project_root, 'scripts')
    ns2_dir = os.path.join(project_root, 'ns2')
    
    # 1. Feature Engineering (Generates Data via internal NS2 calls)
    run_script("feature_engineering.py", cwd=scripts_dir)
    
    # 2. Model Training
    run_script("train_models.py", cwd=scripts_dir)
    
    # 3. Model Evaluation (Visualizations)
    run_script("evaluate_models.py", cwd=scripts_dir)
    
    # 4. Run Baseline topology directly
    run_ns2("topology_baseline.tcl", ns2_dir)
    
    # 5. ML Routing Simulation (Generates ml_topology.tcl & runs it)
    run_script("routing_controller.py", cwd=scripts_dir)
    
    # 6. Performance Comparison
    run_script("performance_comparison.py", cwd=scripts_dir)
    
    elapsed = time.time() - start_time
    
    print_header("PIPELINE EXECUTION COMPLETE")
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print("\nRESULTS GENERATED:")
    print("  Dataset      -> dataset/congestion_dataset.csv")
    print("  Models       -> models/*.pkl")
    print("  Metrics      -> results/model_comparison.csv, results/performance_comparison.csv")
    print("  Graphs       -> results/*.png")
    print("  Simulations  -> ns2/baseline.tr, ns2/ml.tr")
    print("  Animations   -> ns2/baseline.nam, ns2/ml.nam")

if __name__ == "__main__":
    main()