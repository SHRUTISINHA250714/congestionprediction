# ============================================================================
# ML-Based Congestion Control - Automated Pipeline
# File: run_pipeline.py
# Description: Automates the complete ML pipeline execution
# ============================================================================

import os
import sys
import subprocess
import time
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_step(step_num, text):
    """Print step number and description."""
    print(f"{Colors.OKCYAN}{Colors.BOLD}[STEP {step_num}]{Colors.ENDC} {text}")

def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def run_command(command, cwd=None, description=""):
    """Run a shell command and return success status."""
    try:
        print(f"  Running: {description if description else command}")
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_success(f"{description if description else 'Command'} completed")
            return True, result.stdout
        else:
            print_error(f"{description if description else 'Command'} failed")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print_error(f"{description if description else 'Command'} timed out")
        return False, "Timeout"
    except Exception as e:
        print_error(f"Exception: {str(e)}")
        return False, str(e)

def check_dependencies():
    """Check if required Python packages are installed."""
    print_step(0, "Checking Dependencies")
    
    required = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print_success(f"{package} installed")
        except ImportError:
            missing.append(package)
            print_warning(f"{package} not installed")
    
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False
    
    print_success("All dependencies installed")
    return True

def run_ns2_simulation(project_root):
    """Run NS2 simulation if NS2 is installed."""
    print_step(1, "Running NS2 Simulation")
    
    ns2_dir = os.path.join(project_root, 'ns2')
    topology_file = os.path.join(ns2_dir, 'topology.tcl')
    trace_file = os.path.join(ns2_dir, 'out.tr')
    
    # Check if NS2 is available
    success, _ = run_command('ns -version', description="Check NS2 installation")
    
    if not success:
        print_warning("NS2 not installed - will use synthetic data")
        return True  # Continue anyway
    
    # Run simulation
    success, _ = run_command(
        f'ns {topology_file}',
        cwd=ns2_dir,
        description="NS2 simulation"
    )
    
    if success and os.path.exists(trace_file):
        file_size = os.path.getsize(trace_file)
        print_success(f"Trace file generated: {file_size/1024:.2f} KB")
        return True
    else:
        print_warning("NS2 simulation failed - will use synthetic data")
        return True  # Continue with synthetic data

def run_feature_engineering(project_root):
    """Run feature engineering script."""
    print_step(2, "Feature Engineering")
    
    scripts_dir = os.path.join(project_root, 'scripts')
    dataset_file = os.path.join(project_root, 'dataset', 'congestion_dataset.csv')
    
    success, output = run_command(
        'python feature_engineering.py',
        cwd=scripts_dir,
        description="Feature engineering"
    )
    
    if success and os.path.exists(dataset_file):
        # Count lines in dataset
        with open(dataset_file, 'r') as f:
            line_count = sum(1 for _ in f) - 1  # Exclude header
        print_success(f"Dataset created: {line_count} samples")
        return True
    else:
        print_error("Feature engineering failed")
        return False

def run_model_training(project_root):
    """Run model training script."""
    print_step(3, "Training ML Models")
    
    scripts_dir = os.path.join(project_root, 'scripts')
    models_dir = os.path.join(project_root, 'models')
    
    success, output = run_command(
        'python train_models.py',
        cwd=scripts_dir,
        description="Model training"
    )
    
    if success:
        # Check if models were created
        pkl_files = list(Path(models_dir).glob('*.pkl'))
        print_success(f"Models trained: {len(pkl_files)} files saved")
        
        # Extract accuracy from output
        if 'Accuracy' in output:
            print("  Model Results:")
            for line in output.split('\n'):
                if 'Accuracy' in line or 'F1-Score' in line:
                    print(f"    {line.strip()}")
        return True
    else:
        print_error("Model training failed")
        return False

def run_model_evaluation(project_root):
    """Run model evaluation script."""
    print_step(4, "Evaluating Models")
    
    scripts_dir = os.path.join(project_root, 'scripts')
    results_dir = os.path.join(project_root, 'results')
    
    success, output = run_command(
        'python evaluate_models.py',
        cwd=scripts_dir,
        description="Model evaluation"
    )
    
    if success:
        # Check if results were created
        png_files = list(Path(results_dir).glob('*.png'))
        txt_files = list(Path(results_dir).glob('*.txt'))
        print_success(f"Visualizations created: {len(png_files)} PNG, {len(txt_files)} TXT")
        return True
    else:
        print_error("Model evaluation failed")
        return False

def test_prediction(project_root):
    """Test prediction system."""
    print_step(5, "Testing Prediction System")
    
    scripts_dir = os.path.join(project_root, 'scripts')
    
    # Test with sample metrics
    test_code = """
from predict import CongestionPredictor, create_sample_metrics

predictor = CongestionPredictor('../models')
predictor.load_model('Random Forest')

metrics = create_sample_metrics()
results = predictor.predict_and_recommend(metrics)

print(f"Congestion Probability: {results['congestion_probability']}%")
print(f"Action: {results['action']}")
"""
    
    # Write test script
    test_file = os.path.join(scripts_dir, '_test_predict.py')
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    success, output = run_command(
        f'python _test_predict.py',
        cwd=scripts_dir,
        description="Prediction test"
    )
    
    # Clean up test file
    if os.path.exists(test_file):
        os.remove(test_file)
    
    if success and 'Congestion Probability' in output:
        print_success("Prediction system working")
        print(f"  {output.strip()}")
        return True
    else:
        print_error("Prediction test failed")
        return False

def generate_summary(project_root, results):
    """Generate execution summary."""
    print_header("PIPELINE EXECUTION SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print(f"Total Steps: {total}")
    print(f"Passed: {Colors.OKGREEN}{passed}{Colors.ENDC}")
    print(f"Failed: {Colors.FAIL}{total - passed}{Colors.ENDC}")
    print()
    
    for step, success in results.items():
        status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if success else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
        print(f"  {step}: {status}")
    
    print()
    
    if passed == total:
        print_success("ALL STEPS COMPLETED SUCCESSFULLY! 🎉")
        print("\nNext steps:")
        print("  1. Check results/ folder for visualizations")
        print("  2. Read model_metrics.txt for detailed performance")
        print("  3. Run 'python scripts/predict.py' for interactive prediction")
        print("  4. Review README.md for complete documentation")
    else:
        print_error("Some steps failed. Please check errors above.")

def main():
    """Main pipeline execution."""
    print_header("ML-BASED CONGESTION CONTROL - AUTOMATED PIPELINE")
    
    # Get project root
    project_root = Path(__file__).parent.absolute()
    print(f"Project Root: {project_root}\n")
    
    start_time = time.time()
    results = {}
    
    # Execute pipeline steps
    results['Dependencies'] = check_dependencies()
    
    if not results['Dependencies']:
        print_error("Please install dependencies first:")
        print("  pip install -r requirements.txt")
        return
    
    results['NS2 Simulation'] = run_ns2_simulation(project_root)
    results['Feature Engineering'] = run_feature_engineering(project_root)
    
    if not results['Feature Engineering']:
        print_error("Pipeline stopped due to feature engineering failure")
        return
    
    results['Model Training'] = run_model_training(project_root)
    
    if not results['Model Training']:
        print_error("Pipeline stopped due to training failure")
        return
    
    results['Model Evaluation'] = run_model_evaluation(project_root)
    results['Prediction Test'] = test_prediction(project_root)
    
    # Generate summary
    elapsed_time = time.time() - start_time
    generate_summary(project_root, results)
    
    print(f"\nTotal Execution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Unexpected error: {str(e)}{Colors.ENDC}")
        sys.exit(1)
