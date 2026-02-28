# ============================================================================
# ML to NS2 Integration Script
# File: apply_ml_routing.py
# Description: Uses trained ML model to predict congestion and modifies
#              NS2 topology to apply intelligent routing
# ============================================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(script_dir / 'scripts'))

from predict import CongestionPredictor, create_sample_metrics


class NS2Router:
    """Integrates ML predictions with NS2 routing configuration."""
    
    def __init__(self, model_dir, ns2_dir):
        """
        Initialize the NS2 router with ML model.
        
        Args:
            model_dir: Directory containing trained models
            ns2_dir: Directory containing NS2 topology files
        """
        self.model_dir = Path(model_dir)
        self.ns2_dir = Path(ns2_dir)
        self.predictor = CongestionPredictor(str(model_dir))
        self.congestion_threshold = 50.0  # 50% probability threshold
        
    def load_model(self, model_name='Random Forest'):
        """Load the trained ML model."""
        print(f"Loading ML model: {model_name}")
        self.predictor.load_model(model_name)
        print(f"✓ Model loaded successfully\n")
        
    def predict_congestion(self, network_metrics=None):
        """
        Predict congestion probability based on network metrics.
        
        Args:
            network_metrics: Dictionary of network metrics (optional)
                           If None, uses sample metrics
        
        Returns:
            dict: Prediction results with congestion probability and action
        """
        if network_metrics is None:
            # Use sample metrics for demonstration
            network_metrics = create_sample_metrics()
            print("Using sample network metrics for prediction")
        
        print("\n" + "="*60)
        print("NETWORK METRICS")
        print("="*60)
        for key, value in network_metrics.items():
            print(f"  {key}: {value:.2f}")
        
        # Get prediction
        results = self.predictor.predict_and_recommend(network_metrics)
        
        print("\n" + "="*60)
        print("ML PREDICTION RESULTS")
        print("="*60)
        print(f"  Congestion Probability: {results['congestion_probability']:.2f}%")
        print(f"  Prediction: {results['predicted_class']}")
        print(f"  Recommended Action: {results['action']}")
        print(f"  Confidence: {results['confidence']:.2f}%")
        print("="*60 + "\n")
        
        return results
    
    def modify_topology(self, congestion_prob, output_file='topology_ml_modified.tcl'):
        """
        Modify NS2 topology file based on congestion prediction.
        
        If congestion is predicted (probability > threshold):
        - Increase delay on primary link (make it less attractive)
        - This forces routing protocol to choose alternate path
        
        Args:
            congestion_prob: Predicted congestion probability
            output_file: Name of output topology file
        """
        input_file = self.ns2_dir / 'topology_ml.tcl'
        output_path = self.ns2_dir / output_file
        
        # Read original topology
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Determine routing modifications based on ML prediction
        if congestion_prob > self.congestion_threshold:
            # High congestion predicted - force traffic to alternate path
            # Increase delay on primary link to make it less attractive
            primary_delay = "50ms"  # Increased from 20ms
            primary_bw = "2Mb"      # Keep bandwidth same
            
            action = "APPLYING ML OPTIMIZATION"
            details = "Increasing primary link delay → Traffic reroutes to alternate path"
            
        else:
            # Low congestion - use normal configuration
            primary_delay = "20ms"
            primary_bw = "2Mb"
            
            action = "NO MODIFICATION NEEDED"
            details = "Low congestion predicted → Using standard routing"
        
        # Modify the topology file
        # Replace the placeholder values
        modified_content = content.replace(
            "set primary_delay 20ms",
            f"set primary_delay {primary_delay}"
        )
        modified_content = modified_content.replace(
            "set primary_bw 2Mb",
            f"set primary_bw {primary_bw}"
        )
        
        # Add ML prediction comments to the file
        ml_comment = f"""
# ============================================================================
# ML PREDICTION APPLIED
# ============================================================================
# Congestion Probability: {congestion_prob:.2f}%
# Threshold: {self.congestion_threshold}%
# Action: {action}
# Details: {details}
# Primary Link Configuration: {primary_bw} / {primary_delay}
# ============================================================================

"""
        modified_content = ml_comment + modified_content
        
        # Write modified topology
        with open(output_path, 'w') as f:
            f.write(modified_content)
        
        print("="*60)
        print("TOPOLOGY MODIFICATION")
        print("="*60)
        print(f"  Input file: {input_file.name}")
        print(f"  Output file: {output_path.name}")
        print(f"  Action: {action}")
        print(f"  Details: {details}")
        print(f"  Primary link: {primary_bw} / {primary_delay}")
        print("="*60 + "\n")
        
        return str(output_path)
    
    def run_simulation(self, topology_file):
        """
        Run NS2 simulation with the specified topology.
        
        Args:
            topology_file: Path to NS2 topology file
        
        Returns:
            bool: True if simulation successful, False otherwise
        """
        topology_path = Path(topology_file)
        
        if not topology_path.exists():
            print(f"✗ Topology file not found: {topology_file}")
            return False
        
        print(f"Running NS2 simulation: {topology_path.name}")
        print("="*60)
        
        # Change to NS2 directory
        original_dir = os.getcwd()
        os.chdir(self.ns2_dir)
        
        try:
            # Run NS2 simulation
            import subprocess
            result = subprocess.run(
                ['ns', topology_path.name],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✓ Simulation completed successfully")
                print(result.stdout)
                return True
            else:
                print("✗ Simulation failed")
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("✗ NS2 not installed")
            print("  Install NS2 or use synthetic trace files")
            return False
            
        except subprocess.TimeoutExpired:
            print("✗ Simulation timeout")
            return False
            
        finally:
            os.chdir(original_dir)


def create_realistic_metrics(scenario='high_traffic'):
    """
    Create realistic network metrics for different scenarios.
    
    Args:
        scenario: 'high_traffic', 'moderate_traffic', or 'low_traffic'
    
    Returns:
        dict: Network metrics
    """
    scenarios = {
        'high_traffic': {
            'avg_throughput_mbps': 1.8,      # High utilization
            'avg_delay_ms': 45.0,             # High delay
            'packet_loss_rate': 8.5,          # High loss
            'jitter_ms': 12.0,                # High jitter
            'queue_length': 85.0,             # Near full
            'bandwidth_utilization': 90.0,    # Very high
            'tcp_retransmissions': 25,        # Many retransmissions
            'active_connections': 50          # Many connections
        },
        'moderate_traffic': {
            'avg_throughput_mbps': 1.2,
            'avg_delay_ms': 25.0,
            'packet_loss_rate': 3.0,
            'jitter_ms': 6.0,
            'queue_length': 45.0,
            'bandwidth_utilization': 60.0,
            'tcp_retransmissions': 10,
            'active_connections': 25
        },
        'low_traffic': {
            'avg_throughput_mbps': 0.5,
            'avg_delay_ms': 10.0,
            'packet_loss_rate': 0.5,
            'jitter_ms': 2.0,
            'queue_length': 15.0,
            'bandwidth_utilization': 25.0,
            'tcp_retransmissions': 2,
            'active_connections': 10
        }
    }
    
    return scenarios.get(scenario, scenarios['moderate_traffic'])


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print(" ML-BASED PREDICTIVE CONGESTION CONTROL".center(60))
    print(" NS2 Integration System".center(60))
    print("="*60 + "\n")
    
    # Setup paths
    project_root = Path(__file__).parent
    model_dir = project_root / 'models'
    ns2_dir = project_root / 'ns2'
    
    # Check if models exist
    if not model_dir.exists() or not list(model_dir.glob('*.pkl')):
        print("✗ No trained models found!")
        print(f"  Expected location: {model_dir}")
        print("  Run 'python run_pipeline.py' first to train models")
        return
    
    # Initialize router
    router = NS2Router(model_dir, ns2_dir)
    
    # Load ML model
    router.load_model('Random Forest')
    
    # Scenario selection
    print("Select traffic scenario:")
    print("  1. High Traffic (likely congestion)")
    print("  2. Moderate Traffic")
    print("  3. Low Traffic (unlikely congestion)")
    print("  4. Custom metrics")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    scenario_map = {
        '1': 'high_traffic',
        '2': 'moderate_traffic',
        '3': 'low_traffic'
    }
    
    if choice in scenario_map:
        metrics = create_realistic_metrics(scenario_map[choice])
    elif choice == '4':
        metrics = None  # Will use default sample metrics
    else:
        print("Invalid choice, using high traffic scenario")
        metrics = create_realistic_metrics('high_traffic')
    
    # Predict congestion
    results = router.predict_congestion(metrics)
    
    # Modify topology based on prediction
    modified_topology = router.modify_topology(results['congestion_probability'])
    
    # Ask if user wants to run simulation
    print("\nWould you like to run the NS2 simulation? (y/n) [default: n]: ", end='')
    run_sim = input().strip().lower()
    
    if run_sim == 'y':
        router.run_simulation(modified_topology)
    else:
        print("\nSkipping NS2 simulation")
        print(f"Modified topology saved to: {modified_topology}")
        print(f"To run manually: cd {ns2_dir} && ns {Path(modified_topology).name}")
    
    print("\n" + "="*60)
    print("ML-NS2 INTEGRATION COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run baseline simulation: ns topology_baseline.tcl")
    print("  2. Run ML simulation: ns topology_ml_modified.tcl")
    print("  3. Compare results: python compare_performance.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
