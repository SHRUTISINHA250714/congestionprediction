# ============================================================================
# Performance Comparison Script
# File: compare_performance.py
# Description: Compares baseline (without ML) vs ML-based simulation
#              Analyzes packet drops, delay, throughput, and generates reports
# ============================================================================

import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class TraceAnalyzer:
    """Analyzes NS2 trace files and extracts performance metrics."""
    
    def __init__(self, trace_file):
        """
        Initialize trace analyzer.
        
        Args:
            trace_file: Path to NS2 trace file (.tr)
        """
        self.trace_file = Path(trace_file)
        self.events = []
        self.metrics = {}
        
    def parse_trace(self):
        """
        Parse NS2 trace file.
        
        NS2 trace format:
        event time from to pkt_type size flags flow_id src dst seq_num packet_id
        
        Events:
        + : packet enqueued
        - : packet dequeued
        r : packet received
        d : packet dropped
        """
        if not self.trace_file.exists():
            print(f"✗ Trace file not found: {self.trace_file}")
            return False
        
        print(f"Parsing trace file: {self.trace_file.name}")
        
        with open(self.trace_file, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 11:
                    continue
                
                event = {
                    'action': parts[0],       # +, -, r, d
                    'time': float(parts[1]),
                    'src': parts[2],
                    'dst': parts[3],
                    'type': parts[4],         # tcp, cbr, ack
                    'size': int(parts[5]),
                    'flags': parts[6],
                    'flow_id': parts[7],
                    'src_addr': parts[8],
                    'dst_addr': parts[9],
                    'seq_num': parts[10] if len(parts) > 10 else "0"
                }
                
                self.events.append(event)
        
        print(f"  Parsed {len(self.events)} events")
        return True
    
    def calculate_metrics(self):
        """Calculate performance metrics from parsed trace."""
        
        # Initialize counters
        sent_packets = 0
        received_packets = 0
        dropped_packets = 0
        total_delay = 0.0
        delay_count = 0
        
        # Track packet send times for delay calculation
        packet_send_time = {}  # {(src, dst, seq): send_time}
        
        # Track throughput over time
        throughput_data = defaultdict(int)  # {time_window: bytes}
        
        # Analyze events
        for event in self.events:
            action = event['action']
            time = event['time']
            size = event['size']
            
            # Count sent packets (enqueued at source)
            if action == '+':
                sent_packets += 1
                # Track send time for this packet
                key = (event['src_addr'], event['dst_addr'], event['seq_num'])
                packet_send_time[key] = time
            
            # Count received packets
            elif action == 'r':
                received_packets += 1
                
                # Calculate delay
                key = (event['src_addr'], event['dst_addr'], event['seq_num'])
                if key in packet_send_time:
                    delay = time - packet_send_time[key]
                    total_delay += delay
                    delay_count += 1
                
                # Track throughput (bytes per second)
                time_window = int(time)  # 1-second windows
                throughput_data[time_window] += size
            
            # Count dropped packets
            elif action == 'd':
                dropped_packets += 1
        
        # Calculate metrics
        packet_loss_rate = (dropped_packets / sent_packets * 100) if sent_packets > 0 else 0
        avg_delay = (total_delay / delay_count * 1000) if delay_count > 0 else 0  # Convert to ms
        delivery_rate = (received_packets / sent_packets * 100) if sent_packets > 0 else 0
        
        # Calculate throughput (average Mbps)
        if throughput_data:
            avg_throughput_bytes = np.mean(list(throughput_data.values()))
            avg_throughput_mbps = (avg_throughput_bytes * 8) / 1_000_000  # Convert to Mbps
        else:
            avg_throughput_mbps = 0
        
        self.metrics = {
            'sent_packets': sent_packets,
            'received_packets': received_packets,
            'dropped_packets': dropped_packets,
            'packet_loss_rate': packet_loss_rate,
            'avg_delay_ms': avg_delay,
            'delivery_rate': delivery_rate,
            'avg_throughput_mbps': avg_throughput_mbps,
            'throughput_over_time': dict(throughput_data)
        }
        
        return self.metrics
    
    def print_metrics(self, title="Performance Metrics"):
        """Print metrics in formatted table."""
        print("\n" + "="*60)
        print(title.center(60))
        print("="*60)
        print(f"  Total Packets Sent:      {self.metrics['sent_packets']:>10}")
        print(f"  Packets Received:        {self.metrics['received_packets']:>10}")
        print(f"  Packets Dropped:         {self.metrics['dropped_packets']:>10}")
        print(f"  Packet Loss Rate:        {self.metrics['packet_loss_rate']:>9.2f}%")
        print(f"  Delivery Rate:           {self.metrics['delivery_rate']:>9.2f}%")
        print(f"  Average Delay:           {self.metrics['avg_delay_ms']:>9.2f} ms")
        print(f"  Average Throughput:      {self.metrics['avg_throughput_mbps']:>9.2f} Mbps")
        print("="*60 + "\n")


class PerformanceComparator:
    """Compares baseline vs ML-based simulation performance."""
    
    def __init__(self, baseline_trace, ml_trace, results_dir):
        """
        Initialize comparator.
        
        Args:
            baseline_trace: Path to baseline trace file
            ml_trace: Path to ML trace file
            results_dir: Directory to save results
        """
        self.baseline_analyzer = TraceAnalyzer(baseline_trace)
        self.ml_analyzer = TraceAnalyzer(ml_trace)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def analyze_both(self):
        """Analyze both trace files."""
        print("\n" + "="*60)
        print(" ANALYZING SIMULATION TRACES".center(60))
        print("="*60 + "\n")
        
        # Analyze baseline
        print("1. Analyzing BASELINE simulation (without ML)...")
        if not self.baseline_analyzer.parse_trace():
            return False
        self.baseline_analyzer.calculate_metrics()
        self.baseline_analyzer.print_metrics("BASELINE Performance (Without ML)")
        
        # Analyze ML-based
        print("2. Analyzing ML-BASED simulation (with ML)...")
        if not self.ml_analyzer.parse_trace():
            return False
        self.ml_analyzer.calculate_metrics()
        self.ml_analyzer.print_metrics("ML-BASED Performance (With ML)")
        
        return True
    
    def calculate_improvements(self):
        """Calculate percentage improvements."""
        baseline = self.baseline_analyzer.metrics
        ml = self.ml_analyzer.metrics
        
        improvements = {}
        
        # For dropped packets and loss rate, reduction is good
        if baseline['dropped_packets'] > 0:
            drop_reduction = ((baseline['dropped_packets'] - ml['dropped_packets']) / 
                             baseline['dropped_packets'] * 100)
        else:
            drop_reduction = 0
        
        if baseline['packet_loss_rate'] > 0:
            loss_reduction = ((baseline['packet_loss_rate'] - ml['packet_loss_rate']) / 
                             baseline['packet_loss_rate'] * 100)
        else:
            loss_reduction = 0
        
        # For delay, reduction is good
        if baseline['avg_delay_ms'] > 0:
            delay_reduction = ((baseline['avg_delay_ms'] - ml['avg_delay_ms']) / 
                              baseline['avg_delay_ms'] * 100)
        else:
            delay_reduction = 0
        
        # For throughput and delivery rate, increase is good
        if baseline['avg_throughput_mbps'] > 0:
            throughput_improvement = ((ml['avg_throughput_mbps'] - baseline['avg_throughput_mbps']) / 
                                     baseline['avg_throughput_mbps'] * 100)
        else:
            throughput_improvement = 0
        
        if baseline['delivery_rate'] > 0:
            delivery_improvement = ((ml['delivery_rate'] - baseline['delivery_rate']) / 
                                   baseline['delivery_rate'] * 100)
        else:
            delivery_improvement = 0
        
        improvements = {
            'drop_reduction': drop_reduction,
            'loss_reduction': loss_reduction,
            'delay_reduction': delay_reduction,
            'throughput_improvement': throughput_improvement,
            'delivery_improvement': delivery_improvement
        }
        
        return improvements
    
    def print_comparison(self):
        """Print detailed comparison with improvements."""
        improvements = self.calculate_improvements()
        
        print("\n" + "="*60)
        print(" PERFORMANCE COMPARISON: BASELINE vs ML".center(60))
        print("="*60 + "\n")
        
        baseline = self.baseline_analyzer.metrics
        ml = self.ml_analyzer.metrics
        
        print(f"{'Metric':<25} {'Baseline':>12} {'ML-Based':>12} {'Improvement':>12}")
        print("-" * 60)
        
        print(f"{'Packets Dropped':<25} {baseline['dropped_packets']:>12} "
              f"{ml['dropped_packets']:>12} "
              f"{improvements['drop_reduction']:>10.1f}%")
        
        print(f"{'Packet Loss Rate':<25} {baseline['packet_loss_rate']:>11.2f}% "
              f"{ml['packet_loss_rate']:>11.2f}% "
              f"{improvements['loss_reduction']:>10.1f}%")
        
        print(f"{'Average Delay (ms)':<25} {baseline['avg_delay_ms']:>11.2f}  "
              f"{ml['avg_delay_ms']:>11.2f}  "
              f"{improvements['delay_reduction']:>10.1f}%")
        
        print(f"{'Throughput (Mbps)':<25} {baseline['avg_throughput_mbps']:>11.2f}  "
              f"{ml['avg_throughput_mbps']:>11.2f}  "
              f"{improvements['throughput_improvement']:>10.1f}%")
        
        print(f"{'Delivery Rate':<25} {baseline['delivery_rate']:>11.2f}% "
              f"{ml['delivery_rate']:>11.2f}% "
              f"{improvements['delivery_improvement']:>10.1f}%")
        
        print("="*60 + "\n")
        
        # Overall assessment
        avg_improvement = np.mean([
            improvements['drop_reduction'],
            improvements['loss_reduction'],
            improvements['delay_reduction']
        ])
        
        print("OVERALL ASSESSMENT:")
        if avg_improvement > 10:
            print("  ✓ ML-based approach shows SIGNIFICANT improvement")
        elif avg_improvement > 5:
            print("  ✓ ML-based approach shows MODERATE improvement")
        elif avg_improvement > 0:
            print("  ✓ ML-based approach shows SLIGHT improvement")
        else:
            print("  ⚠ ML-based approach shows NO improvement")
        
        print(f"  Average improvement: {avg_improvement:.1f}%\n")
        
        return improvements
    
    def generate_comparison_chart(self):
        """Generate bar chart comparing baseline vs ML metrics."""
        baseline = self.baseline_analyzer.metrics
        ml = self.ml_analyzer.metrics
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Baseline vs ML-Based Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Packet Drops
        ax1 = axes[0, 0]
        metrics = ['Baseline', 'ML-Based']
        values = [baseline['dropped_packets'], ml['dropped_packets']]
        colors = ['#FF6B6B', '#4ECDC4']
        bars1 = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Packets Dropped', fontweight='bold')
        ax1.set_title('Packet Drops')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Packet Loss Rate
        ax2 = axes[0, 1]
        values = [baseline['packet_loss_rate'], ml['packet_loss_rate']]
        bars2 = ax2.bar(metrics, values, color=colors, alpha=0.8)
        ax2.set_ylabel('Packet Loss Rate (%)', fontweight='bold')
        ax2.set_title('Packet Loss Rate')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. Average Delay
        ax3 = axes[1, 0]
        values = [baseline['avg_delay_ms'], ml['avg_delay_ms']]
        bars3 = ax3.bar(metrics, values, color=colors, alpha=0.8)
        ax3.set_ylabel('Average Delay (ms)', fontweight='bold')
        ax3.set_title('Average Delay')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. Throughput
        ax4 = axes[1, 1]
        values = [baseline['avg_throughput_mbps'], ml['avg_throughput_mbps']]
        colors_reversed = ['#FF6B6B', '#51CF66']  # Green is good for throughput
        bars4 = ax4.bar(metrics, values, color=colors_reversed, alpha=0.8)
        ax4.set_ylabel('Throughput (Mbps)', fontweight='bold')
        ax4.set_title('Average Throughput')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.results_dir / 'performance_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison chart saved: {output_file}")
        
        # Show plot
        plt.show()
        
    def save_results_csv(self):
        """Save comparison results to CSV file."""
        baseline = self.baseline_analyzer.metrics
        ml = self.ml_analyzer.metrics
        improvements = self.calculate_improvements()
        
        # Create DataFrame
        data = {
            'Metric': [
                'Packets Sent',
                'Packets Received',
                'Packets Dropped',
                'Packet Loss Rate (%)',
                'Average Delay (ms)',
                'Delivery Rate (%)',
                'Throughput (Mbps)'
            ],
            'Baseline': [
                baseline['sent_packets'],
                baseline['received_packets'],
                baseline['dropped_packets'],
                baseline['packet_loss_rate'],
                baseline['avg_delay_ms'],
                baseline['delivery_rate'],
                baseline['avg_throughput_mbps']
            ],
            'ML_Based': [
                ml['sent_packets'],
                ml['received_packets'],
                ml['dropped_packets'],
                ml['packet_loss_rate'],
                ml['avg_delay_ms'],
                ml['delivery_rate'],
                ml['avg_throughput_mbps']
            ],
            'Improvement (%)': [
                'N/A',
                'N/A',
                improvements['drop_reduction'],
                improvements['loss_reduction'],
                improvements['delay_reduction'],
                improvements['delivery_improvement'],
                improvements['throughput_improvement']
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_file = self.results_dir / 'performance_comparison.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Results saved to CSV: {output_file}\n")


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print(" BASELINE vs ML PERFORMANCE COMPARISON".center(60))
    print("="*60 + "\n")
    
    # Setup paths
    project_root = Path(__file__).parent
    ns2_dir = project_root / 'ns2'
    results_dir = project_root / 'results'
    
    baseline_trace = ns2_dir / 'baseline.tr'
    ml_trace = ns2_dir / 'ml.tr'
    
    # Check if trace files exist
    if not baseline_trace.exists():
        print(f"✗ Baseline trace file not found: {baseline_trace}")
        print("  Run: cd ns2 && ns topology_baseline.tcl")
        print("\nFor testing without NS2, run with --synthetic flag:")
        print("  python compare_performance.py --synthetic")
        return
    
    if not ml_trace.exists():
        print(f"✗ ML trace file not found: {ml_trace}")
        print("  Run: python apply_ml_routing.py")
        print("  Then: cd ns2 && ns topology_ml_modified.tcl")
        return
    
    # Create comparator
    comparator = PerformanceComparator(baseline_trace, ml_trace, results_dir)
    
    # Analyze traces
    if not comparator.analyze_both():
        print("✗ Failed to analyze traces")
        return
    
    # Print comparison
    comparator.print_comparison()
    
    # Generate visualizations
    print("Generating comparison charts...")
    comparator.generate_comparison_chart()
    
    # Save results
    comparator.save_results_csv()
    
    print("="*60)
    print(" COMPARISON COMPLETED".center(60))
    print("="*60)
    print(f"\nResults saved in: {results_dir}")
    print("  - performance_comparison.png (chart)")
    print("  - performance_comparison.csv (data)")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Handle synthetic data mode for testing without NS2
    if '--synthetic' in sys.argv:
        print("\n⚠ SYNTHETIC MODE: Generating sample trace files for testing\n")
        
        project_root = Path(__file__).parent
        ns2_dir = project_root / 'ns2'
        ns2_dir.mkdir(exist_ok=True)
        
        # Create synthetic baseline trace (high drops)
        with open(ns2_dir / 'baseline.tr', 'w') as f:
            for i in range(1000):
                # Simulate sent packets
                f.write(f"+ {i*0.01:.2f} 0 1 tcp {1000} ------- 0 0.0 1.0 {i}\n")
                # Simulate some drops (20% drop rate)
                if i % 5 == 0:
                    f.write(f"d {i*0.01+0.02:.2f} 1 2 tcp {1000} ------- 0 0.0 1.0 {i}\n")
                else:
                    # Simulate received packets with delay
                    f.write(f"r {i*0.01+0.05:.2f} 2 3 tcp {1000} ------- 0 0.0 1.0 {i}\n")
        
        # Create synthetic ML trace (fewer drops)
        with open(ns2_dir / 'ml.tr', 'w') as f:
            for i in range(1000):
                # Simulate sent packets
                f.write(f"+ {i*0.01:.2f} 0 1 tcp {1000} ------- 0 0.0 1.0 {i}\n")
                # Simulate reduced drops (5% drop rate)
                if i % 20 == 0:
                    f.write(f"d {i*0.01+0.02:.2f} 1 2 tcp {1000} ------- 0 0.0 1.0 {i}\n")
                else:
                    # Simulate received packets with lower delay
                    f.write(f"r {i*0.01+0.03:.2f} 2 3 tcp {1000} ------- 0 0.0 1.0 {i}\n")
        
        print("✓ Synthetic trace files created\n")
    
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
