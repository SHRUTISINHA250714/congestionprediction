"""
============================================================================
Final Year Project: ML-Based Predictive Congestion Control Using NS2
File: feature_engineering.py
Description: Transforms parsed metrics into ML features with TRUE temporal prediction
============================================================================
"""

import pandas as pd
import numpy as np
from trace_parser import TraceParser
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineer:
    def __init__(self, metrics_list):
        self.df = pd.DataFrame(metrics_list)
        self.congestion_threshold_util = 85.0
        self.congestion_threshold_loss = 10.0
        self.congestion_threshold_delay = 0.08
        
    def create_features(self):
        print("[INFO] Creating engineered features...")
        
        self._clean_dataset()
        self._label_congestion()
        self._create_temporal_features()
        self._implement_temporal_shift()
        self._filter_columns()
        
        # Drop NaN values created by lagging/shifting
        initial_size = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        dropped = initial_size - len(self.df)
        
        self._check_for_leakage()
        self._remove_zero_variance()
        
        print(f"[SUCCESS] Created {len(self.df.columns)-1} features (excluding label)")
        print(f"[INFO] Dropped {dropped} rows due to NaN from temporal operations")
        print(f"[SUCCESS] Final dataset size: {len(self.df)} samples")
        
        return self.df
        
    def _clean_dataset(self):
        print("[INFO] Cleaning dataset...")
        initial_size = len(self.df)
        
        # Sort by time window
        self.df = self.df.sort_values('time_window').reset_index(drop=True)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        
        final_size = len(self.df)
        print(f"[SUCCESS] Dataset cleaned. Final size: {final_size}")
        
    def _label_congestion(self):
        self.df['congestion'] = (
            (self.df['link_utilization'] > self.congestion_threshold_util) |
            (self.df['packet_loss_rate'] > self.congestion_threshold_loss) |
            (self.df['avg_delay'] > self.congestion_threshold_delay)
        ).astype(int)
        
        congested = self.df['congestion'].sum()
        non_congested = len(self.df) - congested
        
        print(f"[INFO] Initial congestion distribution before filtering:")
        print(f"  Congested samples: {congested} ({congested/max(1, len(self.df))*100:.2f}%)")
        print(f"  Non-congested samples: {non_congested} ({non_congested/max(1, len(self.df))*100:.2f}%)")
        
        self._remove_extreme_simulations()
        
    def _remove_extreme_simulations(self):
        if 'sim_id' not in self.df.columns:
            return
            
        initial_sims = self.df['sim_id'].nunique()
        sim_stats = self.df.groupby('sim_id')['congestion'].mean()
        
        # Keep simulations that have both congested and non-congested windows
        valid_sims = sim_stats[(sim_stats > 0.0) & (sim_stats < 1.0)].index
        
        self.df = self.df[self.df['sim_id'].isin(valid_sims)].reset_index(drop=True)
        dropped_sims = initial_sims - len(valid_sims)
        
        print(f"[INFO] Dropped {dropped_sims} extreme simulations (100% or 0% congested)")
        print(f"[INFO] Retained {len(valid_sims)} valid simulations for balanced dataset")
    
    def _create_temporal_features(self):
        # Create lag 1 features
        self.df['throughput_lag1'] = self.df['throughput'].shift(1)
        self.df['delay_lag1'] = self.df['avg_delay'].shift(1)
        self.df['loss_lag1'] = self.df['packet_loss_rate'].shift(1)
        self.df['utilization_lag1'] = self.df['link_utilization'].shift(1)
        
        # Rolling statistics
        self.df['rolling_mean_util'] = self.df['link_utilization'].rolling(window=3, min_periods=1).mean().shift(1)
        
        # Rate of change
        self.df['utilization_change'] = self.df['link_utilization'].diff().shift(1)
        
    def _implement_temporal_shift(self):
        # Shift label backward (Label(t) = congestion at t+1)
        self.df['congestion'] = self.df['congestion'].shift(-1)

    def _filter_columns(self):
        # KEEP ONLY specified features to avoid data leakage
        keep_cols = [
            'throughput_lag1', 'delay_lag1', 'loss_lag1', 'utilization_lag1',
            'rolling_mean_util', 'utilization_change', 'congestion'
        ]
        
        # Drop everything else
        drop_cols = [c for c in self.df.columns if c not in keep_cols]
        if drop_cols:
            self.df = self.df.drop(columns=drop_cols)
            
    def _remove_zero_variance(self):
        zero_var_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1 and col != 'congestion']
        if zero_var_cols:
            self.df = self.df.drop(columns=zero_var_cols)
    
    def _check_for_leakage(self):
        feature_cols = [col for col in self.df.columns if col != 'congestion']
        
        correlations = {}
        for col in feature_cols:
            try:
                corr = self.df[col].corr(self.df['congestion'])
                if abs(corr) > 0.95:
                    correlations[col] = corr
            except:
                pass
        
        if correlations:
            for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                self.df = self.df.drop(columns=[feature])
            print(f"[SUCCESS] Removed {len(correlations)} highly correlated features.")
            
        self._plot_correlation_heatmap()
    
    def _plot_correlation_heatmap(self):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            corr_matrix = self.df[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'feature_correlation_heatmap.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            pass
    
    def save_dataset(self, output_file):
        # Final class balance check
        congested = self.df['congestion'].sum()
        total_samples = len(self.df)
        non_congested = total_samples - congested
        
        ratio_congested = congested / max(1, total_samples)
        ratio_non_congested = non_congested / max(1, total_samples)
        
        print(f"\n[INFO] Final Dataset Summary:")
        print(f"  Total samples: {total_samples}")
        print(f"  Congested samples: {congested} ({ratio_congested*100:.2f}%)")
        print(f"  Non-congested samples: {non_congested} ({ratio_non_congested*100:.2f}%)")
        
        if ratio_congested > 0.8 or ratio_congested < 0.2:
            print("[ERROR] Dataset imbalance detected. Adjust simulation parameters.")
            sys.exit(1)
            
        if total_samples < 300:
            print("[ERROR] Less than 300 samples generated after filtering. Aborting pipeline.")
            sys.exit(1)
        elif total_samples < 500:
            print("[WARNING] Low sample size. Consider increasing simulations.")
             
        self.df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Dataset saved to: {output_file}")


def generate_multiple_simulations(ns2_dir, num_simulations=20):
    print(f"\n[INFO] Generating {num_simulations} simulations for balanced dataset...")
    all_metrics = []
    
    import subprocess
    ns_cmd = 'ns'
    if os.name == 'nt':
        ns_cmd = 'wsl ns'
        
    for i in range(num_simulations):
        print(f"  Running simulation {i+1}/{num_simulations}...")
        topology_file = os.path.join(ns2_dir, 'topology.tcl')
        trace_file = os.path.join(ns2_dir, 'out.tr')
        
        if i < 6:
            category_name = "LOW"
            bw = np.round(np.random.uniform(2.5, 3.0), 1)
        elif i < 13:
            category_name = "MEDIUM"
            bw = np.round(np.random.uniform(1.5, 2.0), 1)
        else:
            category_name = "HIGH"
            bw = np.round(np.random.uniform(0.5, 1.0), 1)
            
        os.environ["LOAD_TYPE"] = category_name
        
        cmd = f"{ns_cmd} {os.path.basename(topology_file)} {bw}"
        result = subprocess.run(cmd, cwd=ns2_dir, shell=True, capture_output=True, text=True, timeout=90)
        
        if result.returncode == 0 and os.path.exists(trace_file):
            parser = TraceParser(trace_file, link_capacity_mbps=bw)
            metrics = parser.parse(window_size=1.0)
            
            # Tag with sim_id
            for metric in metrics:
                metric['sim_id'] = i
            
            all_metrics.extend(metrics)
            
    if not all_metrics:
        print("[WARNING] NS2 simulations failed or NS2 is not installed.")
        print("[INFO] Falling back to pre-existing trace data (out.tr) for realistic data...")
        if os.path.exists(trace_file):
            parser = TraceParser(trace_file, link_capacity_mbps=0.5)
            metrics = parser.parse(window_size=1.0)
            for metric in metrics:
                metric['sim_id'] = 0
            all_metrics = metrics
            
    return all_metrics


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ns2_dir = os.path.join(project_root, 'ns2')
    output_file = os.path.join(project_root, 'dataset', 'congestion_dataset.csv')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    metrics = generate_multiple_simulations(ns2_dir, num_simulations=20)
    
    if len(metrics) < 300:
        print("[WARNING] Not enough samples generated. Might lead to poor ML performance.")
        
    engineer = FeatureEngineer(metrics)
    dataset = engineer.create_features()
    engineer.save_dataset(output_file)

if __name__ == "__main__":
    main()
