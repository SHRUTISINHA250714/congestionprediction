"""
============================================================================
Final Year Project: ML-Based Predictive Congestion Control Using NS2
File: trace_parser.py
Description: Parses NS2 trace files and extracts network metrics
============================================================================
"""

import sys
from collections import defaultdict

class TraceParser:
    """
    Parses NS2 trace files and computes network congestion metrics.
    """
    
    def __init__(self, trace_file, link_capacity_mbps=0.5):
        self.trace_file = trace_file
        self.link_capacity_mbps = link_capacity_mbps
        
        self.send_time = {}
        self.time_windows = defaultdict(lambda: {
            'sent': 0,
            'received': 0,
            'dropped': 0,
            'bytes': 0,
            'delays': [],
        })
        
    def parse(self, window_size=1.0):
        print(f"[INFO] Parsing trace file: {self.trace_file}")
        print(f"[INFO] Window size: {window_size} seconds")
        print(f"[INFO] Link Capacity: {self.link_capacity_mbps} Mbps")
        
        try:
            with open(self.trace_file, 'r') as f:
                for line in f:
                    self._parse_line(line.strip(), window_size)
        except FileNotFoundError:
            print(f"[ERROR] Trace file not found: {self.trace_file}")
            sys.exit(1)
        
        metrics_list = self._compute_metrics(window_size)
        print(f"[SUCCESS] Parsed {len(metrics_list)} valid time windows")
        return metrics_list
    
    def _parse_line(self, line, window_size):
        if not line or line.startswith('#'):
            return
        
        parts = line.split()
        if len(parts) < 11:
            return
        
        try:
            event = parts[0]      # Event type (+, -, r, d)
            # NS2 older trace: (s/r/d), new: (+/-/r/d)
            if event == '+' or event == 's':
                event = 's'
            elif event == '-':
                event = 's_queue' # just enqueued, ignore for now to avoid double counting send
                return
                
            time = float(parts[1])
            pkt_size = int(parts[5])
            src_addr = parts[8]
            dst_addr = parts[9]
            seq_num = parts[10]
            pkt_id = parts[11] if len(parts) > 11 else f"{src_addr}-{dst_addr}-{seq_num}"
            
            from_node = parts[2]
            to_node = parts[3]
            src_node = src_addr.split('.')[0]
            dst_node = dst_addr.split('.')[0]
            
            window_idx = int(time / window_size)
            
            if event == 's' and from_node == src_node:  # Packet originally sent
                self.time_windows[window_idx]['sent'] += 1
                self.send_time[pkt_id] = time
                
            elif event == 'r' and to_node == dst_node:  # Packet received at destination
                self.time_windows[window_idx]['received'] += 1
                self.time_windows[window_idx]['bytes'] += pkt_size
                
                if pkt_id in self.send_time:
                    delay = time - self.send_time[pkt_id]
                    self.time_windows[window_idx]['delays'].append(delay)
                    
            elif event == 'd':  # Packet dropped anywhere
                self.time_windows[window_idx]['dropped'] += 1
                
        except (ValueError, IndexError):
            pass
    
    def _compute_metrics(self, window_size):
        metrics_list = []
        
        # Sort by time window to keep temporal properties
        for window_idx in sorted(self.time_windows.keys()):
            window = self.time_windows[window_idx]
            
            sent = window['sent']
            received = window['received']
            dropped = window['dropped']
            bytes_transferred = window['bytes']
            
            # Throughput = (bytes_received * 8) / (window_size * 1_000_000)
            throughput = (bytes_transferred * 8) / (window_size * 1_000_000) if window_size > 0 else 0.0
            
            # Remove idle windows where nothing was sent OR throughput is 0
            if sent == 0 or throughput == 0.0:
                continue
            
            # Packet loss rate = (dropped / sent) * 100
            packet_loss_rate = (dropped / sent * 100) if sent > 0 else 0.0
            
            # Average delay
            delays = window['delays']
            avg_delay = sum(delays) / len(delays) if delays else 0.0
            
            # Link utilization = (throughput / link_capacity) * 100
            link_utilization = (throughput / self.link_capacity_mbps * 100) if self.link_capacity_mbps > 0 else 0.0
            
            metrics = {
                'time_window': window_idx * window_size,
                'packets_sent': sent,
                'packets_received': received,
                'packets_dropped': dropped,
                'packet_loss_rate': round(packet_loss_rate, 4),
                'avg_delay': round(avg_delay, 6),
                'throughput': round(throughput, 6),
                'link_utilization': round(link_utilization, 4)
            }
            
            metrics_list.append(metrics)
        
        # Sort just in case
        metrics_list = sorted(metrics_list, key=lambda x: x['time_window'])
        return metrics_list


def main():
    if len(sys.argv) < 2:
        print("Usage: python trace_parser.py <trace_file>")
        print("Example: python trace_parser.py ../ns2/out.tr")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    parser = TraceParser(trace_file)
    metrics = parser.parse(window_size=1.0)
    
    print("\n" + "="*80)
    print("PARSED METRICS (First 10 windows)")
    print("="*80)
    
    for i, m in enumerate(metrics[:10]):
        print(f"\nWindow {i+1} (Time: {m['time_window']}s)")
        print(f"  Packets Sent: {m['packets_sent']}")
        print(f"  Packets Received: {m['packets_received']}")
        print(f"  Packets Dropped: {m['packets_dropped']}")
        print(f"  Packet Loss Rate: {m['packet_loss_rate']}%")
        print(f"  Avg Delay: {m['avg_delay']} sec")
        print(f"  Throughput: {m['throughput']} Mbps")
        print(f"  Link Utilization: {m['link_utilization']}%")
    
    print(f"\nTotal windows parsed: {len(metrics)}")


if __name__ == "__main__":
    main()
