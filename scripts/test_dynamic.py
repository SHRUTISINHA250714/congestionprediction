import pandas as pd
from trace_parser import TraceParser

def get_metrics():
    import subprocess
    cmd = "ns ../ns2/topology_sim.tcl 3.0 50 15 0.2"
    subprocess.run(cmd, shell=True, capture_output=True)
    parser = TraceParser('out.tr', link_capacity_mbps=3.0)
    return parser.parse()

metrics = get_metrics()
df = pd.DataFrame(metrics)
if not df.empty:
    m = df['avg_delay'].mean()
    print("Mean:", m)
    print("Spikes:", (df['avg_delay'] > m * 1.5).sum())
