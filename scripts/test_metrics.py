import pandas as pd
from trace_parser import TraceParser

parser = TraceParser('ns2/out.tr', link_capacity_mbps=2.5)
metrics = parser.parse(window_size=1.0)
df = pd.DataFrame(metrics)
print("Max util:", df['link_utilization'].max())
print("Mean util:", df['link_utilization'].mean())
print("Max delay:", df['avg_delay'].max())
print("Mean loss:", df['packet_loss_rate'].mean())
