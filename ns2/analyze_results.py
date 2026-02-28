# =============================================================
# NS2 ML vs Baseline Advanced Visual Comparison
# Generates professional 4-panel comparison graph
# =============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
BASELINE_FILE = "baseline.tr"
ML_FILE = "ml.tr"
SIMULATION_TIME = 15
PACKET_SIZE_BYTES = 1000
RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# -----------------------------
# TRACE PARSER
# -----------------------------
def parse_trace(file_path):
    sent = received = dropped = 0

    with open(file_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue

            if parts[0] == "+":
                sent += 1
            elif parts[0] == "r":
                received += 1
            elif parts[0] == "d":
                dropped += 1

    return sent, received, dropped


def compute_metrics(sent, received, dropped):
    loss_rate = (dropped / sent * 100) if sent > 0 else 0
    throughput = (received * PACKET_SIZE_BYTES * 8) / (SIMULATION_TIME * 1e6)

    # simple delay estimation (optional placeholder)
    avg_delay = (SIMULATION_TIME / received * 1000) if received > 0 else 0

    return loss_rate, throughput, avg_delay


# -----------------------------
# MAIN
# -----------------------------
def main():

    baseline_sent, baseline_recv, baseline_drop = parse_trace(BASELINE_FILE)
    ml_sent, ml_recv, ml_drop = parse_trace(ML_FILE)

    baseline_loss, baseline_tp, baseline_delay = compute_metrics(
        baseline_sent, baseline_recv, baseline_drop
    )

    ml_loss, ml_tp, ml_delay = compute_metrics(
        ml_sent, ml_recv, ml_drop
    )

    # Save CSV
    df = pd.DataFrame({
        "Metric": ["Total Drops", "Loss Rate (%)", "Avg Delay (ms)", "Throughput (Mbps)"],
        "Baseline": [
            baseline_drop,
            round(baseline_loss, 2),
            round(baseline_delay, 2),
            round(baseline_tp, 2)
        ],
        "ML-Based": [
            ml_drop,
            round(ml_loss, 2),
            round(ml_delay, 2),
            round(ml_tp, 2)
        ]
    })

    df.to_csv(os.path.join(RESULTS_DIR, "comparison_metrics.csv"), index=False)


    # -----------------------------
    # PROFESSIONAL GRAPH
    # -----------------------------

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Baseline vs ML-Based Performance Comparison", fontsize=14, fontweight="bold")

    # 1️⃣ Packet Drops
    axs[0, 0].bar(["Baseline", "ML-Based"], [baseline_drop, ml_drop])
    axs[0, 0].set_title("Packet Drops")
    axs[0, 0].set_ylabel("Packets Dropped")

    # 2️⃣ Packet Loss Rate
    axs[0, 1].bar(["Baseline", "ML-Based"], [baseline_loss, ml_loss])
    axs[0, 1].set_title("Packet Loss Rate")
    axs[0, 1].set_ylabel("Loss Rate (%)")

    # 3️⃣ Average Delay
    axs[1, 0].bar(["Baseline", "ML-Based"], [baseline_delay, ml_delay])
    axs[1, 0].set_title("Average Delay")
    axs[1, 0].set_ylabel("Delay (ms)")

    # 4️⃣ Throughput
    axs[1, 1].bar(["Baseline", "ML-Based"], [baseline_tp, ml_tp])
    axs[1, 1].set_title("Average Throughput")
    axs[1, 1].set_ylabel("Throughput (Mbps)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(RESULTS_DIR, "performance_comparison.png"))
    plt.close()

    print("\n✅ Professional comparison graph generated inside 'results/' folder.")


if __name__ == "__main__":
    main()