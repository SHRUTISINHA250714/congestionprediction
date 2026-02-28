import os
import sys
import pickle
import pandas as pd

STATE_FILE = "runtime_state.pkl"

class CongestionPredictor:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def load_model(self):
        with open(os.path.join(self.models_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(self.models_dir, "feature_names.pkl"), "rb") as f:
            self.feature_names = pickle.load(f)

        with open(os.path.join(self.models_dir, "random_forest.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def predict(self, feature_dict):
        X = pd.DataFrame([feature_dict])
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict_proba(X_scaled)[0][1])


def extract_latest_window(trace_file, current_time, window_size=0.5):

    packets_sent = 0
    packets_received = 0
    packets_dropped = 0

    start_time = current_time - window_size

    with open(trace_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                event = parts[0]
                time = float(parts[1])
            except:
                continue

            if start_time <= time <= current_time:
                if event == "+":
                    packets_sent += 1
                elif event == "r":
                    packets_received += 1
                elif event == "d":
                    packets_dropped += 1

    loss_rate = (packets_dropped / packets_sent * 100) if packets_sent > 0 else 0
    throughput = packets_received * 1000 / 1e6
    utilization = min(100, throughput * 50)

    return {
        "throughput": throughput,
        "avg_delay": 0.02,
        "packet_loss_rate": loss_rate,
        "link_utilization": utilization
    }


def main():

    if len(sys.argv) != 3:
        sys.exit(1)

    trace_file = sys.argv[1]
    current_time = float(sys.argv[2])

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

    predictor = CongestionPredictor(models_dir)
    predictor.load_model()

    current_metrics = extract_latest_window(trace_file, current_time)

    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "wb") as f:
            pickle.dump(current_metrics, f)
        print(0.0)
        return

    with open(STATE_FILE, "rb") as f:
        previous_metrics = pickle.load(f)

    feature_dict = {
        "throughput_lag1": previous_metrics["throughput"],
        "delay_lag1": previous_metrics["avg_delay"],
        "loss_lag1": previous_metrics["packet_loss_rate"],
        "utilization_lag1": previous_metrics["link_utilization"],
        "rolling_mean_util": (
            previous_metrics["link_utilization"]
            + current_metrics["link_utilization"]
        ) / 2,
        "utilization_change": (
            current_metrics["link_utilization"]
            - previous_metrics["link_utilization"]
        )
    }

    prob = predictor.predict(feature_dict)

    with open(STATE_FILE, "wb") as f:
        pickle.dump(current_metrics, f)

    print(prob)


if __name__ == "__main__":
    main()