import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
NS2_DIR = PROJECT_ROOT / "ns2"

def main():

    print("\n===============================")
    print(" PARALLEL ML CONGESTION DEMO ")
    print("===============================\n")

    # 1️⃣ Remove runtime_state.pkl before ML run
    state_file = PROJECT_ROOT / "runtime_state.pkl"
    if state_file.exists():
        print("[INFO] Removing old runtime_state.pkl...")
        state_file.unlink()

    # 2️⃣ Start Baseline Simulation (Parallel)
    print("[STEP 1] Starting Baseline Simulation...")
    baseline_proc = subprocess.Popen(
        "ns topology_baseline.tcl",
        shell=True,
        cwd=NS2_DIR
    )

    # 3️⃣ Start ML Simulation (Parallel)
    print("[STEP 2] Starting ML Simulation...")
    ml_proc = subprocess.Popen(
        "ns topology_ml.tcl",
        shell=True,
        cwd=NS2_DIR
    )

    # 4️⃣ Wait for both to finish
    baseline_proc.wait()
    ml_proc.wait()

    print("\n[INFO] Both simulations completed.")

    # 5️⃣ Open NAM files simultaneously
    print("[STEP 3] Opening NAM Animations...")

    subprocess.Popen("nam baseline.nam", shell=True, cwd=NS2_DIR)
    subprocess.Popen("nam ml.nam", shell=True, cwd=NS2_DIR)

    print("\n✅ Parallel Demo Completed Successfully\n")

if __name__ == "__main__":
    main()