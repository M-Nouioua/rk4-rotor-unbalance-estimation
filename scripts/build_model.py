"""
Build the ROSS model and report first critical speed(s).
Validate the printed critical speed against a measured run-up (Bode peak).

    python -m scripts.build_model
"""
from __future__ import annotations

from config import load_config
from model.rk4_model import build_rotor, critical_speeds


def main():
    cfg = load_config()
    rotor = build_rotor(cfg)
    print(rotor)
    res = critical_speeds(rotor, max_rpm=cfg["order_tracking"]["max_rpm"])
    print("Predicted critical speeds (rpm):")
    for i, c in enumerate(res["critical_rpm"], 1):
        print(f"  mode {i}: {c:8.1f} rpm")
    print("\nCompare with the measured run-up first-critical peak, then adjust "
          "bearing stiffness (kxx/kyy) in config/rig.yaml to match.")


if __name__ == "__main__":
    main()
