"""Launch the live probe gap/health monitor:  python -m scripts.monitor [--sim]"""
import sys
from gui.gap_monitor import main

if __name__ == "__main__":
    main(simulate="--sim" in sys.argv)
