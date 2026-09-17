"""Launch the campaign runner GUI:  python -m scripts.campaign [--sim]"""
import sys
from gui.campaign import main

if __name__ == "__main__":
    main(simulate="--sim" in sys.argv)
