"""
Bring-up helper: list connected NI-DAQmx devices + their analog-input channels.

    python -m scripts.list_daq

Use it to confirm the exact module names in your chassis (e.g. cDAQ1Mod1,
cDAQ1Mod2) and then make sure config/rig.yaml matches. Needs the NI-DAQmx
runtime installed and the chassis connected.
"""
from __future__ import annotations


def main():
    try:
        import nidaqmx.system as nisys
    except ImportError:
        raise SystemExit("nidaqmx not installed -> pip install nidaqmx")

    try:
        system = nisys.System.local()
        devices = list(system.devices)
    except Exception as e:
        raise SystemExit(f"Could not query DAQ system (NI-DAQmx runtime installed?): {e}")

    if not devices:
        print("No DAQ devices found. Check the NI-DAQmx runtime and that the "
              "cDAQ-9174 + both NI-9234 modules are connected/powered.")
        return

    print(f"{len(devices)} device(s):")
    for d in devices:
        try:
            ptype = d.product_type
        except Exception:
            ptype = "?"
        print(f"\n  {d.name}   ({ptype})")
        try:
            chans = [c.name for c in d.ai_physical_chans]
            print("    AI channels: " + (", ".join(chans) if chans else "(none)"))
        except Exception as e:
            print(f"    AI channels unavailable: {e}")

    print("\nExpected mapping (config/rig.yaml):")
    print("  KΦ=Mod1/ai0  P1x=Mod1/ai1  P1y=Mod1/ai2  P2x=Mod1/ai3  P2y=Mod2/ai0")


if __name__ == "__main__":
    main()
