#!/usr/bin/env python3
"""Report which shell-isolation mechanism this host can actually provide.

Run it INSIDE the container/pod that will host the routes, before a
multi-route experiment. It prints the mechanism `sandbox: auto` would select
and what to put in your configs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avo.agent.sandbox import bwrap_works, readable_residue, uid_isolation_available


def main() -> int:
    print(f"uid: {os.geteuid()} ({'root' if os.geteuid() == 0 else 'non-root'})")
    print(f"setpriv: {shutil.which('setpriv') or 'MISSING'}")
    print(f"bwrap:   {shutil.which('bwrap') or 'MISSING'}")
    if shutil.which("bwrap"):
        probe = subprocess.run(
            ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/tmp", "true"],
            capture_output=True, text=True)
        print(f"  namespace probe: "
              f"{'OK' if probe.returncode == 0 else probe.stderr.strip()[:80]}")

    bw, uidiso = bwrap_works(), uid_isolation_available()
    mech = "bwrap" if bw else ("uid" if uidiso else "none")
    print(f"\n=> mechanism available: {mech}")
    if mech == "bwrap":
        print("   Multi-route safe. Use:  sandbox: require")
    elif mech == "uid":
        print("   Multi-route safe WITHOUT capabilities (per-route uid +\n"
              "   0700 run dirs). Use:  sandbox: require")
    else:
        print("   NO isolation available in this container.")
        print("   - If you can run the framework as root here, uid isolation")
        print("     engages automatically (needs no capabilities).")
        print("   - Otherwise a multi-route comparison in ONE filesystem")
        print("     cannot be enforced; run routes SEQUENTIALLY with")
        print("     `avo audit` after each, or accept + document the risk")
        print("     (sandbox: none + allow_peer_visibility: true).")

    residue = readable_residue()
    if residue:
        print(f"\nWARNING: bootstrappable residue readable in /tmp: {residue[:6]}")
        print("Clean it before a comparative run (this is how R4 was contaminated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
