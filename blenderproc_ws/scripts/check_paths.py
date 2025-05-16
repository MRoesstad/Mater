import blenderproc as bproc  # Required first!
import sys

print("[DEBUG] sys.path entries:")
for p in sys.path:
    print(" -", p)

try:
    import bop_toolkit_lib
    print("[SUCCESS] bop_toolkit_lib was found.")
except ImportError:
    print("[ERROR] bop_toolkit_lib NOT found.")
