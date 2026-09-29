"""Reproduce every table and figure in results/ from scratch.

    python run_all.py

Deterministic: fixed seeds throughout. Running it twice gives identical output.
"""
import time

from src import data, evaluate, features, interpret, mixtures, plots

STEPS = [
    ("data      ", data.build),
    ("features  ", features.build),
    ("evaluate  ", evaluate.main),
    ("interpret ", interpret.main),
    ("mixtures  ", mixtures.main),
    ("plots     ", plots.main),
]

if __name__ == "__main__":
    t0 = time.time()
    for name, fn in STEPS:
        t = time.time()
        print(f"\n\n>>> {name} ...\n")
        fn()
        print(f"\n<<< {name} done in {time.time() - t:.1f}s")
    print(f"\nTotal {time.time() - t0:.1f}s")
