import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

results_name = "cpu_load/monitor.csv"
os.makedirs("files", exist_ok=True)

results = pd.read_csv(results_name)

fig, ax = plt.subplots(figsize=(10, 6))

load = results["cpu_percent"]
times = results["timestamp_ms"]

ax.plot(times, load, marker='o', linestyle='-', color='tab:orange', linewidth=2.0)

ax.set_xlabel('Time', fontsize=12, fontweight='bold')
ax.set_ylabel('CPU Usage (%)', fontsize=12, fontweight='bold')
ax.set_title(f'CPU Load: {results["threads"][0]} threads', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0)

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"files/load.png", bbox_inches='tight', dpi=300)
plt.close()  # Don't show, just save

print("Plot saved to files/ directory!")
