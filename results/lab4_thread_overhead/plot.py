import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

results = pd.read_csv("results/lab4_thread_overhead/speed_results.csv")
os.makedirs("files", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 10))

thread_counts = sorted(results['thread_count'].unique())[:12]
colors = plt.cm.tab20(np.linspace(0, 1, len(thread_counts)))

for idx, threads in enumerate(thread_counts):
    thread_data = results[results['thread_count'] == threads]
    
    if thread_data.empty:
        continue
    
    grouped = thread_data.groupby('elements_count')['time_taken'].mean()
    
    x = grouped.index.values
    y = grouped.values
    
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    ax.plot(x, y, '-', linewidth=3, color=colors[idx], 
           label=f'{threads} threads', alpha=0.8)

ax.set_title('Time vs Elements Count', fontsize=16, fontweight='bold')
ax.set_xlabel('Elements Count', fontsize=14, fontweight='bold')
ax.set_ylabel('Time Taken (ms)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.4, linewidth=0.5)
ax.legend(fontsize=10, ncol=3)

# if results['elements_count'].max() / results['elements_count'].min() > 100:
#     ax.set_xscale('log')
# if results['time_taken'].max() / results['time_taken'].min() > 100:
#     ax.set_yscale('log')

plt.tight_layout()
plt.savefig("files/time_lines.png", bbox_inches='tight', dpi=300)
plt.close()

print(f"Generated files/time_lines.png")