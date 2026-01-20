import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

results = pd.read_csv("results/lab4_thread_overhead/speed_results.csv")
os.makedirs("files", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 10))

# Get unique thread counts and element counts
thread_counts = sorted(results['thread_count'].unique())
element_counts = sorted(results['elements_count'].unique())

# Select a subset of element counts to plot (avoid clutter)
# Or plot all if you prefer
selected_elements = element_counts[::max(1, len(element_counts)//6)]  # Select ~6 element counts

colors = plt.cm.tab20(np.linspace(0, 1, len(selected_elements)))

baseline_threads = 1

for idx, elem_count in enumerate(selected_elements):
    elem_data = results[results['elements_count'] == elem_count]
    
    if elem_data.empty:
        continue
    
    # Get baseline (single-threaded) time for this element count
    baseline_time = elem_data[elem_data['thread_count'] == baseline_threads]['time_taken'].mean()
    
    if np.isnan(baseline_time):
        continue
    
    # Calculate speedup for each thread count
    speedups = []
    valid_threads = []
    
    for threads in thread_counts:
        thread_time = elem_data[elem_data['thread_count'] == threads]['time_taken'].mean()
        if not np.isnan(thread_time) and threads != baseline_threads:
            speedup = baseline_time / thread_time
            speedups.append(speedup)
            valid_threads.append(threads)
    
    if speedups:
        ax.plot(valid_threads, speedups, '-o', linewidth=2, markersize=8,
               color=colors[idx], label=f'{elem_count:,} elements', alpha=0.8)

# Add ideal speedup line (y = x)
max_threads = max(thread_counts)
ideal_x = np.array([1, max_threads])
ideal_y = ideal_x  # y = x for ideal speedup
ax.plot(ideal_x, ideal_y, 'k--', linewidth=2, alpha=0.5, label='Ideal speedup')

ax.set_title('Speedup vs Number of Threads', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Threads', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup (relative to single-threaded)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.4, linewidth=0.5)
ax.legend(fontsize=10, ncol=2)

# Ensure axes start at reasonable values
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("files/speedup_vs_threads.png", bbox_inches='tight', dpi=300)
plt.close()

print(f"Generated files/speedup_vs_threads.png")