import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Configuration
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
colors = plt.cm.viridis(np.linspace(0, 1, 10))  # Nice color palette

results_name = "plot/speed_results.csv"
os.makedirs("files", exist_ok=True)

# Load and prepare data
results = pd.read_csv(results_name).sort_values("elements_count")

# Group by elements_count for cleaner iteration
for elements_c, group in results.groupby("elements_count"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for stacked/grouped bars
    threads = group["thread_count"].values
    times = group["time_taken"].values
    thread_pos = np.arange(len(threads))
    
    # Create bars with better positioning
    bars = ax.bar(thread_pos, times, 0.7, color='orange', 
                  edgecolor='white', linewidth=1.5, alpha=0.85)
    
    # Customize appearance
    ax.set_xlabel('Thread Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time Taken (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance: {elements_c:,} Elements', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(thread_pos)
    ax.set_xticklabels(threads, fontsize=11)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Improve layout and grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Dynamic y-limit with padding
    bottom = min(times) * 0.9
    ax.set_ylim(bottom=bottom)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(f"files/bar_{elements_c}.png", bbox_inches='tight', dpi=300)
    plt.close()  # Don't show, just save

print("All bar plots saved to files/ directory!")
