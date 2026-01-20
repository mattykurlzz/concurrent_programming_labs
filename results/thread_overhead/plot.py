import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

results_name = "results/thread_overhead/speed_results.csv"
os.makedirs("files", exist_ok=True)

results = pd.read_csv(results_name)

print(f"Data loaded: {len(results)} rows")
print(f"Unique elements_count: {sorted(results['elements_count'].unique())}")
print(f"Unique schedule_type: {results['schedule_type'].unique()}")
print(f"Unique thread_count: {sorted(results['thread_count'].unique())}")

# Функция для вычисления ускорения
def calculate_speedup(df):
    """Вычисляет ускорение для каждого количества потоков относительно 1 потока"""
    speedup_data = []
    
    for schedule in df['schedule_type'].unique():
        schedule_df = df[df['schedule_type'] == schedule]
        
        for elements in schedule_df['elements_count'].unique():
            elements_df = schedule_df[schedule_df['elements_count'] == elements]
            
            # Находим время на 1 потоке
            time_1_thread = elements_df[elements_df['thread_count'] == 1]['time_taken']
            
            if len(time_1_thread) == 0:
                continue
                
            # Усредняем, если несколько измерений
            t1 = time_1_thread.mean()
            
            if t1 == 0:  # Избегаем деления на 0
                continue
                
            # Вычисляем ускорение для каждого количества потоков
            for thread_count in elements_df['thread_count'].unique():
                if thread_count == 1:
                    continue
                    
                time_p_threads = elements_df[elements_df['thread_count'] == thread_count]['time_taken']
                
                if len(time_p_threads) == 0:
                    continue
                    
                tp = time_p_threads.mean()
                
                if tp == 0:
                    speedup = float('inf')
                else:
                    speedup = t1 / tp
                
                # Идеальное ускорение (линейное)
                ideal_speedup = thread_count
                
                # Эффективность
                efficiency = speedup / thread_count if speedup != float('inf') else 1.0
                
                speedup_data.append({
                    'elements_count': elements,
                    'schedule_type': schedule,
                    'thread_count': thread_count,
                    'time_1': t1,
                    'time_p': tp,
                    'speedup': speedup,
                    'ideal_speedup': ideal_speedup,
                    'efficiency': efficiency
                })
    
    return pd.DataFrame(speedup_data)

# Вычисляем ускорение
speedup_df = calculate_speedup(results)

if len(speedup_df) == 0:
    print("Невозможно вычислить ускорение. Проверьте наличие данных для 1 потока.")
    exit()

print(f"\nSpeedup data calculated: {len(speedup_df)} points")

# 1. График ускорения для разных N
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
axes = axes.flatten()

schedule_types = speedup_df['schedule_type'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(schedule_types)))

for idx, schedule in enumerate(schedule_types):
    ax = axes[idx]
    schedule_data = speedup_df[speedup_df['schedule_type'] == schedule]
    
    # Группируем по elements_count
    for elements in sorted(schedule_data['elements_count'].unique()):
        elements_data = schedule_data[schedule_data['elements_count'] == elements]
        
        if len(elements_data) > 0:
            # Сортируем по thread_count
            elements_data = elements_data.sort_values('thread_count')
            
            ax.plot(elements_data['thread_count'], elements_data['speedup'], 
                   'o-', linewidth=2, markersize=6,
                   label=f'N={elements}', alpha=0.8)
    
    # Линия идеального ускорения
    thread_counts = sorted(schedule_data['thread_count'].unique())
    ax.plot(thread_counts, thread_counts, 'k--', linewidth=1, 
            label='Ideal speedup', alpha=0.5)
    
    ax.set_title(f'Schedule: {schedule}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Thread Count', fontsize=10)
    ax.set_ylabel('Speedup (T₁/Tₚ)', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    
    # Порог ускорения = 1
    ax.axhline(y=1, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.5, 1.05, 'Break-even', transform=ax.get_yaxis_transform(),
            fontsize=8, color='red', ha='center')

plt.suptitle('Parallel Speedup for Different Problem Sizes (N)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig("files/speedup_by_N.png", bbox_inches='tight', dpi=300)
plt.close()

# 2. График эффективности
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
axes = axes.flatten()

for idx, schedule in enumerate(schedule_types):
    ax = axes[idx]
    schedule_data = speedup_df[speedup_df['schedule_type'] == schedule]
    
    # Группируем по elements_count
    for elements in sorted(schedule_data['elements_count'].unique()):
        elements_data = schedule_data[schedule_data['elements_count'] == elements]
        
        if len(elements_data) > 0:
            elements_data = elements_data.sort_values('thread_count')
            
            ax.plot(elements_data['thread_count'], elements_data['efficiency'], 
                   's-', linewidth=2, markersize=5,
                   label=f'N={elements}', alpha=0.8)
    
    ax.set_title(f'Schedule: {schedule}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Thread Count', fontsize=10)
    ax.set_ylabel('Efficiency (Speedup/Threads)', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    
    # Порог эффективности = 0.5 (50%)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.5, 0.53, '50% efficiency', transform=ax.get_yaxis_transform(),
            fontsize=8, color='red', ha='center')

plt.suptitle('Parallel Efficiency for Different Problem Sizes (N)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig("files/efficiency_by_N.png", bbox_inches='tight', dpi=300)
plt.close()

# 3. Нахождение точек, где накладные расходы превышают выигрыш (speedup < 1)
print("\n" + "="*60)
print("ANALYSIS: Overhead Exceeds Benefit (Speedup < 1)")
print("="*60)

overhead_points = {}

for schedule in schedule_types:
    schedule_data = speedup_df[speedup_df['schedule_type'] == schedule]
    
    # Находим точки, где ускорение < 1
    overhead_cases = schedule_data[schedule_data['speedup'] < 1]
    
    if len(overhead_cases) > 0:
        print(f"\nSchedule: {schedule}")
        print("-" * 40)
        
        # Группируем по elements_count
        for elements in sorted(overhead_cases['elements_count'].unique()):
            elements_cases = overhead_cases[overhead_cases['elements_count'] == elements]
            
            # Находим минимальное ускорение для этого N
            min_speedup = elements_cases['speedup'].min()
            worst_threads = elements_cases.loc[elements_cases['speedup'].idxmin(), 'thread_count']
            
            print(f"  N = {elements}:")
            print(f"    Min speedup = {min_speedup:.3f} at {worst_threads} threads")
            print(f"    Overhead cases: {len(elements_cases)} thread configurations")
            
            # Детали для каждого количества потоков
            for _, row in elements_cases.iterrows():
                print(f"      Threads {row['thread_count']}: speedup = {row['speedup']:.3f}, "
                      f"efficiency = {row['efficiency']:.1%}")
        
        # Находим максимальное N с overhead (speedup < 1)
        if len(overhead_cases['elements_count'].unique()) > 0:
            max_N_overhead = overhead_cases['elements_count'].max()
            print(f"\n  Maximum N with overhead: {max_N_overhead}")
            
            # Проверяем, есть ли данные для N > max_N_overhead
            larger_N = schedule_data[schedule_data['elements_count'] > max_N_overhead]
            if len(larger_N) > 0:
                # Проверяем, все ли ускорения для N > max_N_overhead >= 1
                min_speedup_larger = larger_N['speedup'].min()
                if min_speedup_larger >= 1:
                    print(f"  For N > {max_N_overhead}, all speedups >= 1")
                    print(f"  Threshold N₁ ≈ {max_N_overhead}")
                else:
                    print(f"  Warning: Overhead persists for N > {max_N_overhead}")
                    print(f"  Min speedup for N > {max_N_overhead}: {min_speedup_larger:.3f}")
        else:
            print("  No clear threshold found")
    else:
        print(f"\nSchedule: {schedule}")
        print("-" * 40)
        print("  No cases with speedup < 1 found")
        print("  Overhead never exceeds benefit for measured N")
        
        # Находим минимальный N в данных
        min_N = schedule_data['elements_count'].min()
        print(f"  Smallest measured N = {min_N}")
        
        # Проверяем минимальное ускорение для минимального N
        min_N_data = schedule_data[schedule_data['elements_count'] == min_N]
        if len(min_N_data) > 0:
            min_speedup_min_N = min_N_data['speedup'].min()
            print(f"  Minimum speedup for N={min_N}: {min_speedup_min_N:.3f}")

# 4. График зависимости ускорения от N для разных thread_count
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
axes = axes.flatten()

for idx, schedule in enumerate(schedule_types):
    ax = axes[idx]
    schedule_data = speedup_df[speedup_df['schedule_type'] == schedule]
    
    # Группируем по thread_count
    for threads in sorted(schedule_data['thread_count'].unique()):
        thread_data = schedule_data[schedule_data['thread_count'] == threads]
        
        if len(thread_data) > 0:
            thread_data = thread_data.sort_values('elements_count')
            
            ax.plot(thread_data['elements_count'], thread_data['speedup'], 
                   'o-', linewidth=2, markersize=6,
                   label=f'{threads} threads', alpha=0.8)
    
    ax.set_title(f'Schedule: {schedule}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Elements Count (N)', fontsize=10)
    ax.set_ylabel('Speedup (T₁/Tₚ)', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    
    # Порог ускорения = 1
    ax.axhline(y=1, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # Логарифмическая шкала по X, если значения сильно различаются
    if schedule_data['elements_count'].max() / max(1, schedule_data['elements_count'].min()) > 10:
        ax.set_xscale('log')
        ax.set_xlabel('Elements Count (N, log scale)', fontsize=10)

plt.suptitle('Speedup vs Problem Size for Different Thread Counts', 
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig("files/speedup_vs_N.png", bbox_inches='tight', dpi=300)
plt.close()

# 5. Сводная таблица пороговых значений
print("\n" + "="*60)
print("SUMMARY: Overhead Thresholds")
print("="*60)

summary_data = []

for schedule in schedule_types:
    schedule_data = speedup_df[speedup_df['schedule_type'] == schedule]
    
    if len(schedule_data) == 0:
        continue
    
    # Находим все N с speedup < 1
    overhead_N = schedule_data[schedule_data['speedup'] < 1]['elements_count'].unique()
    
    if len(overhead_N) > 0:
        max_overhead_N = max(overhead_N)
        # Находим минимальное N без overhead (если есть)
        no_overhead_N = [n for n in schedule_data['elements_count'].unique() 
                        if n not in overhead_N]
        
        if no_overhead_N:
            min_no_overhead_N = min(no_overhead_N)
            threshold_range = f"{max_overhead_N} < N₁ < {min_no_overhead_N}"
        else:
            threshold_range = f"N₁ > {max_overhead_N} (overhead for all measured N)"
    else:
        max_overhead_N = 0
        min_N = min(schedule_data['elements_count'].unique())
        threshold_range = f"N₁ < {min_N} (no overhead measured)"
    
    # Статистика по эффективности
    avg_efficiency = schedule_data['efficiency'].mean()
    min_efficiency = schedule_data['efficiency'].min()
    
    summary_data.append({
        'Schedule': schedule,
        'Max N with overhead': max_overhead_N,
        'Threshold N₁ range': threshold_range,
        'Avg Efficiency': f"{avg_efficiency:.1%}",
        'Min Efficiency': f"{min_efficiency:.1%}",
        'Data Points': len(schedule_data)
    })

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# 6. Визуализация порогов
fig, ax = plt.subplots(figsize=(12, 6))

for schedule in schedule_types:
    schedule_data = speedup_df[speedup_df['schedule_type'] == schedule]
    
    if len(schedule_data) == 0:
        continue
    
    # Для каждого N находим минимальное ускорение
    min_speedup_by_N = []
    N_values = []
    
    for elements in sorted(schedule_data['elements_count'].unique()):
        elements_data = schedule_data[schedule_data['elements_count'] == elements]
        min_speedup = elements_data['speedup'].min()
        min_speedup_by_N.append(min_speedup)
        N_values.append(elements)
    
    ax.plot(N_values, min_speedup_by_N, 'o-', linewidth=2, markersize=8,
            label=f'{schedule}', alpha=0.8)

ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label='Break-even (speedup = 1)')
ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, 
           label='50% of ideal')

ax.set_title('Minimum Speedup vs Problem Size (Overhead Analysis)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Elements Count (N)', fontsize=12, fontweight='bold')
ax.set_ylabel('Minimum Speedup across all thread counts', 
              fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10, framealpha=0.9)

# Логарифмическая шкала если нужно
if max(N_values) / max(1, min(N_values)) > 10:
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig("files/overhead_thresholds.png", bbox_inches='tight', dpi=300)
plt.close()

print("\n" + "="*60)
print("GENERATED FILES:")
print("="*60)
print("1. files/speedup_by_N.png - Speedup vs thread count for different N")
print("2. files/efficiency_by_N.png - Efficiency vs thread count")
print("3. files/speedup_vs_N.png - Speedup vs N for different thread counts")
print("4. files/overhead_thresholds.png - Minimum speedup vs problem size")
print("="*60)
print("\nKEY INSIGHTS:")
print("="*60)
print("1. Speedup < 1: Overhead exceeds parallelization benefit")
print("2. N₁: Problem size threshold where overhead stops dominating")
print("3. Look for points where curves cross the red line (speedup=1)")
print("="*60)