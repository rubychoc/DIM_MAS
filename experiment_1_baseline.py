"""
Experiment 1: Baseline Characterization
Purpose: Establish baseline performance across different classroom sizes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from coalition_formation import ClassroomModel
import random
from typing import Dict, List
import time as time_module


def run_single_simulation(n_students: int, k: int, ws: float, wf: float, wp: float, 
                          seed: int, total_time: float = 120.0) -> Dict:
    """Run a single simulation and collect metrics."""
    random.seed(seed)
    np.random.seed(seed)
    
    grid_size = max(10, int(np.ceil(np.sqrt(n_students * 1.5))))
    
    model = ClassroomModel(
        n_students=n_students,
        k=k,
        grid_size=grid_size,
        total_time=total_time,
        ws=ws, wf=wf, wp=wp
    )
    
    model.running = True
    max_steps = 1000
    steps = 0
    
    start_time = time_module.time()
    while model.running and steps < max_steps:
        model.step()
        steps += 1
    elapsed = time_module.time() - start_time
    
    # Calculate metrics
    total_welfare = model.compute_total_welfare()
    locked_entities = [e for e in model.entities if e.state == "locked"]
    backup_entities = [e for e in model.entities if e.state == "locked (backup)"]
    students_in_proper_groups = sum(e.size for e in locked_entities)
    students_in_backup = sum(e.size for e in backup_entities)
    num_orphans = students_in_backup
    
    # Count deadlock events across entire simulation history
    deadlock_count = 0
    for snapshot in model.history:
        for neg in snapshot['negotiations']:
            if neg.get('type') == 'deadlock-breakup':
                deadlock_count += 1
    
    return {
        'n_students': n_students,
        'k': k,
        'total_welfare': total_welfare,
        'welfare_per_student': total_welfare / n_students if n_students > 0 else 0,
        'convergence_steps': steps,
        'convergence_time': elapsed,
        'num_orphans': num_orphans,
        'num_proper_groups': len(locked_entities),
        'num_backup_groups': len(backup_entities),
        'deadlock_count': deadlock_count,
        'seed': seed
    }


def run_experiment_1():
    """Run Experiment 1: Baseline Characterization."""
    print("="*80)
    print("EXPERIMENT 1: BASELINE CHARACTERIZATION")
    print("="*80)
    print("\nTesting scalability across different classroom sizes")
    print("Fixed parameters: k=4, ws=0.4, wf=0.3, wp=0.3")
    print("Varied parameter: n_students = [12, 16, 20, 24, 28, 32, 40, 50, 60, 70, 80, 90, 100]")
    print("Runs per configuration: 30")
    print("\n")
    
    # Experimental parameters
    n_students_values = [12, 16, 20, 24, 28, 32, 40, 52, 60, 72, 80, 92, 100]
    k = 4
    ws, wf, wp = 0.4, 0.3, 0.3
    runs_per_config = 30
    
    results = []
    total_runs = len(n_students_values) * runs_per_config
    current_run = 0
    
    for n_students in n_students_values:
        print(f"\nRunning n_students={n_students} ({runs_per_config} runs)...")
        for run in range(runs_per_config):
            current_run += 1
            seed = 1000 + current_run
            
            result = run_single_simulation(n_students, k, ws, wf, wp, seed)
            results.append(result)
            
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/{runs_per_config} runs ({current_run}/{total_runs} total)")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Statistical Analysis: ANOVA
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: ONE-WAY ANOVA")
    print("="*80)
    
    # Test if classroom size affects welfare per student
    groups_welfare = [df[df['n_students'] == n]['welfare_per_student'].values 
                     for n in n_students_values]
    f_stat_welfare, p_value_welfare = stats.f_oneway(*groups_welfare)
    
    print(f"\nWelfare per Student:")
    print(f"  F-statistic: {f_stat_welfare:.4f}")
    print(f"  p-value: {p_value_welfare:.4e}")
    print(f"  Significant: {'YES' if p_value_welfare < 0.05 else 'NO'} (α=0.05)")
    
    # Test if classroom size affects convergence time
    groups_convergence = [df[df['n_students'] == n]['convergence_steps'].values 
                         for n in n_students_values]
    f_stat_conv, p_value_conv = stats.f_oneway(*groups_convergence)
    
    print(f"\nConvergence Steps:")
    print(f"  F-statistic: {f_stat_conv:.4f}")
    print(f"  p-value: {p_value_conv:.4e}")
    print(f"  Significant: {'YES' if p_value_conv < 0.05 else 'NO'} (α=0.05)")
    
    # Test if classroom size affects deadlock frequency
    groups_deadlock = [df[df['n_students'] == n]['deadlock_count'].values 
                      for n in n_students_values]
    f_stat_deadlock, p_value_deadlock = stats.f_oneway(*groups_deadlock)
    
    print(f"\nDeadlock Frequency:")
    print(f"  F-statistic: {f_stat_deadlock:.4f}")
    print(f"  p-value: {p_value_deadlock:.4e}")
    print(f"  Significant: {'YES' if p_value_deadlock < 0.05 else 'NO'} (α=0.05)")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY CLASSROOM SIZE")
    print("="*80)
    
    summary = df.groupby('n_students').agg({
        'welfare_per_student': ['mean', 'std'],
        'convergence_steps': ['mean', 'std'],
        'num_orphans': ['mean', 'std'],
        'deadlock_count': ['mean', 'std']
    }).round(3)
    
    print("\n", summary)
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 1: Baseline Characterization Across Classroom Sizes', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Welfare per Student
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='n_students', y='welfare_per_student', ax=ax1, palette='viridis')
    ax1.set_title('Social Welfare per Student', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Students', fontsize=10)
    ax1.set_ylabel('Welfare per Student', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Convergence Steps (line plot with error bars)
    ax2 = axes[0, 1]
    conv_summary = df.groupby('n_students')['convergence_steps'].agg(['mean', 'std'])
    ax2.plot(conv_summary.index, conv_summary['mean'], marker='o', linewidth=2, 
             markersize=8, color='coral')
    ax2.fill_between(conv_summary.index, 
                     conv_summary['mean'] - conv_summary['std'],
                     conv_summary['mean'] + conv_summary['std'],
                     alpha=0.3, color='coral')
    ax2.set_title('Convergence Time (Steps)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Students', fontsize=10)
    ax2.set_ylabel('Steps to Convergence', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Average Total Social Welfare
    ax3 = axes[1, 0]
    welfare_summary = df.groupby('n_students')['total_welfare'].mean()
    ax3.bar(welfare_summary.index, welfare_summary.values, color='steelblue', alpha=0.7)
    ax3.set_title('Average Total Social Welfare per Classroom', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Students', fontsize=10)
    ax3.set_ylabel('Avg. Total Welfare', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Deadlock Frequency
    ax4 = axes[1, 1]
    deadlock_summary = df.groupby('n_students')['deadlock_count'].mean()
    ax4.bar(deadlock_summary.index, deadlock_summary.values, color='coral', alpha=0.7)
    ax4.set_title('Average Deadlock Events', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Students', fontsize=10)
    ax4.set_ylabel('Avg. Deadlock Count', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_1_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: experiment_1_results.png")
    
    # Save data
    df.to_csv('experiment_1_data.csv', index=False)
    print("✓ Saved data: experiment_1_data.csv")
    
    # Save summary report
    with open('experiment_1_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT 1: BASELINE CHARACTERIZATION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("How does classroom size affect coalition formation performance?\n\n")
        
        f.write("EXPERIMENTAL SETUP:\n")
        f.write("- Fixed parameters: k=4, ws=0.4, wf=0.3, wp=0.3\n")
        f.write("- Varied: n_students = [12, 16, 20, 24, 28, 32, 40, 50, 60, 70, 80, 90, 100]\n")
        f.write("- Runs per configuration: 30\n")
        f.write("- Total simulations: 390\n\n")
        
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        # Best and worst configurations
        best_n = df.groupby('n_students')['welfare_per_student'].mean().idxmax()
        best_welfare = df.groupby('n_students')['welfare_per_student'].mean().max()
        worst_n = df.groupby('n_students')['welfare_per_student'].mean().idxmin()
        worst_welfare = df.groupby('n_students')['welfare_per_student'].mean().min()
        
        f.write(f"1. OPTIMAL CLASSROOM SIZE:\n")
        f.write(f"   - Best: n={best_n} students (avg welfare/student = {best_welfare:.3f})\n")
        f.write(f"   - Worst: n={worst_n} students (avg welfare/student = {worst_welfare:.3f})\n")
        f.write(f"   - Difference: {((best_welfare - worst_welfare) / worst_welfare * 100):.1f}% improvement\n\n")
        
        f.write(f"2. STATISTICAL SIGNIFICANCE:\n")
        f.write(f"   - Welfare: F={f_stat_welfare:.2f}, p={p_value_welfare:.2e} ")
        f.write(f"({'SIGNIFICANT' if p_value_welfare < 0.05 else 'NOT SIGNIFICANT'})\n")
        f.write(f"   - Convergence: F={f_stat_conv:.2f}, p={p_value_conv:.2e} ")
        f.write(f"({'SIGNIFICANT' if p_value_conv < 0.05 else 'NOT SIGNIFICANT'})\n")
        f.write(f"   - Deadlock Frequency: F={f_stat_deadlock:.2f}, p={p_value_deadlock:.2e} ")
        f.write(f"({'SIGNIFICANT' if p_value_deadlock < 0.05 else 'NOT SIGNIFICANT'})\n\n")
        
        f.write(f"3. SCALABILITY:\n")
        avg_conv_small = df[df['n_students'] <= 20]['convergence_steps'].mean()
        avg_conv_large = df[df['n_students'] >= 40]['convergence_steps'].mean()
        f.write(f"   - Small classrooms (≤20): {avg_conv_small:.1f} steps average\n")
        f.write(f"   - Large classrooms (≥40): {avg_conv_large:.1f} steps average\n")
        f.write(f"   - Scaling factor: {(avg_conv_large / avg_conv_small):.2f}x slower\n\n")
        
        f.write(f"4. CONVERGENCE BY CLASSROOM SIZE:\n")
        for n in n_students_values:
            conv_mean = df[df['n_students'] == n]['convergence_steps'].mean()
            conv_std = df[df['n_students'] == n]['convergence_steps'].std()
            f.write(f"   - n={n:2d}: {conv_mean:.1f} ± {conv_std:.1f} steps\n")
        
        f.write(f"\n5. ORPHAN STUDENTS:\n")
        for n in n_students_values:
            orphans = df[df['n_students'] == n]['num_orphans'].mean()
            f.write(f"   - n={n:2d}: {orphans:.2f} orphans on average\n")
        
        f.write(f"\n6. DEADLOCK FREQUENCY:\n")
        total_deadlocks = df['deadlock_count'].sum()
        f.write(f"   - Total deadlocks across all runs: {int(total_deadlocks)}\n")
        f.write(f"   - Average per simulation: {df['deadlock_count'].mean():.2f}\n")
        for n in n_students_values:
            deadlocks = df[df['n_students'] == n]['deadlock_count'].mean()
            f.write(f"   - n={n:2d}: {deadlocks:.2f} deadlocks per simulation\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        if p_value_welfare < 0.05:
            f.write("Classroom size has a SIGNIFICANT effect on social welfare.\n")
        else:
            f.write("Classroom size does NOT significantly affect social welfare.\n")
        
        if best_welfare > worst_welfare * 1.1:
            f.write(f"There is a substantial ({((best_welfare - worst_welfare) / worst_welfare * 100):.1f}%) ")
            f.write(f"difference between best and worst configurations.\n")
        else:
            f.write("System performs consistently across different classroom sizes.\n")
        
        f.write("\nFiles generated:\n")
        f.write("- experiment_1_results.png (visualizations)\n")
        f.write("- experiment_1_data.csv (raw data)\n")
        f.write("- experiment_1_summary.txt (this file)\n")
    
    print("✓ Saved summary: experiment_1_summary.txt")
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETE")
    print("="*80)
    
    return df


if __name__ == "__main__":
    df_results = run_experiment_1()
    plt.show()
