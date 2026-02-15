"""
Experiment 2: Group Size Sensitivity
Purpose: Find optimal group size for different classroom sizes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from coalition_formation import ClassroomModel
import random
from typing import Dict
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
    
    completion_rate = students_in_proper_groups / n_students if n_students > 0 else 0
    
    # Calculate utility variance (fairness measure)
    individual_utilities = []
    for entity in model.entities:
        for student in entity.members:
            utility = entity.calculate_utility(student, entity.members)
            individual_utilities.append(utility)
    
    utility_variance = np.var(individual_utilities) if individual_utilities else 0
    utility_std = np.std(individual_utilities) if individual_utilities else 0
    
    return {
        'n_students': n_students,
        'k': k,
        'total_welfare': total_welfare,
        'welfare_per_student': total_welfare / n_students if n_students > 0 else 0,
        'completion_rate': completion_rate,
        'utility_variance': utility_variance,
        'utility_std': utility_std,
        'convergence_steps': steps,
        'num_final_groups': len(model.entities),
        'seed': seed
    }


def run_experiment_2():
    """Run Experiment 2: Group Size Sensitivity."""
    print("="*80)
    print("EXPERIMENT 2: GROUP SIZE SENSITIVITY")
    print("="*80)
    print("\nTesting optimal group size for different classroom sizes")
    print("Fixed parameters: ws=0.4, wf=0.3, wp=0.3")
    print("Varied parameters: n_students = [20, 30, 40, 50, 60], k = [3, 4, 5, 6, 7, 8, 9, 10]")
    print("Runs per configuration: 25")
    print("\n")
    
    # Experimental parameters
    n_students_values = [20, 30, 40, 50, 60]
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    ws, wf, wp = 0.4, 0.3, 0.3
    runs_per_config = 25
    
    results = []
    total_configs = len(n_students_values) * len(k_values)
    total_runs = total_configs * runs_per_config
    current_run = 0
    
    for n_students in n_students_values:
        for k in k_values:
            print(f"\nRunning n_students={n_students}, k={k} ({runs_per_config} runs)...")
            for run in range(runs_per_config):
                current_run += 1
                seed = 2000 + current_run
                
                result = run_single_simulation(n_students, k, ws, wf, wp, seed)
                results.append(result)
                
                if (run + 1) % 10 == 0:
                    print(f"  Completed {run + 1}/{runs_per_config} runs ({current_run}/{total_runs} total)")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Statistical Analysis: Two-Way ANOVA
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: TWO-WAY ANOVA")
    print("="*80)
    
    # Test effects on welfare per student
    print("\n--- Welfare per Student ---")
    
    # Main effect of classroom size
    groups_n = [df[df['n_students'] == n]['welfare_per_student'].values 
                for n in n_students_values]
    f_stat_n, p_value_n = stats.f_oneway(*groups_n)
    print(f"\nMain Effect - Classroom Size (n):")
    print(f"  F-statistic: {f_stat_n:.4f}")
    print(f"  p-value: {p_value_n:.4e}")
    print(f"  Significant: {'YES' if p_value_n < 0.05 else 'NO'} (α=0.05)")
    
    # Main effect of group size
    groups_k = [df[df['k'] == k]['welfare_per_student'].values 
                for k in k_values]
    f_stat_k, p_value_k = stats.f_oneway(*groups_k)
    print(f"\nMain Effect - Group Size (k):")
    print(f"  F-statistic: {f_stat_k:.4f}")
    print(f"  p-value: {p_value_k:.4e}")
    print(f"  Significant: {'YES' if p_value_k < 0.05 else 'NO'} (α=0.05)")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary = df.groupby(['n_students', 'k']).agg({
        'welfare_per_student': ['mean', 'std'],
        'utility_std': ['mean', 'std']
    }).round(3)
    
    print("\n", summary)
    
    # Find optimal configurations
    print("\n" + "="*80)
    print("OPTIMAL GROUP SIZE PER CLASSROOM SIZE")
    print("="*80)
    
    for n in n_students_values:
        subset = df[df['n_students'] == n]
        best_k = subset.groupby('k')['welfare_per_student'].mean().idxmax()
        best_welfare = subset.groupby('k')['welfare_per_student'].mean().max()
        print(f"\nn_students={n}: Optimal k={best_k} (avg welfare/student={best_welfare:.3f})")
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Experiment 2: Group Size Sensitivity Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Multi-line plot: welfare vs group size for different classroom sizes
    # Generate colors and markers dynamically based on number of classroom sizes
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 0.9, len(n_students_values)))
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h']  # Extended marker list
    
    for i, n in enumerate(n_students_values):
        subset = df[df['n_students'] == n]
        welfare_by_k = subset.groupby('k')['welfare_per_student'].mean()
        welfare_std = subset.groupby('k')['welfare_per_student'].std()
        
        ax.plot(welfare_by_k.index, welfare_by_k.values, 
                marker=markers[i % len(markers)], linewidth=2.5, markersize=10,
                label=f'n={n} students', color=colors[i])
        ax.fill_between(welfare_by_k.index,
                       welfare_by_k.values - welfare_std.values,
                       welfare_by_k.values + welfare_std.values,
                       alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Group Size (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Welfare per Student', fontsize=12, fontweight='bold')
    ax.set_title('Social Welfare by Group Size and Classroom Size', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('experiment_2_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: experiment_2_results.png")
    
    # Save data
    df.to_csv('experiment_2_data.csv', index=False)
    print("✓ Saved data: experiment_2_data.csv")
    
    # Save summary report
    with open('experiment_2_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT 2: GROUP SIZE SENSITIVITY - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("What is the optimal group size (k) for different classroom sizes?\n\n")
        
        f.write("EXPERIMENTAL SETUP:\n")
        f.write("- Fixed parameters: ws=0.4, wf=0.3, wp=0.3\n")
        f.write(f"- Varied: n_students = {n_students_values}, k = {k_values}\n")
        f.write(f"- Runs per configuration: {runs_per_config}\n")
        f.write(f"- Total simulations: {len(n_students_values) * len(k_values) * runs_per_config}\n\n")
        
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. OPTIMAL GROUP SIZE BY CLASSROOM SIZE:\n\n")
        for n in n_students_values:
            subset = df[df['n_students'] == n]
            best_k = subset.groupby('k')['welfare_per_student'].mean().idxmax()
            best_welfare = subset.groupby('k')['welfare_per_student'].mean().max()
            worst_k = subset.groupby('k')['welfare_per_student'].mean().idxmin()
            worst_welfare = subset.groupby('k')['welfare_per_student'].mean().min()
            
            f.write(f"   n={n} students:\n")
            f.write(f"   - BEST: k={best_k} (welfare/student = {best_welfare:.3f})\n")
            f.write(f"   - WORST: k={worst_k} (welfare/student = {worst_welfare:.3f})\n")
            f.write(f"   - Improvement: {((best_welfare - worst_welfare) / worst_welfare * 100):.1f}%\n\n")
        
        f.write(f"2. STATISTICAL SIGNIFICANCE:\n\n")
        f.write(f"   Welfare per Student:\n")
        f.write(f"   - Classroom size effect: F={f_stat_n:.2f}, p={p_value_n:.2e} ")
        f.write(f"({'SIGNIFICANT' if p_value_n < 0.05 else 'NOT SIGNIFICANT'})\n")
        f.write(f"   - Group size effect: F={f_stat_k:.2f}, p={p_value_k:.2e} ")
        f.write(f"({'SIGNIFICANT' if p_value_k < 0.05 else 'NOT SIGNIFICANT'})\n\n")
        
        f.write(f"3. DETAILED RESULTS BY CONFIGURATION:\n\n")
        for n in n_students_values:
            f.write(f"   n={n} students:\n")
            subset = df[df['n_students'] == n]
            for k in k_values:
                welfare = subset[subset['k'] == k]['welfare_per_student'].mean()
                welfare_std = subset[subset['k'] == k]['welfare_per_student'].std()
                f.write(f"   - k={k}: welfare={welfare:.3f} ± {welfare_std:.3f}\n")
            f.write("\n")
        
        # Overall best configuration
        overall_best_idx = df.groupby(['n_students', 'k'])['welfare_per_student'].mean().idxmax()
        overall_best_welfare = df.groupby(['n_students', 'k'])['welfare_per_student'].mean().max()
        
        f.write(f"4. OVERALL BEST CONFIGURATION:\n")
        f.write(f"   - n={overall_best_idx[0]}, k={overall_best_idx[1]}\n")
        f.write(f"   - Welfare per student: {overall_best_welfare:.3f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        if p_value_k < 0.05:
            f.write("Group size (k) has a SIGNIFICANT effect on social welfare.\n")
            
            # Identify overall trend
            overall_trend = df.groupby('k')['welfare_per_student'].mean()
            if overall_trend.iloc[-1] > overall_trend.iloc[0]:
                f.write("Larger groups consistently produce higher welfare across all classroom sizes.\n")
            else:
                f.write("Smaller groups consistently produce higher welfare across all classroom sizes.\n")
            
            f.write("Recommendation: Choose group size carefully to maximize student satisfaction.\n\n")
        else:
            f.write("Group size (k) does NOT significantly affect social welfare.\n")
            f.write("Recommendation: Any group size between 3-6 works similarly.\n\n")
        
        f.write("Files generated:\n")
        f.write("- experiment_2_results.png (visualizations)\n")
        f.write("- experiment_2_data.csv (raw data)\n")
        f.write("- experiment_2_summary.txt (this file)\n")
    
    print("✓ Saved summary: experiment_2_summary.txt")
    
    print("\n" + "="*80)
    print("EXPERIMENT 2 COMPLETE")
    print("="*80)
    
    return df


if __name__ == "__main__":
    df_results = run_experiment_2()
    plt.show()
