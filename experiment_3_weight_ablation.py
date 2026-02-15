"""
Experiment 3: Utility Weight Ablation Study
Purpose: Isolate the contribution of each utility component
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from coalition_formation import ClassroomModel
import random
from typing import Dict, Tuple
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
    
    # Calculate total welfare
    total_welfare = model.compute_total_welfare()
    
    # Calculate component-specific scores
    total_complementarity = 0
    total_social = 0
    total_role_balance = 0
    student_count = 0
    
    for entity in model.entities:
        for student in entity.members:
            comp = entity.calculate_skill_complementarity(student, entity.members)
            social = entity.calculate_social_satisfaction(student, entity.members)
            role = entity.calculate_role_balance(student, entity.members)
            
            total_complementarity += comp
            total_social += social
            total_role_balance += role
            student_count += 1
    
    avg_complementarity = total_complementarity / student_count if student_count > 0 else 0
    avg_social = total_social / student_count if student_count > 0 else 0
    avg_role_balance = total_role_balance / student_count if student_count > 0 else 0
    
    # Group composition quality
    locked_entities = [e for e in model.entities if e.state == "locked"]
    avg_group_size = np.mean([e.size for e in locked_entities]) if locked_entities else 0
    
    # Leader ratios in final groups
    leader_ratios = [e.get_leader_ratio() for e in model.entities]
    avg_leader_ratio = np.mean(leader_ratios) if leader_ratios else 0
    
    return {
        'ws': ws,
        'wf': wf,
        'wp': wp,
        'weight_config': f"({ws:.1f}, {wf:.1f}, {wp:.1f})",
        'total_welfare': total_welfare,
        'welfare_per_student': total_welfare / n_students if n_students > 0 else 0,
        'avg_complementarity': avg_complementarity,
        'avg_social_satisfaction': avg_social,
        'avg_role_balance': avg_role_balance,
        'convergence_steps': steps,
        'avg_group_size': avg_group_size,
        'avg_leader_ratio': avg_leader_ratio,
        'num_final_groups': len(model.entities),
        'seed': seed
    }


def run_experiment_3():
    """Run Experiment 3: Utility Weight Ablation Study."""
    print("="*80)
    print("EXPERIMENT 3: UTILITY WEIGHT ABLATION STUDY")
    print("="*80)
    print("\nIsolating contributions of each utility component")
    print("Fixed parameters: n_students=24, k=4")
    print("Varied parameter: Weight configurations (ws, wf, wp)")
    print("Runs per configuration: 30")
    print("\n")
    
    # Experimental parameters
    n_students = 24
    k = 4
    runs_per_config = 30
    
    # Weight configurations: (ws, wf, wp, label)
    weight_configs = [
        (1.0, 0.0, 0.0, "Skills-only"),
        (0.0, 1.0, 0.0, "Social-only"),
        (0.0, 0.0, 1.0, "Personality-only"),
        (0.5, 0.5, 0.0, "Skills+Social"),
        (0.5, 0.0, 0.5, "Skills+Personality"),
        (0.0, 0.5, 0.5, "Social+Personality"),
        (0.33, 0.33, 0.33, "Balanced"),
        (0.4, 0.3, 0.3, "Default")
    ]
    
    results = []
    total_runs = len(weight_configs) * runs_per_config
    current_run = 0
    
    for ws, wf, wp, label in weight_configs:
        print(f"\nRunning {label}: ws={ws:.2f}, wf={wf:.2f}, wp={wp:.2f} ({runs_per_config} runs)...")
        for run in range(runs_per_config):
            current_run += 1
            seed = 3000 + current_run
            
            result = run_single_simulation(n_students, k, ws, wf, wp, seed)
            result['config_label'] = label
            results.append(result)
            
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/{runs_per_config} runs ({current_run}/{total_runs} total)")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Statistical Analysis: One-Way ANOVA
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: ONE-WAY ANOVA")
    print("="*80)
    
    # Test if weight configuration affects total welfare
    config_labels = [label for _, _, _, label in weight_configs]
    groups_welfare = [df[df['config_label'] == label]['total_welfare'].values 
                     for label in config_labels]
    f_stat_welfare, p_value_welfare = stats.f_oneway(*groups_welfare)
    
    print(f"\nOverall Social Welfare (Total Classroom):")
    print(f"  F-statistic: {f_stat_welfare:.4f}")
    print(f"  p-value: {p_value_welfare:.4e}")
    print(f"  Significant: {'YES' if p_value_welfare < 0.05 else 'NO'} (α=0.05)")
    
    # Post-hoc: Pairwise comparisons for top configurations
    print("\n--- Post-hoc Analysis: Top 3 Configurations ---")
    top3_labels = df.groupby('config_label')['total_welfare'].mean().nlargest(3).index.tolist()
    
    for i, label1 in enumerate(top3_labels):
        for label2 in top3_labels[i+1:]:
            group1 = df[df['config_label'] == label1]['total_welfare'].values
            group2 = df[df['config_label'] == label2]['total_welfare'].values
            t_stat, p_val = stats.ttest_ind(group1, group2)
            print(f"\n{label1} vs {label2}:")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
            print(f"  Significant difference: {'YES' if p_val < 0.05 else 'NO'}")
    
    # Test if weight configuration affects convergence time
    groups_convergence = [df[df['config_label'] == label]['convergence_steps'].values 
                         for label in config_labels]
    f_stat_conv, p_value_conv = stats.f_oneway(*groups_convergence)
    
    print(f"\n\nConvergence Steps:")
    print(f"  F-statistic: {f_stat_conv:.4f}")
    print(f"  p-value: {p_value_conv:.4e}")
    print(f"  Significant: {'YES' if p_value_conv < 0.05 else 'NO'} (α=0.05)")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY WEIGHT CONFIGURATION")
    print("="*80)
    
    summary = df.groupby('config_label').agg({
        'total_welfare': ['mean', 'std'],
        'avg_complementarity': ['mean'],
        'avg_social_satisfaction': ['mean'],
        'avg_role_balance': ['mean'],
        'convergence_steps': ['mean', 'std']
    }).round(3)
    
    # Sort by welfare
    summary = summary.sort_values(('total_welfare', 'mean'), ascending=False)
    print("\n", summary)
    
    # Best configuration
    best_config = df.groupby('config_label')['total_welfare'].mean().idxmax()
    best_welfare = df.groupby('config_label')['total_welfare'].mean().max()
    
    print("\n" + "="*80)
    print(f"OPTIMAL CONFIGURATION: {best_config}")
    print(f"Average Overall Social Welfare: {best_welfare:.3f}")
    print("="*80)
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Sort configurations by welfare for better visualization
    config_order = df.groupby('config_label')['total_welfare'].mean().sort_values(ascending=False).index
    
    # Calculate mean welfare per configuration
    mean_welfare = df.groupby('config_label')['total_welfare'].mean().loc[config_order]
    std_welfare = df.groupby('config_label')['total_welfare'].std().loc[config_order]
    
    # Create single bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(config_order)), mean_welfare.values, 
                   color='steelblue',
                   edgecolor='black', linewidth=1.2)
    
    ax.set_yticks(range(len(config_order)))
    ax.set_yticklabels(config_order, fontsize=11)
    ax.set_xlabel('Average Overall Social Welfare', fontsize=13, fontweight='bold')
    ax.set_ylabel('Weight Configuration', fontsize=13, fontweight='bold')
    ax.set_title('Overall Social Welfare by Weight Configuration', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('experiment_3_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: experiment_3_results.png")
    
    # Save data
    df.to_csv('experiment_3_data.csv', index=False)
    print("✓ Saved data: experiment_3_data.csv")
    
    # Save summary report
    with open('experiment_3_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT 3: UTILITY WEIGHT ABLATION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("Which factors (skills, social, personality) matter most for welfare?\n\n")
        
        f.write("EXPERIMENTAL SETUP:\n")
        f.write("- Fixed parameters: n_students=24, k=4\n")
        f.write("- Varied: Weight configurations (ws, wf, wp)\n")
        f.write("- Runs per configuration: 30\n")
        f.write("- Total simulations: 240\n\n")
        
        f.write("WEIGHT CONFIGURATIONS TESTED:\n")
        for ws, wf, wp, label in weight_configs:
            f.write(f"- {label:20s}: ws={ws:.2f}, wf={wf:.2f}, wp={wp:.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        # Rank configurations by welfare
        ranking = df.groupby('config_label')['total_welfare'].mean().sort_values(ascending=False)
        
        f.write("1. RANKING BY OVERALL SOCIAL WELFARE (BEST TO WORST):\n\n")
        for i, (label, welfare) in enumerate(ranking.items(), 1):
            config_data = df[df['config_label'] == label]
            std = config_data['total_welfare'].std()
            conv = config_data['convergence_steps'].mean()
            f.write(f"   {i}. {label:20s}: {welfare:.3f} ± {std:.3f} ")
            f.write(f"(converges in {conv:.0f} steps)\n")
        
        best_config = ranking.index[0]
        worst_config = ranking.index[-1]
        improvement = ((ranking.iloc[0] - ranking.iloc[-1]) / ranking.iloc[-1]) * 100
        
        f.write(f"\n   → Best beats worst by {improvement:.1f}%\n\n")
        
        f.write(f"2. STATISTICAL SIGNIFICANCE:\n")
        f.write(f"   - Welfare: F={f_stat_welfare:.2f}, p={p_value_welfare:.2e} ")
        f.write(f"({'SIGNIFICANT' if p_value_welfare < 0.05 else 'NOT SIGNIFICANT'})\n")
        f.write(f"   - Convergence: F={f_stat_conv:.2f}, p={p_value_conv:.2e} ")
        f.write(f"({'SIGNIFICANT' if p_value_conv < 0.05 else 'NOT SIGNIFICANT'})\n\n")
        
        f.write(f"3. COMPONENT CONTRIBUTIONS:\n\n")
        for label in ranking.index[:3]:  # Top 3
            config_data = df[df['config_label'] == label]
            comp = config_data['avg_complementarity'].mean()
            social = config_data['avg_social_satisfaction'].mean()
            role = config_data['avg_role_balance'].mean()
            f.write(f"   {label}:\n")
            f.write(f"   - Skill complementarity: {comp:.3f}\n")
            f.write(f"   - Social satisfaction: {social:.3f}\n")
            f.write(f"   - Role balance: {role:.3f}\n\n")
        
        f.write(f"4. CONVERGENCE SPEED:\n\n")
        conv_ranking = df.groupby('config_label')['convergence_steps'].mean().sort_values()
        f.write(f"   Fastest: {conv_ranking.index[0]} ({conv_ranking.iloc[0]:.0f} steps)\n")
        f.write(f"   Slowest: {conv_ranking.index[-1]} ({conv_ranking.iloc[-1]:.0f} steps)\n")
        f.write(f"   Difference: {((conv_ranking.iloc[-1] - conv_ranking.iloc[0]) / conv_ranking.iloc[0] * 100):.1f}% slower\n\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        if p_value_welfare < 0.05:
            f.write("Weight configuration has a SIGNIFICANT effect on social welfare.\n\n")
            f.write(f"RECOMMENDATION: Use '{best_config}' configuration for optimal welfare.\n\n")
        else:
            f.write("Weight configuration does NOT significantly affect welfare.\n\n")
            f.write("RECOMMENDATION: Any reasonable weight combination works similarly.\n\n")
        
        # Analyze single-component vs combined
        single_component = ['Skills-only', 'Social-only', 'Personality-only']
        combined = ['Skills+Social', 'Skills+Personality', 'Social+Personality', 'Balanced', 'Default']
        
        single_avg = df[df['config_label'].isin(single_component)]['total_welfare'].mean()
        combined_avg = df[df['config_label'].isin(combined)]['total_welfare'].mean()
        
        f.write(f"SINGLE vs COMBINED COMPONENTS:\n")
        f.write(f"- Single component average: {single_avg:.3f}\n")
        f.write(f"- Combined components average: {combined_avg:.3f}\n")
        if combined_avg > single_avg:
            f.write(f"- Combined is {((combined_avg - single_avg) / single_avg * 100):.1f}% better\n")
            f.write("→ Conclusion: Balancing multiple factors improves outcomes.\n")
        else:
            f.write("→ Conclusion: Focusing on single factors can be effective.\n")
        
        f.write("\nFiles generated:\n")
        f.write("- experiment_3_results.png (visualizations)\n")
        f.write("- experiment_3_data.csv (raw data)\n")
        f.write("- experiment_3_summary.txt (this file)\n")
    
    print("✓ Saved summary: experiment_3_summary.txt")
    
    print("\n" + "="*80)
    print("EXPERIMENT 3 COMPLETE")
    print("="*80)
    
    return df


if __name__ == "__main__":
    df_results = run_experiment_3()
    plt.show()
