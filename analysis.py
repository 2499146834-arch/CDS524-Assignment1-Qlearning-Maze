"""
Analysis and visualization for Q-learning maze experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from trainer import Trainer


plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

# Shared improved defaults for all experiments
DEFAULT_TRAINER_KWARGS = dict(
    maze_size=10,
    step_penalty=-1,
    optimistic_init=50.0,
    distance_reward=0.0,
    epsilon=0.9,
    epsilon_decay=0.997,
    epsilon_min=0.01,
    max_steps=400,
    consecutive_success=30,
    optimal_step_threshold=60,
)


def plot_convergence(metrics_df, title="Q-Learning Training Convergence",
                     save_path=None):
    """Plot training convergence: steps, reward, epsilon, consecutive success."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    episodes = metrics_df['episode'].values
    n = len(episodes)

    # Panel 1: Steps per episode
    ax = axes[0][0]
    ax.plot(episodes, metrics_df['steps'].values, linewidth=0.7, color='#2196F3', alpha=0.9)
    tail = metrics_df['steps'].values[-min(100, n):]
    ax.axhline(y=tail.mean(), color='red', linestyle='--', linewidth=0.8,
               label=f"Final avg: {tail.mean():.1f}")
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel 2: Cumulative reward per episode
    ax = axes[0][1]
    ax.plot(episodes, metrics_df['reward'].values, linewidth=0.7, color='#4CAF50', alpha=0.9)
    tail_r = metrics_df['reward'].values[-min(100, n):]
    ax.axhline(y=tail_r.mean(), color='red', linestyle='--', linewidth=0.8,
               label=f"Final avg: {tail_r.mean():.0f}")
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Reward per Episode')
    ax.legend(fontsize=8)

    # Panel 3: Epsilon decay
    ax = axes[1][0]
    ax.plot(episodes, metrics_df['epsilon'].values, linewidth=1.0, color='#FF9800')
    ax.set_ylabel('Epsilon')
    ax.set_xlabel('Episode')
    ax.set_title('Exploration Rate (Exponential Decay)')
    ax.set_ylim(0, 1.05)

    # Panel 4: Consecutive success
    ax = axes[1][1]
    ax.plot(episodes, metrics_df['consecutive_success'].values, linewidth=0.8, color='#9C27B0')
    ax.axhline(y=30, color='green', linestyle='--', linewidth=0.8, label='Convergence threshold')
    ax.set_ylabel('Consecutive Success')
    ax.set_xlabel('Episode')
    ax.set_title('Consecutive Optimal Solves')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def compare_fixed_vs_random(max_episodes=3000, seed=42, save_path=None):
    """Compare training on a fixed maze vs. random maze each episode."""
    print("=" * 55)
    print("Experiment: Fixed Maze vs Random Maze")
    print("=" * 55)

    results = {}
    for mode in ['fixed', 'random']:
        label = "Fixed Maze" if mode == 'fixed' else "Random Maze (new each ep)"
        print(f"\nTraining {label}...")
        t = Trainer(maze_mode=mode, seed=seed, **DEFAULT_TRAINER_KWARGS)
        t.train(max_episodes=max_episodes, verbose=True, verbose_interval=400)
        results[mode] = t

    # Evaluation
    print("\n" + "=" * 55)
    print("Evaluation (100 test episodes, epsilon=0):")
    for mode, trainer in results.items():
        r = trainer.evaluate(num_episodes=100)
        label = "Fixed Maze" if mode == 'fixed' else "Random Maze"
        print(f"  {label}: avg_steps={r['avg_steps']:.1f}, "
              f"success={r['success_rate']:.1%}, optimal={r['optimal_rate']:.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fixed Maze vs Random Maze Training", fontsize=13, fontweight='bold')
    colors = {'fixed': '#2196F3', 'random': '#FF5722'}
    labels = {'fixed': 'Fixed Maze', 'random': 'Random Maze (new per episode)'}

    for mode, trainer in results.items():
        df = trainer.get_metrics_dataframe()
        axes[0].plot(df['episode'], df['steps'], linewidth=0.6,
                     color=colors[mode], alpha=0.8, label=labels[mode])
        axes[1].plot(df['episode'], df['reward'], linewidth=0.6,
                     color=colors[mode], alpha=0.8, label=labels[mode])

    for ax in axes:
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Steps')
    axes[0].set_title('Steps per Episode')
    axes[1].set_ylabel('Cumulative Reward')
    axes[1].set_title('Reward per Episode')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    return results


def hyperparameter_comparison(max_episodes=3000, seed=42, save_path=None):
    """Grid search over key hyperparameters."""
    param_grid = [
        # Format: (alpha, gamma, eps_decay, optimistic_init, distance_reward, label)
        (0.10, 0.90, 0.997, 50.0, 0.0, "Baseline: α=0.1 γ=0.90 opt=50"),
        (0.05, 0.90, 0.997, 50.0, 0.0, "α=0.05 (slower learning)"),
        (0.30, 0.90, 0.997, 50.0, 0.0, "α=0.30 (faster learning)"),
        (0.10, 0.95, 0.997, 50.0, 0.0, "γ=0.95 (more far-sighted)"),
        (0.10, 0.90, 0.995, 50.0, 0.0, "ε_d=0.995 (faster decay)"),
        (0.10, 0.90, 0.999, 50.0, 0.0, "ε_d=0.999 (slower decay)"),
        (0.10, 0.90, 0.997, 30.0, 0.0, "opt init=30 (lower optimism)"),
        (0.10, 0.90, 0.997,  0.0, 0.0, "opt init=0 (zero init, original)"),
    ]

    print("=" * 55)
    print("Hyperparameter Comparison")
    print("=" * 55)

    all_metrics = []
    for alpha, gamma, eps_decay, opt_init, dist_r, label in param_grid:
        print(f"\n{label}")
        t = Trainer(
            maze_size=10, maze_mode='fixed',
            lr=alpha, gamma=gamma, epsilon=0.9,
            epsilon_decay=eps_decay, epsilon_min=0.01,
            optimistic_init=opt_init, distance_reward=dist_r,
            step_penalty=-1, max_steps=400,
            consecutive_success=30, optimal_step_threshold=60,
            seed=seed,
        )
        metrics = t.train(max_episodes=max_episodes, verbose=False)

        # Compute summary statistics
        last200 = metrics[-200:]
        successful = [m for m in last200 if m['success']]
        final_steps = np.mean([m['steps'] for m in successful]) if successful else 400
        success_rate = np.mean([m['success'] for m in last200])
        final_reward = np.mean([m['reward'] for m in last200])
        # Approximate convergence: first episode with 20 consecutive successes
        conv_ep = next((m['episode'] for m in metrics
                        if m['consecutive_success'] >= 20), max_episodes)

        all_metrics.append({
            'label': label,
            'lr': alpha, 'gamma': gamma, 'eps_decay': eps_decay,
            'dist_reward': dist_r, 'optim_init': opt_init,
            'final_avg_steps': final_steps,
            'final_avg_reward': final_reward,
            'success_rate': success_rate,
            'approx_convergence_ep': conv_ep,
            'raw_metrics': metrics,
        })
        print(f"  Avg steps (successful): {final_steps:.1f} | "
              f"Success rate: {success_rate:.1%} | Conv ep: {conv_ep}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Label':<45} {'Steps':>7} {'Success':>8} {'Conv Ep':>8}")
    print("-" * 90)
    for m in all_metrics:
        short = m['label'].split('(')[0].strip()[:43]
        print(f"{short:<45} {m['final_avg_steps']:>7.1f} {m['success_rate']:>7.1%} "
              f"{m['approx_convergence_ep']:>8}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hyperparameter Comparison", fontsize=13, fontweight='bold')
    cmap = plt.cm.tab10

    for i, m in enumerate(all_metrics):
        c = cmap(i % 10)
        episodes = [e['episode'] for e in m['raw_metrics']]
        axes[0][0].plot(episodes, [e['steps'] for e in m['raw_metrics']],
                        linewidth=0.6, color=c, alpha=0.8, label=m['label'])
        axes[0][1].plot(episodes, [e['reward'] for e in m['raw_metrics']],
                        linewidth=0.6, color=c, alpha=0.8, label=m['label'])

    axes[0][0].set_ylabel('Steps')
    axes[0][0].set_title('Steps per Episode')
    axes[0][0].legend(fontsize=6, loc='upper right')
    axes[0][1].set_ylabel('Cumulative Reward')
    axes[0][1].set_title('Reward per Episode')
    axes[0][1].legend(fontsize=6, loc='lower right')

    # Bar charts
    short_labels = [m['label'].split('(')[0].strip()[:35] for m in all_metrics]
    colors = [cmap(i % 10) for i in range(len(all_metrics))]

    axes[1][0].bar(range(len(all_metrics)), [m['final_avg_steps'] for m in all_metrics], color=colors)
    axes[1][0].set_xticks(range(len(all_metrics)))
    axes[1][0].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=6)
    axes[1][0].set_ylabel('Avg Steps (successful)')
    axes[1][0].set_title('Final Step Efficiency')

    axes[1][1].bar(range(len(all_metrics)), [m['success_rate'] for m in all_metrics], color=colors)
    axes[1][1].set_xticks(range(len(all_metrics)))
    axes[1][1].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=6)
    axes[1][1].set_ylabel('Success Rate')
    axes[1][1].set_title('Final Success Rate')
    axes[1][1].set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    plt.show()

    return all_metrics


def run_pygame_visualization(trainer, fps=10):
    """Launch pygame to visualize a trained agent solving the maze."""
    from visualizer import run_visualization
    run_visualization(trainer, fps=fps)


if __name__ == "__main__":
    print("Analysis module loaded.")
    print("Functions: plot_convergence(), compare_fixed_vs_random(),")
    print("           hyperparameter_comparison(), run_pygame_visualization()")
