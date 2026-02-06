"""
Generate graphs from saved incremental calibration data.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the data
data_path = "test_results/incremental_calibration_20260204_1507.json"
with open(data_path, 'r') as f:
    results = json.load(f)

steps = results["steps"]
vocab_sizes = results["vocab_sizes"]
val_sims = results["validation_similarities"]
train_sims = results["training_similarities"]
step_times = results["step_times"]
cumulative_times = results["cumulative_times"]

# Create output directory
os.makedirs("test_results", exist_ok=True)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Vocabulary size vs Validation similarity
axes[0, 0].plot(vocab_sizes, val_sims, 'o-', linewidth=2, markersize=3, color='blue', alpha=0.7)
axes[0, 0].set_xlabel('Vocabulary Size', fontsize=12)
axes[0, 0].set_ylabel('Validation Similarity', fontsize=12)
axes[0, 0].set_title('Validation Similarity vs Vocabulary Size', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale('log')
axes[0, 0].set_ylim([0.95, 1.05])

# Plot 2: Step time vs Vocabulary size (shows why time increases)
axes[0, 1].plot(vocab_sizes, step_times, 'o-', linewidth=2, markersize=3, color='orange', alpha=0.7)
axes[0, 1].set_xlabel('Vocabulary Size (words encoded per step)', fontsize=12)
axes[0, 1].set_ylabel('Step Time (seconds)', fontsize=12)
axes[0, 1].set_title('Step Time vs Vocabulary Size\n(Time increases linearly: each step encodes MORE words)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')

# Add explanation text
axes[0, 1].text(0.05, 0.95, 
                'Why step time increases:\n'
                '• Step 1: Encode 500 words\n'
                '• Step 2: Encode 1000 words (2x)\n'
                '• Step 3: Encode 1500 words (3x)\n'
                '• Step N: Encode N×500 words\n'
                '\n'
                'Time ∝ Vocabulary Size',
                transform=axes[0, 1].transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Cumulative time
axes[1, 0].plot(vocab_sizes, cumulative_times, 'o-', linewidth=2, markersize=3, color='green', alpha=0.7)
axes[1, 0].set_xlabel('Vocabulary Size', fontsize=12)
axes[1, 0].set_ylabel('Cumulative Time (seconds)', fontsize=12)
axes[1, 0].set_title('Cumulative Training Time', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xscale('log')
# Convert to minutes for readability
cumulative_minutes = [t / 60 for t in cumulative_times]
ax1_twin = axes[1, 0].twinx()
ax1_twin.plot(vocab_sizes, cumulative_minutes, '--', linewidth=1, color='darkgreen', alpha=0.5)
ax1_twin.set_ylabel('Cumulative Time (minutes)', fontsize=10, color='darkgreen')
ax1_twin.tick_params(axis='y', labelcolor='darkgreen')

# Plot 4: Training vs Validation similarity
axes[1, 1].plot(vocab_sizes, train_sims, 'o-', linewidth=2, markersize=3, label='Training', color='purple', alpha=0.7)
axes[1, 1].plot(vocab_sizes, val_sims, 'o-', linewidth=2, markersize=3, label='Validation', color='blue', alpha=0.7)
axes[1, 1].set_xlabel('Vocabulary Size', fontsize=12)
axes[1, 1].set_ylabel('Similarity', fontsize=12)
axes[1, 1].set_title('Training vs Validation Similarity', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='best')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xscale('log')
axes[1, 1].set_ylim([0.95, 1.05])

plt.suptitle('Incremental Calibration Results: 50K Vocabulary Test\n(90 steps, validating every 500 words)', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plot_path = "test_results/incremental_calibration_graphs.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved graph to: {plot_path}")

# Also create a detailed step time analysis plot
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(vocab_sizes, step_times, 'o-', linewidth=2, markersize=4, color='coral', label='Actual Step Time')
ax.set_xlabel('Vocabulary Size (words encoded in this step)', fontsize=12)
ax.set_ylabel('Step Time (seconds)', fontsize=12)
ax.set_title('Why Step Time Increases: Each Step Encodes More Vocabulary', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add linear fit line to show proportionality
import numpy as np
z = np.polyfit(vocab_sizes, step_times, 1)
p = np.poly1d(z)
ax.plot(vocab_sizes, p(vocab_sizes), '--', color='red', linewidth=2, 
        label=f'Linear fit: time = {z[0]:.4f} × vocab_size + {z[1]:.2f}')

ax.legend()
ax.text(0.02, 0.98,
        'Key Insight:\n'
        'Step time increases because each step processes\n'
        'MORE vocabulary than the previous step.\n'
        '\n'
        'This is EXPECTED behavior:\n'
        '• Step 1: Process 500 words → ~0.5s\n'
        '• Step 2: Process 1000 words → ~1.0s\n'
        '• Step 3: Process 1500 words → ~1.5s\n'
        '• Step N: Process N×500 words → ~N×0.5s\n'
        '\n'
        'Time complexity: O(vocab_size)',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plot_path2 = "test_results/step_time_analysis.png"
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
print(f"✓ Saved step time analysis to: {plot_path2}")

plt.close('all')
print("\n✓ All graphs generated successfully!")
