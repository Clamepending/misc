import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Define noise levels (variance schedule)
noise_levels = np.linspace(0, 1, 40)
t_steps = noise_levels

# Initial point (Dirac delta location)
x_0 = 0.0

# Generate sample paths
n_paths = 1000

# Black color for all paths with transparency
path_color = 'black'
path_alpha = 0.1

# ========== DDPM (Stochastic) ==========
ax1.set_title('DDPM: Stochastic Sampling', fontsize=14, fontweight='bold')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Noise Level (t)', fontsize=12)
ax1.set_ylim(0, 1)
ax1.set_xlim(-4, 4)

# Simulate FORWARD diffusion (random walk), then plot in reverse
for path_idx in range(n_paths):
    path_x = []
    path_t = []
    
    # Start from x_0 at t=0 (clean data)
    current_x = x_0
    
    # Forward process: go from t=0 to t=1 (add noise)
    for i in range(len(t_steps)):
        t = t_steps[i]
        path_x.append(current_x)
        path_t.append(t)
        
        if i < len(t_steps) - 1:
            # Add noise for forward diffusion (random walk)
            dt = t_steps[i+1] - t_steps[i]
            # Standard Brownian motion increment
            noise = np.random.randn() * np.sqrt(dt)
            current_x = current_x + noise
    
    # Plot the path (already goes from t=0 to t=1, showing forward diffusion)
    # But we want to visualize reverse process, so reverse the arrays
    ax1.plot(path_x[::-1], path_t[::-1], color=path_color, alpha=path_alpha, linewidth=1.5, zorder=4)

# ========== DDIM (Deterministic) ==========
ax2.set_title('DDIM: Deterministic Sampling', fontsize=14, fontweight='bold')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_xlim(-4, 4)

# Generate deterministic reverse paths (from noise to data)
# Sample starting points from unit Gaussian
starting_points = np.random.randn(n_paths)

for path_idx in range(n_paths):
    path_x = []
    path_t = []
    
    # Start from unit Gaussian at t=1
    current_x = starting_points[path_idx]
    
    # Deterministic reverse process: straight line to origin
    for i, t in enumerate(t_steps[::-1]):
        # Linear interpolation from noise to data
        # At t=1: x stays as sampled
        # At t=0: x moves to x_0
        x_deterministic = current_x * np.sqrt(t) if t > 0 else x_0
        path_x.append(x_deterministic)
        path_t.append(t)
    
    # Plot deterministic path (straight line)
    ax2.plot(path_x, path_t, color=path_color, alpha=path_alpha, linewidth=1.5, zorder=4)

# Add arrow for reverse process direction
ax2.annotate('', xy=(3.5, 0.05), xytext=(3.5, 0.95),
             arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax2.text(3.6, 0.5, 'Reverse Process', ha='left', va='center', fontsize=10, color='gray', rotation=90)

# Add main title
fig.suptitle('DDPM vs DDIM: Stochastic vs Deterministic Sampling Paths', fontsize=16, fontweight='bold', y=0.98)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color='black', alpha=0.3, linewidth=2, label='Sample trajectories')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.05))

# Add caption
fig.text(0.5, -0.12, 
         'Key Insight: DDPM uses stochastic noise in the reverse process (left) while DDIM uses\n' +
         'deterministic paths (right), enabling faster and more controllable inference.',
         ha='center', fontsize=11, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
# You may need to change this save path
# plt.savefig('ddpm_ddim_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# print("Visualization saved as ddpm_ddim_comparison.png")