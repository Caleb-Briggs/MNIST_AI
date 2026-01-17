"""
Test the exact scenario from the demo notebook that shows teleporting.
"""

import numpy as np
import matplotlib.pyplot as plt
from physics_sim import Ball, Barrier, PhysicsSimulation, generate_trajectory

# Exact code from demo notebook "Basic Usage" section
ball = Ball(x=10, y=10, vx=1.5, vy=0.8, radius=3)
barrier = Barrier(x=25, y=15, width=15, height=3)

sim = PhysicsSimulation(
    ball=ball,
    barriers=[barrier],
    elasticity=0.9,
    gravity=0.0,
    wrap_boundaries=True,
    seed=42
)

# Generate trajectory
frames = generate_trajectory(sim, num_frames=100)

print(f"Generated {len(frames)} frames, shape: {frames.shape}")
print(f"Values: min={frames.min()}, max={frames.max()}")

# Now manually step through and log positions
print("\n" + "="*60)
print("Manual step-by-step simulation:")
print("="*60)

sim2 = PhysicsSimulation(
    ball=Ball(x=10, y=10, vx=1.5, vy=0.8, radius=3),
    barriers=[Barrier(x=25, y=15, width=15, height=3)],
    elasticity=0.9,
    gravity=0.0,
    wrap_boundaries=True,
    seed=42
)

print(f"Barrier: x={25}, y={15}, width={15}, height={3}")
print(f"Barrier bounds: x=[{25}, {25+15}], y=[{15}, {15+3}]")
print()

for step in range(30):
    prev_x, prev_y = sim2.ball.x, sim2.ball.y
    prev_vx, prev_vy = sim2.ball.vx, sim2.ball.vy

    sim2.step(dt=1.0)

    dx = sim2.ball.x - prev_x
    dy = sim2.ball.y - prev_y
    dist = np.sqrt(dx**2 + dy**2)
    max_dist = np.sqrt(prev_vx**2 + prev_vy**2) * 1.0

    # Check if near barrier
    near_barrier = (22 <= sim2.ball.x <= 43 and 12 <= sim2.ball.y <= 21)
    marker = " <-- NEAR BARRIER" if near_barrier else ""

    # Check for velocity change (collision)
    vx_changed = abs(sim2.ball.vx - prev_vx) > 0.01
    vy_changed = abs(sim2.ball.vy - prev_vy) > 0.01
    collision = " [COLLISION]" if (vx_changed or vy_changed) else ""

    # Check for teleport
    teleport = " *** TELEPORT ***" if dist > max_dist + 0.1 else ""

    print(f"Step {step:2d}: x={sim2.ball.x:6.2f}, y={sim2.ball.y:6.2f}, " +
          f"vx={sim2.ball.vx:6.2f}, vy={sim2.ball.vy:6.2f}, " +
          f"moved={dist:5.2f}/{max_dist:5.2f}{collision}{marker}{teleport}")

print("\n" + "="*60)
print("Saving visualization...")
print("="*60)

# Save first 20 frames
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(frames[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Frame {i}')
    ax.axis('off')

    # Draw barrier bounds for reference
    ax.axhline(y=15, color='red', linestyle=':', alpha=0.5, linewidth=0.5)
    ax.axhline(y=18, color='red', linestyle=':', alpha=0.5, linewidth=0.5)
    ax.axvline(x=25, color='red', linestyle=':', alpha=0.5, linewidth=0.5)
    ax.axvline(x=40, color='red', linestyle=':', alpha=0.5, linewidth=0.5)

plt.tight_layout()
plt.savefig('results/demo_scenario.png', dpi=150, bbox_inches='tight')
print("Saved to results/demo_scenario.png")
