"""
Visual test - save trajectory as images to inspect visually.
"""

import numpy as np
import matplotlib.pyplot as plt
from physics_sim import Ball, Barrier, PhysicsSimulation, generate_trajectory

# Test 1: Simple head-on collision
print("Generating test trajectory...")
ball = Ball(x=10, y=32, vx=2.0, vy=0.0, radius=3)
barrier = Barrier(x=30, y=28, width=10, height=8)

sim = PhysicsSimulation(
    width=64,
    height=64,
    ball=ball,
    barriers=[barrier],
    elasticity=1.0,
    gravity=0.0,
    wrap_boundaries=False,
    seed=42
)

frames = generate_trajectory(sim, num_frames=15, resolution=64)

# Display frames
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(frames[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Frame {i}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('results/test_collision.png', dpi=150, bbox_inches='tight')
print("Saved to results/test_collision.png")

# Test 2: Diagonal collision
ball2 = Ball(x=10, y=10, vx=1.5, vy=1.5, radius=3)
barrier2 = Barrier(x=25, y=25, width=10, height=10)

sim2 = PhysicsSimulation(
    width=64,
    height=64,
    ball=ball2,
    barriers=[barrier2],
    elasticity=0.9,
    gravity=0.0,
    wrap_boundaries=False,
    seed=42
)

frames2 = generate_trajectory(sim2, num_frames=15, resolution=64)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(frames2[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Frame {i}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('results/test_diagonal.png', dpi=150, bbox_inches='tight')
print("Saved to results/test_diagonal.png")

print("\nDone! Check the PNG files to see if collisions look correct.")
