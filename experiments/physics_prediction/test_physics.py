"""
Test script for physics simulation - with detailed logging to debug collision issues.
"""

import numpy as np
from physics_sim import Ball, Barrier, PhysicsSimulation

def test_simple_collision():
    """Test a simple head-on collision with a barrier."""
    print("\n" + "="*60)
    print("TEST 1: Simple head-on collision")
    print("="*60)

    # Ball moving right toward a barrier
    ball = Ball(x=10, y=32, vx=2.0, vy=0.0, radius=3)
    barrier = Barrier(x=30, y=28, width=10, height=8)

    sim = PhysicsSimulation(
        width=64,
        height=64,
        ball=ball,
        barriers=[barrier],
        elasticity=1.0,  # Perfect elastic for testing
        gravity=0.0,
        wrap_boundaries=False,
        seed=42
    )

    print(f"Initial state:")
    print(f"  Ball: x={sim.ball.x:.2f}, y={sim.ball.y:.2f}, vx={sim.ball.vx:.2f}, vy={sim.ball.vy:.2f}")
    print(f"  Barrier: x={barrier.x}, y={barrier.y}, width={barrier.width}, height={barrier.height}")
    print(f"  Expected collision at x={barrier.x - ball.radius} = {barrier.x - ball.radius}")
    print(f"  Expected collision time t={(barrier.x - ball.radius - ball.x) / ball.vx:.2f}")

    # Step and watch
    for step in range(10):
        prev_x = sim.ball.x
        prev_vx = sim.ball.vx
        sim.step(dt=1.0)
        print(f"Step {step+1}: x={sim.ball.x:.2f}, y={sim.ball.y:.2f}, vx={sim.ball.vx:.2f}, vy={sim.ball.vy:.2f}")

        # Check for teleporting
        distance_moved = abs(sim.ball.x - prev_x)
        max_possible = abs(prev_vx) * 1.0
        if distance_moved > max_possible + 0.1:  # Small epsilon for numerical errors
            print(f"  WARNING: Ball teleported! Moved {distance_moved:.2f} but max possible was {max_possible:.2f}")

        # Check if ball went through barrier
        if barrier.x <= sim.ball.x <= barrier.right and barrier.y <= sim.ball.y <= barrier.top:
            print(f"  ERROR: Ball is inside barrier!")
            return False

    print("✓ Test passed - no teleporting or clipping detected")
    return True

def test_corner_approach():
    """Test approaching a barrier corner."""
    print("\n" + "="*60)
    print("TEST 2: Corner approach")
    print("="*60)

    # Ball approaching barrier corner at an angle
    ball = Ball(x=10, y=10, vx=1.5, vy=1.5, radius=3)
    barrier = Barrier(x=25, y=25, width=10, height=10)

    sim = PhysicsSimulation(
        width=64,
        height=64,
        ball=ball,
        barriers=[barrier],
        elasticity=0.9,
        gravity=0.0,
        wrap_boundaries=False,
        seed=42
    )

    print(f"Initial state:")
    print(f"  Ball: x={sim.ball.x:.2f}, y={sim.ball.y:.2f}, vx={sim.ball.vx:.2f}, vy={sim.ball.vy:.2f}")
    print(f"  Barrier: x={barrier.x}, y={barrier.y}, width={barrier.width}, height={barrier.height}")

    for step in range(15):
        prev_x, prev_y = sim.ball.x, sim.ball.y
        prev_vx, prev_vy = sim.ball.vx, sim.ball.vy
        sim.step(dt=1.0)

        dx = sim.ball.x - prev_x
        dy = sim.ball.y - prev_y
        distance_moved = np.sqrt(dx**2 + dy**2)
        max_possible = np.sqrt(prev_vx**2 + prev_vy**2) * 1.0

        print(f"Step {step+1}: x={sim.ball.x:.2f}, y={sim.ball.y:.2f}, vx={sim.ball.vx:.2f}, vy={sim.ball.vy:.2f}")

        if distance_moved > max_possible + 0.1:
            print(f"  WARNING: Ball teleported! Moved {distance_moved:.2f} but max possible was {max_possible:.2f}")

        # Check penetration
        if (barrier.x - ball.radius < sim.ball.x < barrier.right + ball.radius and
            barrier.y - ball.radius < sim.ball.y < barrier.top + ball.radius):
            # Ball is near barrier, check if it's actually inside
            if (barrier.x < sim.ball.x < barrier.right and
                barrier.y < sim.ball.y < barrier.top):
                print(f"  ERROR: Ball penetrated barrier!")
                return False

    print("✓ Test passed")
    return True

def test_detailed_collision():
    """Test with detailed logging of collision detection."""
    print("\n" + "="*60)
    print("TEST 3: Detailed collision logging")
    print("="*60)

    ball = Ball(x=15, y=32, vx=3.0, vy=0.0, radius=3)
    barrier = Barrier(x=35, y=28, width=8, height=8)

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

    print(f"Ball at x={ball.x:.2f}, moving right with vx={ball.vx:.2f}")
    print(f"Barrier left edge at x={barrier.x}")
    print(f"Ball should hit at x={barrier.x - ball.radius} = {barrier.x - ball.radius:.2f}")
    print(f"Expected collision in {(barrier.x - ball.radius - ball.x) / ball.vx:.2f} steps")

    # Manually check collision detection
    result = sim._find_collision(barrier, dt=10.0)
    if result:
        t, side = result
        print(f"\nCollision detected: t={t:.2f}, side={side}")
        print(f"Ball position at collision: x={ball.x + ball.vx * t:.2f}, y={ball.y + ball.vy * t:.2f}")
    else:
        print("\nNo collision detected - BUG!")
        return False

    # Now step and watch
    for step in range(8):
        prev_state = (sim.ball.x, sim.ball.y, sim.ball.vx, sim.ball.vy)
        sim.step(dt=1.0)
        print(f"Step {step+1}: x={sim.ball.x:.2f}, y={sim.ball.y:.2f}, vx={sim.ball.vx:.2f}, vy={sim.ball.vy:.2f}")

        # Check if velocity flipped (collision happened)
        if np.sign(sim.ball.vx) != np.sign(prev_state[2]):
            print(f"  → Collision! Velocity flipped from {prev_state[2]:.2f} to {sim.ball.vx:.2f}")

    print("✓ Test passed")
    return True

def test_gravity_bounce():
    """Test gravity with ground bounce."""
    print("\n" + "="*60)
    print("TEST 4: Gravity bounce")
    print("="*60)

    ball = Ball(x=32, y=50, vx=0.0, vy=0.0, radius=3)

    sim = PhysicsSimulation(
        width=64,
        height=64,
        ball=ball,
        barriers=[],
        elasticity=0.9,
        gravity=0.2,
        wrap_boundaries=False,
        seed=42
    )

    print(f"Ball starting at y={ball.y:.2f}, should fall and bounce off ground (y=0)")

    for step in range(20):
        prev_y = sim.ball.y
        prev_vy = sim.ball.vy
        sim.step(dt=1.0)
        print(f"Step {step+1}: y={sim.ball.y:.2f}, vy={sim.ball.vy:.2f}")

        # Check if ball went below ground
        if sim.ball.y < ball.radius:
            print(f"  WARNING: Ball below ground! y={sim.ball.y:.2f}, radius={ball.radius}")

        # Check for bounce
        if prev_vy < 0 and sim.ball.vy > 0:
            print(f"  → Bounced! vy changed from {prev_vy:.2f} to {sim.ball.vy:.2f}")

    print("✓ Test passed")
    return True

def test_position_continuity():
    """Test that position changes are continuous - no jumps."""
    print("\n" + "="*60)
    print("TEST 5: Position continuity check")
    print("="*60)

    ball = Ball(x=10, y=32, vx=2.0, vy=0.5, radius=3)
    barrier = Barrier(x=28, y=28, width=8, height=8)

    sim = PhysicsSimulation(
        width=64,
        height=64,
        ball=ball,
        barriers=[barrier],
        elasticity=0.9,
        gravity=0.0,
        wrap_boundaries=False,
        seed=42
    )

    print(f"Testing that ball never moves more than velocity*dt in a single step")
    print(f"Initial: x={sim.ball.x:.2f}, y={sim.ball.y:.2f}, vx={sim.ball.vx:.2f}, vy={sim.ball.vy:.2f}")

    dt = 1.0
    max_violation = 0.0

    for step in range(20):
        prev_x, prev_y = sim.ball.x, sim.ball.y
        prev_vx, prev_vy = sim.ball.vx, sim.ball.vy

        sim.step(dt=dt)

        dx = sim.ball.x - prev_x
        dy = sim.ball.y - prev_y
        distance_moved = np.sqrt(dx**2 + dy**2)

        # Maximum distance that could be moved in one timestep
        # This is the INITIAL velocity * dt (before any collision)
        max_distance = np.sqrt(prev_vx**2 + prev_vy**2) * dt

        print(f"Step {step+1}: x={sim.ball.x:.2f}, y={sim.ball.y:.2f}, " +
              f"moved={distance_moved:.2f}, max_allowed={max_distance:.2f}, " +
              f"vx={sim.ball.vx:.2f}, vy={sim.ball.vy:.2f}")

        # Check if ball moved too far
        violation = distance_moved - max_distance
        if violation > 0.01:  # Small epsilon for numerical errors
            print(f"  ERROR: Ball moved {distance_moved:.2f} but max should be {max_distance:.2f}!")
            print(f"  This is a jump of {violation:.2f} pixels!")
            max_violation = max(max_violation, violation)

    if max_violation > 0.01:
        print(f"\n✗ Test FAILED: Max violation was {max_violation:.2f} pixels")
        return False
    else:
        print(f"\n✓ Test passed - no discontinuous jumps")
        return True

if __name__ == "__main__":
    print("\nRunning physics tests...\n")

    results = []
    results.append(("Simple collision", test_simple_collision()))
    results.append(("Corner approach", test_corner_approach()))
    results.append(("Detailed collision", test_detailed_collision()))
    results.append(("Gravity bounce", test_gravity_bounce()))
    results.append(("Position continuity", test_position_continuity()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed - physics needs fixing")
