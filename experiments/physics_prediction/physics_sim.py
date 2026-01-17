"""
2D Physics Simulation for Video Prediction

Clean, reusable physics engine with continuous collision detection.
Designed for generating training data for autoregressive video prediction models.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Ball:
    """Ball with position, velocity, and radius."""
    x: float
    y: float
    vx: float
    vy: float
    radius: float

    def copy(self):
        return Ball(self.x, self.y, self.vx, self.vy, self.radius)


@dataclass
class Barrier:
    """Rectangular barrier."""
    x: float  # Left edge
    y: float  # Bottom edge
    width: float
    height: float

    @property
    def right(self):
        return self.x + self.width

    @property
    def top(self):
        return self.y + self.height

    def copy(self):
        return Barrier(self.x, self.y, self.width, self.height)


class PhysicsSimulation:
    """2D physics simulation with elastic collisions and continuous collision detection."""

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        ball: Optional[Ball] = None,
        barriers: Optional[List[Barrier]] = None,
        elasticity: float = 0.9,
        gravity: float = 0.0,
        wrap_boundaries: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize physics simulation.

        Args:
            width: Simulation width (pixels)
            height: Simulation height (pixels)
            ball: Initial ball state (random if None)
            barriers: List of barriers (empty if None)
            elasticity: Coefficient of restitution (0-1)
            gravity: Gravitational acceleration (pixels/frame^2)
            wrap_boundaries: Wrap at edges if True, bounce if False
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.elasticity = elasticity
        self.gravity = gravity
        self.wrap_boundaries = wrap_boundaries

        if seed is not None:
            np.random.seed(seed)

        # Initialize ball
        if ball is None:
            self.ball = Ball(
                x=np.random.uniform(8, width - 8),
                y=np.random.uniform(8, height - 8),
                vx=np.random.uniform(-2, 2),
                vy=np.random.uniform(-2, 2),
                radius=3.0
            )
        else:
            self.ball = ball.copy()

        # Initialize barriers
        self.barriers = [b.copy() for b in barriers] if barriers else []

    def step(self, dt: float = 1.0):
        """Advance simulation by dt time units with continuous collision detection."""
        # Apply gravity
        self.ball.vy += self.gravity * dt

        remaining_dt = dt
        max_iterations = 10
        iteration = 0

        while remaining_dt > 1e-6 and iteration < max_iterations:
            iteration += 1

            # Find earliest collision
            earliest_t = remaining_dt
            collision_barrier = None
            collision_side = None

            for barrier in self.barriers:
                result = self._find_collision(barrier, remaining_dt)
                if result is not None:
                    t, side = result
                    if t < earliest_t:
                        earliest_t = t
                        collision_barrier = barrier
                        collision_side = side

            # Move to collision point (or end of timestep)
            self.ball.x += self.ball.vx * earliest_t
            self.ball.y += self.ball.vy * earliest_t

            # Handle collision
            if collision_barrier is not None:
                self._reflect_velocity(collision_side)

            remaining_dt -= earliest_t

        # Handle boundary conditions
        self._handle_boundaries()

    def render(self, resolution: int = 64) -> np.ndarray:
        """
        Render current state to grayscale image.

        Returns:
            np.ndarray: Shape (H, W) with values in [0, 1]
                0.0 = background, 0.5 = barrier, 1.0 = ball
        """
        frame = np.zeros((resolution, resolution), dtype=np.float32)

        # Draw barriers
        for barrier in self.barriers:
            x1 = int(np.clip(barrier.x, 0, resolution - 1))
            y1 = int(np.clip(barrier.y, 0, resolution - 1))
            x2 = int(np.clip(barrier.right, 0, resolution))
            y2 = int(np.clip(barrier.top, 0, resolution))
            frame[y1:y2, x1:x2] = 0.5

        # Draw ball (hard edges, no anti-aliasing)
        y_coords, x_coords = np.ogrid[:resolution, :resolution]
        ball_x = self.ball.x % resolution
        ball_y = self.ball.y % resolution
        dist = np.sqrt((x_coords - ball_x)**2 + (y_coords - ball_y)**2)
        mask = (dist <= self.ball.radius).astype(np.float32)
        frame = np.maximum(frame, mask)

        return frame

    def get_state(self) -> dict:
        """Get current simulation state."""
        return {
            'ball': self.ball.copy(),
            'barriers': [b.copy() for b in self.barriers]
        }

    def set_state(self, state: dict):
        """Set simulation state."""
        self.ball = state['ball'].copy()
        self.barriers = [b.copy() for b in state['barriers']]

    def _find_collision(self, barrier: Barrier, dt: float) -> Optional[Tuple[float, str]]:
        """
        Find collision time between circle and rectangle using sweep test.

        Returns:
            (collision_time, side) where side is 'left', 'right', 'top', or 'bottom'
            Returns None if no collision within dt
        """
        # Current ball position and velocity
        bx, by = self.ball.x, self.ball.y
        vx, vy = self.ball.vx, self.ball.vy
        r = self.ball.radius

        # Barrier bounds
        x1, x2 = barrier.x, barrier.right
        y1, y2 = barrier.y, barrier.top

        # Find potential collision times with each side
        collision_times = []

        # Left side (x = x1)
        if vx > 0:  # Moving right
            t = (x1 - r - bx) / vx
            if 0 <= t <= dt:
                # Check if y position at collision is within barrier height
                y_at_t = by + vy * t
                if y1 - r <= y_at_t <= y2 + r:
                    collision_times.append((t, 'left'))

        # Right side (x = x2)
        if vx < 0:  # Moving left
            t = (x2 + r - bx) / vx
            if 0 <= t <= dt:
                y_at_t = by + vy * t
                if y1 - r <= y_at_t <= y2 + r:
                    collision_times.append((t, 'right'))

        # Bottom side (y = y1)
        if vy > 0:  # Moving up
            t = (y1 - r - by) / vy
            if 0 <= t <= dt:
                x_at_t = bx + vx * t
                if x1 - r <= x_at_t <= x2 + r:
                    collision_times.append((t, 'bottom'))

        # Top side (y = y2)
        if vy < 0:  # Moving down
            t = (y2 + r - by) / vy
            if 0 <= t <= dt:
                x_at_t = bx + vx * t
                if x1 - r <= x_at_t <= x2 + r:
                    collision_times.append((t, 'top'))

        # Return earliest collision
        if collision_times:
            return min(collision_times, key=lambda x: x[0])
        return None

    def _reflect_velocity(self, side: str):
        """Reflect velocity based on collision side."""
        if side in ['left', 'right']:
            self.ball.vx = -self.ball.vx * self.elasticity
        else:  # top or bottom
            self.ball.vy = -self.ball.vy * self.elasticity

    def _handle_boundaries(self):
        """Handle boundary conditions (wrap or bounce)."""
        if self.wrap_boundaries:
            # Wrap around edges
            if self.ball.x < 0:
                self.ball.x += self.width
            elif self.ball.x > self.width:
                self.ball.x -= self.width

            if self.ball.y < 0:
                self.ball.y += self.height
            elif self.ball.y > self.height:
                self.ball.y -= self.height
        else:
            # Bounce off boundaries
            if self.ball.x - self.ball.radius < 0:
                self.ball.x = self.ball.radius
                self.ball.vx = -self.ball.vx * self.elasticity
            elif self.ball.x + self.ball.radius > self.width:
                self.ball.x = self.width - self.ball.radius
                self.ball.vx = -self.ball.vx * self.elasticity

            if self.ball.y - self.ball.radius < 0:
                self.ball.y = self.ball.radius
                self.ball.vy = -self.ball.vy * self.elasticity
            elif self.ball.y + self.ball.radius > self.height:
                self.ball.y = self.height - self.ball.radius
                self.ball.vy = -self.ball.vy * self.elasticity


def generate_trajectory(
    sim: PhysicsSimulation,
    num_frames: int,
    resolution: int = 64,
    dt: float = 1.0
) -> np.ndarray:
    """
    Generate trajectory by running simulation.

    Args:
        sim: PhysicsSimulation instance
        num_frames: Number of frames to generate
        resolution: Frame resolution
        dt: Time step per frame (larger = faster motion)

    Returns:
        np.ndarray: Shape (num_frames, H, W)
    """
    frames = []
    for _ in range(num_frames):
        frames.append(sim.render(resolution))
        sim.step(dt=dt)
    return np.array(frames)


def _rects_overlap(x1, y1, w1, h1, x2, y2, w2, h2, margin: float = 0) -> bool:
    """Check if two rectangles overlap (with optional margin)."""
    return not (x1 + w1 + margin < x2 or
                x2 + w2 + margin < x1 or
                y1 + h1 + margin < y2 or
                y2 + h2 + margin < y1)


def _circle_rect_overlap(cx, cy, radius, rx, ry, rw, rh, margin: float = 0) -> bool:
    """Check if a circle overlaps with a rectangle (with optional margin)."""
    # Expand rectangle by radius + margin, then check if center is inside
    expanded_x = rx - radius - margin
    expanded_y = ry - radius - margin
    expanded_w = rw + 2 * (radius + margin)
    expanded_h = rh + 2 * (radius + margin)

    return (expanded_x <= cx <= expanded_x + expanded_w and
            expanded_y <= cy <= expanded_y + expanded_h)


def generate_random_barriers(
    num_barriers: int,
    width: int = 64,
    height: int = 64,
    min_size: int = 5,
    max_size: int = 15,
    margin: float = 2.0,
    exclude_ball: Optional[Tuple[float, float, float]] = None,
    seed: Optional[int] = None
) -> List[Barrier]:
    """
    Generate random non-overlapping barriers.

    Args:
        num_barriers: Number of barriers to generate
        width: Simulation width
        height: Simulation height
        min_size: Minimum barrier dimension
        max_size: Maximum barrier dimension
        margin: Minimum spacing between barriers
        exclude_ball: (x, y, radius) of ball to avoid, or None
        seed: Random seed

    Returns:
        List of Barrier objects
    """
    if seed is not None:
        np.random.seed(seed)

    barriers = []
    max_attempts = 100
    edge_margin = 5  # Keep barriers away from edges

    for _ in range(num_barriers):
        placed = False
        for attempt in range(max_attempts):
            # Random size and position
            w = np.random.randint(min_size, max_size + 1)
            h = np.random.randint(min_size, max_size + 1)
            x = np.random.uniform(edge_margin, width - w - edge_margin)
            y = np.random.uniform(edge_margin, height - h - edge_margin)

            # Check overlap with existing barriers (including margin)
            overlap = False
            for existing in barriers:
                if _rects_overlap(x, y, w, h,
                                  existing.x, existing.y, existing.width, existing.height,
                                  margin=margin):
                    overlap = True
                    break

            # Check overlap with ball exclusion zone
            if not overlap and exclude_ball is not None:
                ball_x, ball_y, ball_radius = exclude_ball
                if _circle_rect_overlap(ball_x, ball_y, ball_radius, x, y, w, h, margin=margin):
                    overlap = True

            if not overlap:
                barriers.append(Barrier(x, y, w, h))
                placed = True
                break

        if not placed:
            # Could not place this barrier, continue with fewer barriers
            pass

    return barriers


def create_random_simulation(
    num_barriers: int = 5,
    with_gravity: bool = False,
    width: int = 64,
    height: int = 64,
    ball_radius: float = 3.0,
    barrier_margin: float = 2.0,
    seed: Optional[int] = None
) -> PhysicsSimulation:
    """
    Create simulation with random terrain and ball position.

    Args:
        num_barriers: Number of barriers
        with_gravity: Whether to include gravity
        width: Simulation width
        height: Simulation height
        ball_radius: Radius of the ball
        barrier_margin: Minimum spacing between barriers and ball
        seed: Random seed

    Returns:
        PhysicsSimulation instance
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate ball position first
    edge_margin = ball_radius + 5
    ball_x = np.random.uniform(edge_margin, width - edge_margin)
    ball_y = np.random.uniform(edge_margin, height - edge_margin)

    # Generate barriers that avoid the ball position
    barriers = generate_random_barriers(
        num_barriers,
        width,
        height,
        margin=barrier_margin,
        exclude_ball=(ball_x, ball_y, ball_radius),
        seed=None  # Don't reset seed, already set above
    )

    # Create ball
    ball = Ball(
        x=ball_x,
        y=ball_y,
        vx=np.random.uniform(-1.5, 1.5),
        vy=np.random.uniform(-1.5, 1.5) if not with_gravity else np.random.uniform(-0.5, 0.5),
        radius=ball_radius
    )

    return PhysicsSimulation(
        width=width,
        height=height,
        ball=ball,
        barriers=barriers,
        elasticity=0.85,
        gravity=0.1 if with_gravity else 0.0,
        wrap_boundaries=not with_gravity,
        seed=seed
    )


def generate_dataset(
    num_trajectories: int,
    num_frames: int,
    num_barriers: int = 5,
    with_gravity: bool = False,
    resolution: int = 64,
    base_seed: int = 42,
    dt: float = 1.0
) -> np.ndarray:
    """
    Generate dataset of trajectories.

    Args:
        num_trajectories: Number of trajectories
        num_frames: Frames per trajectory
        num_barriers: Barriers per simulation
        with_gravity: Whether to include gravity
        resolution: Frame resolution
        base_seed: Base random seed
        dt: Time step per frame (larger = faster motion)

    Returns:
        np.ndarray: Shape (num_trajectories, num_frames, H, W)
    """
    dataset = []

    for i in range(num_trajectories):
        sim = create_random_simulation(
            num_barriers=num_barriers,
            with_gravity=with_gravity,
            seed=base_seed + i
        )
        frames = generate_trajectory(sim, num_frames, resolution, dt=dt)
        dataset.append(frames)

    return np.array(dataset)
