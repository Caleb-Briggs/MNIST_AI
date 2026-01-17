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
            collision_normal = None

            for barrier in self.barriers:
                result = self._circle_rect_collision_time(self.ball, barrier, remaining_dt)
                if result is not None:
                    t, normal = result
                    if t < earliest_t:
                        earliest_t = t
                        collision_barrier = barrier
                        collision_normal = normal

            # Move to collision point (or end of timestep)
            self.ball.x += self.ball.vx * earliest_t
            self.ball.y += self.ball.vy * earliest_t

            # Handle collision
            if collision_barrier is not None:
                self._handle_barrier_collision(collision_barrier, collision_normal)

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

    def _circle_rect_collision_time(
        self,
        ball: Ball,
        barrier: Barrier,
        dt: float
    ) -> Optional[Tuple[float, str]]:
        """
        Find collision time between moving circle and static rectangle.

        Returns:
            (collision_time, normal_direction) or None
            normal_direction: 'left', 'right', 'top', or 'bottom'
        """
        # Expand rectangle by ball radius (treat ball as point)
        expanded = Barrier(
            barrier.x - ball.radius,
            barrier.y - ball.radius,
            barrier.width + 2 * ball.radius,
            barrier.height + 2 * ball.radius
        )

        t_min = 0.0
        t_max = dt
        collision_normal = None

        # X axis
        if ball.vx != 0:
            t1 = (expanded.x - ball.x) / ball.vx
            t2 = (expanded.right - ball.x) / ball.vx

            if t1 > t2:
                t1, t2 = t2, t1
                entering_from = 'right'
            else:
                entering_from = 'left'

            if t1 > t_min:
                t_min = t1
                collision_normal = entering_from
            if t2 < t_max:
                t_max = t2
        else:
            if ball.x < expanded.x or ball.x > expanded.right:
                return None

        # Y axis
        if ball.vy != 0:
            t1 = (expanded.y - ball.y) / ball.vy
            t2 = (expanded.top - ball.y) / ball.vy

            if t1 > t2:
                t1, t2 = t2, t1
                entering_from_y = 'top'
            else:
                entering_from_y = 'bottom'

            if t1 > t_min:
                t_min = t1
                collision_normal = entering_from_y
            if t2 < t_max:
                t_max = t2
        else:
            if ball.y < expanded.y or ball.y > expanded.top:
                return None

        # Check if intersection is valid
        if t_min > t_max or t_max < 0 or t_min > dt:
            return None

        if t_min >= 0:
            return (t_min, collision_normal)

        return None

    def _handle_barrier_collision(self, barrier: Barrier, normal: str):
        """Handle collision with barrier by reflecting velocity."""
        epsilon = 0.01  # Prevent immediate re-collision

        if normal in ['left', 'right']:
            self.ball.vx = -self.ball.vx * self.elasticity
            if normal == 'left':
                self.ball.x = barrier.x - self.ball.radius - epsilon
            else:
                self.ball.x = barrier.right + self.ball.radius + epsilon
        else:  # top or bottom
            self.ball.vy = -self.ball.vy * self.elasticity
            if normal == 'bottom':
                self.ball.y = barrier.y - self.ball.radius - epsilon
            else:
                self.ball.y = barrier.top + self.ball.radius + epsilon

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
    resolution: int = 64
) -> np.ndarray:
    """
    Generate trajectory by running simulation.

    Args:
        sim: PhysicsSimulation instance
        num_frames: Number of frames to generate
        resolution: Frame resolution

    Returns:
        np.ndarray: Shape (num_frames, H, W)
    """
    frames = []
    for _ in range(num_frames):
        frames.append(sim.render(resolution))
        sim.step()
    return np.array(frames)


def generate_random_barriers(
    num_barriers: int,
    width: int = 64,
    height: int = 64,
    min_size: int = 5,
    max_size: int = 15,
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
        seed: Random seed

    Returns:
        List of Barrier objects
    """
    if seed is not None:
        np.random.seed(seed)

    barriers = []
    max_attempts = 100

    for _ in range(num_barriers):
        for attempt in range(max_attempts):
            # Random size and position
            w = np.random.randint(min_size, max_size + 1)
            h = np.random.randint(min_size, max_size + 1)
            x = np.random.uniform(5, width - w - 5)
            y = np.random.uniform(5, height - h - 5)

            new_barrier = Barrier(x, y, w, h)

            # Check overlap
            overlap = False
            for existing in barriers:
                if not (new_barrier.right < existing.x or
                       new_barrier.x > existing.right or
                       new_barrier.top < existing.y or
                       new_barrier.y > existing.top):
                    overlap = True
                    break

            if not overlap:
                barriers.append(new_barrier)
                break

    return barriers


def create_random_simulation(
    num_barriers: int = 5,
    with_gravity: bool = False,
    width: int = 64,
    height: int = 64,
    seed: Optional[int] = None
) -> PhysicsSimulation:
    """
    Create simulation with random terrain and ball position.

    Args:
        num_barriers: Number of barriers
        with_gravity: Whether to include gravity
        width: Simulation width
        height: Simulation height
        seed: Random seed

    Returns:
        PhysicsSimulation instance
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate barriers
    barriers = generate_random_barriers(num_barriers, width, height, seed=seed)

    # Find valid ball position (not inside barriers)
    max_attempts = 100
    ball_radius = 3.0

    for _ in range(max_attempts):
        ball_x = np.random.uniform(10, width - 10)
        ball_y = np.random.uniform(10, height - 10)

        # Check if ball overlaps with any barrier
        valid = True
        for barrier in barriers:
            if (barrier.x - ball_radius <= ball_x <= barrier.right + ball_radius and
                barrier.y - ball_radius <= ball_y <= barrier.top + ball_radius):
                valid = False
                break

        if valid:
            break

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
    base_seed: int = 42
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
        frames = generate_trajectory(sim, num_frames, resolution)
        dataset.append(frames)

    return np.array(dataset)
