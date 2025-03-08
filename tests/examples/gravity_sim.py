import sys
import time
import random
import numpy as np
import pygame

from ecs.world import World
from ecs.component import Component
from ecs.system import System

FPS = 60
PHYSICS_UPDATE_MULTIPLIER = 10

# -----------------------------------------------------------------------------
# Component Definitions
# -----------------------------------------------------------------------------


class Position(Component):
    """component for 2D positions."""
    @property
    def dimensions(self) -> int:
        return 2


class Velocity(Component):
    """component for 2D velocities."""
    @property
    def dimensions(self) -> int:
        return 2


class Mass(Component):
    """component for scalar mass values."""
    @property
    def dimensions(self) -> int:
        return 1


class Renderable(Component):
    """component for rendering data. Stored data: (R, G, B, radius) """
    @property
    def dimensions(self) -> int:
        return 4


class Locked(Component):
    """component indicating that the entity's position is locked."""
    @property
    def dimensions(self) -> int:
        return 1


# -----------------------------------------------------------------------------
# System Definitions
# -----------------------------------------------------------------------------

class GravitySystem(System):
    """
    Vectorized gravity system for n‑body simulation.

    Uses hybrid components for Position, Velocity, and Mass.
    All physics calculations are performed using NumPy vectorized operations.
    """
    group = "physics"

    def update(self, world: World, dt: float) -> None:
        G = 6.67430e-3  # Gravitational constant

        pos_comp = world.get_component_instance(Position)
        vel_comp = world.get_component_instance(Velocity)
        mass_comp = world.get_component_instance(Mass)
        n = pos_comp.size
        if n == 0:
            return

        # Retrieve physics data from dense arrays.
        positions = pos_comp.array[:n]         # shape: (n, 2)
        velocities = vel_comp.array[:n]        # shape: (n, 2)
        masses = mass_comp.array[:n].flatten()    # shape: (n,)

        # Compute pairwise differences: diff[i, j] = positions[j] - positions[i]
        diff = positions[None, :, :] - positions[:, None, :]  # shape: (n, n, 2)
        eps = 1e-3  # Avoid division by zero
        dist_sq = np.sum(diff ** 2, axis=2) + eps  # shape: (n, n)
        dist = np.sqrt(dist_sq)  # shape: (n, n)

        # Compute force magnitudes: F = G * m_i * m_j / r^2.
        force_mag = G * (masses[:, None] * masses[None, :]) / dist_sq  # shape: (n, n)
        force_dir = diff / dist[:, :, None]  # shape: (n, n, 2)
        forces = force_mag[:, :, None] * force_dir  # shape: (n, n, 2)

        # Zero self-interaction.
        np.fill_diagonal(forces[:, :, 0], 0)
        np.fill_diagonal(forces[:, :, 1], 0)

        net_force = np.sum(forces, axis=1)  # shape: (n, 2)
        acceleration = net_force / masses[:, None]  # shape: (n, 2)

        new_velocities = velocities + acceleration * dt
        new_positions = positions + new_velocities * dt

        # For each entity, if it is locked, do not update its data.
        for entity_id, comps in world.query([Position]):
            # Check if entity has Locked component.
            if Locked in world.entity_components.get(entity_id, {}):
                # For locked entities, restore previous velocity and position.
                # (Alternatively, you could simply skip writing updates.)
                idx = pos_comp.entity_to_index.get(entity_id)
                if idx is not None:
                    new_velocities[idx] = velocities[idx]
                    new_positions[idx] = positions[idx]
        vel_comp.array[:n] = new_velocities
        pos_comp.array[:n] = new_positions


class RenderSystem(System):
    """Render entities using their Position and Renderable data."""
    group = "render"

    def __init__(self, screen: pygame.Surface):
        super().__init__()
        self.screen = screen

    def update(self, world: World, dt: float) -> None:
        pos_comp = world.get_component_instance(Position)
        rend_comp = world.get_component_instance(Renderable)
        n = rend_comp.size
        for i in range(n):
            pos = pos_comp.array[i]
            rend_data = rend_comp.array[i]
            color = (int(rend_data[0]), int(rend_data[1]), int(rend_data[2]))
            radius = int(rend_data[3])
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), radius)


class CleanupSystem(System):
    """
    Removes entities that are farther than a threshold from the screen center.
    """
    group = "physics"

    def update(self, world: World, dt: float) -> None:
        pos_comp = world.get_component_instance(Position)
        if pos_comp.size == 0:
            return
        center = np.array([400, 300])
        to_remove = []
        # Iterate over all entities via the component's mapping.
        for entity_id, idx in pos_comp.entity_to_index.items():
            pos = pos_comp.array[idx]
            if np.linalg.norm(pos - center) > 500:
                to_remove.append(entity_id)
        for entity_id in to_remove:
            world.remove_entity(entity_id)


# -----------------------------------------------------------------------------
# UI Helper Functions
# -----------------------------------------------------------------------------

def draw_slider(screen: pygame.Surface, slider_rect: pygame.Rect,
                handle_pos: int) -> None:
    pygame.draw.rect(screen, (100, 100, 100), slider_rect)
    handle_rect = pygame.Rect(0, 0, 10, slider_rect.height + 4)
    handle_rect.centerx = handle_pos
    handle_rect.centery = slider_rect.centery
    pygame.draw.rect(screen, (200, 200, 200), handle_rect)


def slider_value_from_pos(x: int, slider_rect: pygame.Rect, min_val: float,
                          max_val: float) -> float:
    rel_x = x - slider_rect.left
    ratio = max(0, min(1, rel_x / slider_rect.width))
    return min_val + ratio * (max_val - min_val)


def slider_handle_pos(value: float, slider_rect: pygame.Rect, min_val: float,
                      max_val: float) -> int:
    ratio = (value - min_val) / (max_val - min_val)
    return int(slider_rect.left + ratio * slider_rect.width)


# -----------------------------------------------------------------------------
# Main Simulation Loop and Interaction
# -----------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((1000, 600))
    pygame.display.set_caption("N-Body Gravity Simulator")
    clock = pygame.time.Clock()

    world = World()

    # Register components.
    world.register_component(Position)
    world.register_component(Velocity)
    world.register_component(Mass)
    world.register_component(Renderable)
    world.register_component(Locked)

    # Register systems.
    world.register_system(GravitySystem())
    world.register_system(RenderSystem(screen))
    world.register_system(CleanupSystem())

    # Variables for planet creation.
    dragging = False
    start_pos = (0, 0)
    current_drag_pos = (0, 0)
    velocity_scale = 0.05

    # UI: Slider for planet size (radius).
    slider_rect = pygame.Rect(820, 50, 150, 20)
    min_radius = 2
    max_radius = 20
    selected_radius = 5  # Initial radius.

    # UI: Checkbox for "Lock" mode.
    lock_checkbox_rect = pygame.Rect(820, 100, 20, 20)
    lock_enabled = False

    slider_dragging = False

    # FPS counter font.
    font = pygame.font.SysFont(None, 24)

    # Physics update: <PHYSICS_UPDATE_MULTIPLIER> times per frame.
    physics_dt = 1 / (PHYSICS_UPDATE_MULTIPLIER * FPS)
    accumulator = 0.0
    last_time = time.perf_counter()

    running = True
    while running:
        now = time.perf_counter()
        frame_dt = now - last_time
        last_time = now
        accumulator += frame_dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx >= 800:
                    # Check if clicking on slider.
                    if slider_rect.collidepoint(mx, my):
                        slider_dragging = True
                        selected_radius = slider_value_from_pos(mx, slider_rect, min_radius, max_radius)
                    # Check if clicking on lock checkbox.
                    elif lock_checkbox_rect.collidepoint(mx, my):
                        lock_enabled = not lock_enabled
                else:
                    dragging = True
                    start_pos = pygame.mouse.get_pos()
                    current_drag_pos = start_pos

            elif event.type == pygame.MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()
                if slider_dragging:
                    selected_radius = slider_value_from_pos(mx, slider_rect, min_radius, max_radius)
                elif dragging:
                    current_drag_pos = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                if slider_dragging:
                    slider_dragging = False
                elif dragging:
                    dragging = False
                    end_pos = pygame.mouse.get_pos()
                    vx = (end_pos[0] - start_pos[0]) * velocity_scale
                    vy = (end_pos[1] - start_pos[1]) * velocity_scale
                    r = random.randint(100, 255)
                    g = random.randint(100, 255)
                    b = random.randint(100, 255)
                    mass = (selected_radius ** 3) * 500  # mass ∝ radius³
                    # Build list of component types.
                    comp_types = [Position, Velocity, Mass, Renderable]
                    if lock_enabled:
                        comp_types.append(Locked)
                    init_data = {
                        Position: start_pos,
                        Velocity: (vx, vy),
                        Mass: (mass,),
                        Renderable: (r, g, b, int(selected_radius))
                    }
                    if lock_enabled:
                        init_data[Locked] = (1,)  # Arbitrary value indicating locked.
                    world.create_entity(comp_types, component_initial_data=init_data)

        # Fixed-step physics update.
        while accumulator >= physics_dt:
            world.update(physics_dt, group="physics")
            accumulator -= physics_dt

        # Render update.
        screen.fill((0, 0, 0))
        pygame.draw.line(screen, (50, 50, 50), (800, 0), (800, 600), 2)
        world.update(frame_dt, group="render")

        # Draw UI panel.
        ui_rect = pygame.Rect(800, 0, 200, 600)
        pygame.draw.rect(screen, (30, 30, 30), ui_rect)
        # Slider label.
        label = font.render("Object Size", True, (200, 200, 200))
        screen.blit(label, (slider_rect.left, slider_rect.top - 25))
        handle_x = slider_handle_pos(selected_radius, slider_rect, min_radius, max_radius)
        draw_slider(screen, slider_rect, handle_x)
        value_label = font.render(f"{int(selected_radius)}", True, (200, 200, 200))
        screen.blit(value_label, (slider_rect.right + 10, slider_rect.top))
        # Draw lock checkbox.
        pygame.draw.rect(screen, (100, 100, 100), lock_checkbox_rect)
        if lock_enabled:
            pygame.draw.line(screen, (200, 200, 200), lock_checkbox_rect.topleft, lock_checkbox_rect.bottomright, 2)
            pygame.draw.line(screen, (200, 200, 200), lock_checkbox_rect.topright, lock_checkbox_rect.bottomleft, 2)
        lock_label = font.render("Lock", True, (200, 200, 200))
        screen.blit(lock_label, (lock_checkbox_rect.right + 5, lock_checkbox_rect.top))

        # Draw drag indicator.
        if dragging:
            pygame.draw.line(screen, (255, 255, 255), start_pos, current_drag_pos, 2)
            pygame.draw.circle(screen, (255, 255, 255), start_pos, int(selected_radius))

        # FPS counter.
        fps_text = font.render(
            f"FPS: {int(clock.get_fps())}, "
            f"Entities: {world.num_entities}", True, (255, 255, 255))
        screen.blit(fps_text, (820, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
