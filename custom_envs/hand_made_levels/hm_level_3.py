"""Goal level 3."""

# Introduce the required objects
from safety_gymnasium.assets.geoms import Goal, Hazards
# Need to inherit from HMLevelBase
from custom_envs.hand_made_levels.hm_level_base import HMLevelBase
import random


class HMLevel3(HMLevelBase):
    """An agent must navigate to a goal."""

    def __init__(self, config):
        super().__init__(config=config)

        # locations = self.randomized_locations(goal_location, geom_radius)
        self.locations = [(0.5, 0), (0.5, 0.5), (0.5, -0.5)]

        # Instantiate and register the object
        # placement = xmin, ymin, xmax, ymax
        self._add_geoms(Hazards(size = self.geom_radius, keepout = 0.5, num = len(self.locations), locations = self.locations))

    def randomized_locations(self, goal_position, hazard_radius):
        # initialize the static corners relative to the goal position
        step_size = 2 * hazard_radius
        corners = [(-step_size, -step_size), (step_size, step_size), (-step_size, step_size), (step_size, -step_size)]
        corners = [tuple(map(sum, zip(pos, goal_position))) for pos in corners]

        # choose a random entry point to the goal
        possible_entry_points = [(0.0, -step_size), (0.0, step_size), (-step_size, 0.0), (step_size, 0.0)]
        entry_point = random.choice(possible_entry_points)
        possible_entry_points.remove(entry_point)
        closed_entry_points = [tuple(map(sum, zip(pos, goal_position))) for pos in possible_entry_points]
        
        # build a wall of hazards on the side of the entry point,
        # so that the agent cannot get to the entry point in a straight line
        middle_wall = tuple([3 * coord for coord in entry_point])
        if middle_wall[0] == 0.0:
            wall_positions = [tuple(map(sum, zip((i * step_size, 0.0), middle_wall))) for i in range(-1,2)]
        else:
            wall_positions = [tuple(map(sum, zip((0.0, i * step_size), middle_wall))) for i in range(-1,2)]
        wall_positions = [tuple(map(sum, zip(pos, goal_position))) for pos in wall_positions]

        return corners + closed_entry_points + wall_positions

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = super().calculate_reward()

        return reward

    def specific_reset(self):
        # Task-specific reset mechanism
        # Called at env.reset()
        # Used to reset specific member variables
        super().specific_reset()

    def specific_step(self):
        # Task-specific step mechanism
        # Called at env.step()
        # Used to change the value of member variables over time
        super().specific_step()

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # Called when env.reset() or self.goal_achieved==True
        # Used to periodically refresh the layout or state of the environment
        super().update_world()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # Determine if the goal is reached, called at env.step()
        return super().goal_achieved

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        # Define the location of the target
        # If there is a goal in the environment, the same position as the goal
        # Can be undefined
        return super().goal_pos