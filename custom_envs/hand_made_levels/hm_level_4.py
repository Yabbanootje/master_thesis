"""Goal level 4."""

# Introduce the required objects
from safety_gymnasium.assets.geoms import Goal, Hazards
# Need to inherit from HMLevelBase
from custom_envs.hand_made_levels.hm_level_base import HMLevelBase


class HMLevel4(HMLevelBase):
    """An agent must navigate to a goal."""

    def __init__(self, config):
        super().__init__(config=config)
        self._add_geoms(Goal(size = self.geom_radius, keepout = 0, locations=[self.goal_location], reward_goal=self.goal_reward))
        self.goal.reward_distance = self._goal_reward_distance

        self.locations = [(0.5, 0), (0.5, 0.5), (0.5, -0.5), (0.5, 0.25), (0.5, -0.25)]

        # Instantiate and register the object
        # placement = xmin, ymin, xmax, ymax
        self._add_geoms(Hazards(size = self.geom_radius, keepout = 0, num = len(self.locations), locations = self.locations))

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