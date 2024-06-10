"""Goal level 0."""

# Introduce the required objects
from safety_gymnasium.assets.geoms import Goal, Hazards, IndicationArea
# Need to inherit from BaseTask
from safety_gymnasium.bases.base_task import BaseTask

import random
import string

class HMLevelBase(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config):
        super().__init__(config=config)

        # Define randomness of the environment
        # If the variable is not assigned specifically to each object
        # then the global area specified here is used by default
        self.placements_conf.extents = [-5, -5, 5, 5]

        self.goal_location = (0, 0)
        self.geom_radius = 0.25
        self.goal_reward = 20
        self.step_penalty = 0
        self.lidar_conf.max_dist = 4
        self.agent.placements = [(2, -3, 3, 2),
                                 (-2, 2, 3, 3),
                                 (-3, -2, -2, 3),
                                 (-3, -3, 2, -2),
                                 # Bias towards difficult starting positions
                                 (2, -2, 3, 2),
                                 (2, -1.5, 3, 1.5),
                                 (2, -1, 3, 1),
                                 (2, -0.5, 3, 0.5),
                                 ]
        # self.agent.locations = [(2.5, 0)]

        # Show start locations
        # start_areas = [(2, -3, 3, 2),
        #                          (-2, 2, 3, 3),
        #                          (-3, -2, -2, 3),
        #                          (-3, -3, 2, -2),
        #                          # Bias towards difficult starting positions
        #                          (2, -2, 3, 2),
        #                          (2, -1.5, 3, 1.5),
        #                          (2, -1, 3, 1),
        #                          (2, -0.5, 3, 0.5),
        #                          ]
        # for i, (x_min, y_min, x_max, y_max) in enumerate(start_areas):
        #     x_width = (x_max - x_min) / 2
        #     y_width = (y_max - y_min) / 2
        #     x_pos = (x_max + x_min) / 2
        #     y_pos = (y_max + y_min) / 2
        #     print(x_width, y_width, x_pos, y_pos)
        #     print("inidication_area"+str(i))
        #     random.seed(i)
        #     name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        #     self._add_geoms(IndicationArea(name=name, num=1, keepout=-100, locations=[(x_pos, y_pos)], x_width=abs(x_width), y_width=abs(y_width)))
        
        self._steps = 0

        # self._add_geoms(Goal(size = self.geom_radius, keepout = 0, locations=[self.goal_location], reward_goal=self.goal_reward))
        self._goal_reward_distance = 1
        # - in x is to the right
        # - in y is to the top
        # (0, 0) is in the middle

        # Calculate the specific data members needed for the reward
        self.last_dist_goal = None

        self._is_load_static_geoms = False

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # Defining the ideal reward function, which is the goal of the whole task
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        # Punishing longer routes
        reward -= self.step_penalty

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        # Task-specific reset mechanism
        # Called at env.reset()
        # Used to reset specific member variables
        # print("-------- resetting in safety gymnasium 2 with nr. of steps:", self._steps)
        # if self._steps > 100:
        #     print("trying to change geoms during step")
        #     self._geoms = {}
        #     self._add_geoms(Goal(keepout = 0, locations=[(0, 0)])) # placements=[(-0.1, -0.1, 0.1, 0.1)]))
        pass

    def specific_step(self):
        # Task-specific step mechanism
        # Called at env.step()
        # Used to change the value of member variables over time
        self._steps += 1
        # if self._steps == 10:
        #     print("trying to change agent location during step")
            # self.agent.locations = [(-0.5, 0)]

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # Called when env.reset() or self.goal_achieved==True
        # Used to periodically refresh the layout or state of the environment
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # Determine if the goal is reached, called at env.step()
        return self.dist_goal() <= self.goal.size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        # Define the location of the target
        # If there is a goal in the environment, the same position as the goal
        # Can be undefined
        return self.goal.pos