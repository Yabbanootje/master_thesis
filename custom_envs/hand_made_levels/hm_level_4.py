"""Goal level 4."""

# Introduce the required objects
from safety_gymnasium.assets.geoms import Goal, Hazards
# Need to inherit from BaseTask
from safety_gymnasium.bases.base_task import BaseTask
import random


class HMLevel4(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config):
        super().__init__(config=config)

        # Define randomness of the environment
        # If the variable is not assigned specifically to each object
        # then the global area specified here is used by default
        self.placements_conf.extents = [-5, -5, 5, 5]

        goal_location = (0, 0)
        geom_radius = 0.25
        # self.agent.placements = [3, -2, 4, 2]
        self.agent.locations = [(4, 0)]
        # locations = self.randomized_locations(goal_location, geom_radius)
        locations = [(0.5, 0), (0.5, 0.5), (0.5, -0.5)] + self.make_corridor(geom_radius)

        # Instantiate and register the object
        # placement = xmin, ymin, xmax, ymax
        self._add_geoms(Goal(size = geom_radius, keepout = 0, locations=[goal_location]))
        self._add_geoms(Hazards(size = geom_radius, keepout = 0, num = len(locations), locations = locations))

        self._steps = 0
                                                               
        # - in x is to the right
        # - in y is to the top
        # (0, 0) is in the middle

        # Calculate the specific data members needed for the reward
        self.last_dist_goal = None

        self._is_load_static_geoms = False

    def make_corridor(self, hazard_radius):
        locations = []
        for x in range(3):
            locations.append((0.5 + 2 * hazard_radius * x, 2))
            locations.append((0.5 + 2 * hazard_radius * x, -2))
        return locations

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
        # Defining the ideal reward function, which is the goal of the whole task
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        # print("reward of a single step:", reward)
        # print("reward of reaching the goal", self.goal.reward_goal)

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