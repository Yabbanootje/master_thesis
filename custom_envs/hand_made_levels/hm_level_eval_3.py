"""Goal level 3."""

# Introduce the required objects
from safety_gymnasium.assets.geoms import Goal, Hazards
# Need to inherit from HMLevelBase
from custom_envs.hand_made_levels.hm_level_3 import HMLevel3
import random


class HMLevelEval3(HMLevel3):
    """An agent must navigate to a goal."""

    def __init__(self, config):
        super().__init__(config=config)
        
        self.agent.placements = None
        self.agent.locations = [(2.5, 0)]