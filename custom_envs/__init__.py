from custom_envs.curriculum_level_0 import CurriculumLevel0
from safety_gymnasium.utils.registration import register

register(id="Curriculum0-v0", entry_point="custom_envs.curriculum_level_0:CurriculumLevel0")