# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Safety-Gymnasium Environments.

This file lets us make environment from our own tasks, by using the combine function with
our tasks and the agents we want to use.
"""

import copy

from gymnasium import make as gymnasium_make
from gymnasium import register as gymnasium_register

from safety_gymnasium import vector, wrappers
# from safety_gymnasium.tasks.safe_multi_agent.tasks.velocity.safe_mujoco_multi import make_ma
from safety_gymnasium.utils.registration import make, register
from safety_gymnasium.version import __version__

from safety_gymnasium.builder import Builder
from safety_gymnasium.utils.task_utils import get_task_class_name

from custom_envs.preliminary_levels.curriculum_level_0 import CurriculumLevel0
from custom_envs.preliminary_levels.curriculum_level_1 import CurriculumLevel1
from custom_envs.preliminary_levels.curriculum_level_2 import CurriculumLevel2

from custom_envs.hand_made_levels.hm_level_0 import HMLevel0
from custom_envs.hand_made_levels.hm_level_1 import HMLevel1
from custom_envs.hand_made_levels.hm_level_2 import HMLevel2
from custom_envs.hand_made_levels.hm_level_3 import HMLevel3
from custom_envs.hand_made_levels.hm_level_4 import HMLevel4
from custom_envs.hand_made_levels.hm_level_5 import HMLevel5
from custom_envs.hand_made_levels.hm_level_target import HMLevelTarget

from custom_envs.hand_made_levels.hm_eval_level_3 import HMEvalLevel3

from custom_envs.hand_made_levels_reward.hmr_level_0 import HMRLevel0
from custom_envs.hand_made_levels_reward.hmr_level_1 import HMRLevel1
from custom_envs.hand_made_levels_reward.hmr_level_2 import HMRLevel2
from custom_envs.hand_made_levels_reward.hmr_level_3 import HMRLevel3
from custom_envs.hand_made_levels_reward.hmr_level_4 import HMRLevel4
from custom_envs.hand_made_levels_reward.hmr_level_5 import HMRLevel5

__all__ = [
    'register',
    'make',
    'gymnasium_make',
    'gymnasium_register',
]

VERSION = 'v0'
ROBOT_NAMES = ('Point', 'Car', 'Doggo', 'Racecar', 'Ant')
MAKE_VISION_ENVIRONMENTS = True
MAKE_DEBUG_ENVIRONMENTS = True

# ========================================
# Helper Methods for Easy Registration
# ========================================

PREFIX = 'Safety'

robots = ROBOT_NAMES


def __register_helper(env_id, entry_point, spec_kwargs=None, **kwargs):
    """Register a environment to both Safety-Gymnasium and Gymnasium registry."""
    env_name, dash, version = env_id.partition('-')
    if spec_kwargs is None:
        spec_kwargs = {}

    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=spec_kwargs,
        **kwargs,
    )
    gymnasium_register(
        id=f'{env_name}Gymnasium{dash}{version}',
        entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{env_name}Gymnasium{dash}{version}', **copy.deepcopy(spec_kwargs)},
        **kwargs,
    )


def __combine(tasks, agents, max_episode_steps, builder_name):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        # Vector inputs
        for robot_name in agents:
            env_id = f'{PREFIX}{robot_name}{task_name}-{VERSION}'
            combined_config = copy.deepcopy(task_config)
            combined_config.update({'agent_name': robot_name})

            __register_helper(
                env_id=env_id,
                # entry_point='safety_gymnasium.builder:Builder',
                entry_point='custom_envs.__init__:' + builder_name,
                spec_kwargs={'config': combined_config, 'task_id': env_id},
                max_episode_steps=max_episode_steps,
            )

            if MAKE_VISION_ENVIRONMENTS:
                # Vision inputs
                vision_env_name = f'{PREFIX}{robot_name}{task_name}Vision-{VERSION}'
                vision_config = {
                    'observe_vision': True,
                    'observation_flatten': False,
                }
                vision_config.update(combined_config)
                __register_helper(
                    env_id=vision_env_name,
                    entry_point='safety_gymnasium.builder:Builder',
                    spec_kwargs={'config': vision_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )

            if MAKE_DEBUG_ENVIRONMENTS and robot_name in ['Point', 'Car', 'Racecar']:
                # Keyboard inputs for debugging
                debug_env_name = f'{PREFIX}{robot_name}{task_name}Debug-{VERSION}'
                debug_config = {'debug': True}
                debug_config.update(combined_config)
                __register_helper(
                    env_id=debug_env_name,
                    entry_point='safety_gymnasium.builder:Builder',
                    spec_kwargs={'config': debug_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )

# Goal Environments
# ----------------------------------------
preliminary_goal_tasks = {'Curriculum0': {}, 'Curriculum1': {}, 'Curriculum2': {}}

class CustomBuilderPreliminary(Builder):
    # def __init__(self):
    #     super.__init__()
    #     self.tasks = {

    #     }

    def _get_task_from_string(self, class_name):
        if class_name == "CurriculumLevel0":
            return CurriculumLevel0
        elif class_name == "CurriculumLevel1":
            return CurriculumLevel1
        elif class_name == "CurriculumLevel2":
            return CurriculumLevel2
        else:
            return super()._get_task()

    def _get_task(self):
            class_name = get_task_class_name(self.task_id)
            task_class = self._get_task_from_string(class_name)
            task = task_class(config=self.config)
            task.build_observation_space()
            return task
    
__combine(preliminary_goal_tasks, robots, max_episode_steps=1000, builder_name="CustomBuilderPreliminary")


hm_goal_tasks = {'HM0': {}, 'HM1': {}, 'HM2': {}, 'HM3': {}, 'HM4': {}, 'HM5': {}, 'HMT': {}, 
                 'HMEval3': {}}

class CustomBuilderHandMade(Builder):
    # def __init__(self):
    #     super.__init__()
    #     self.tasks = {

    #     }

    def _get_task_from_string(self, class_name):
        if class_name == "HMLevel0":
            return HMLevel0
        elif class_name == "HMLevel1":
            return HMLevel1
        elif class_name == "HMLevel2":
            return HMLevel2
        elif class_name == "HMLevel3":
            return HMLevel3
        elif class_name == "HMLevel4":
            return HMLevel4
        elif class_name == "HMLevel5":
            return HMLevel5
        elif class_name == "HMLevelT":
            return HMLevelTarget
        elif class_name == "HMEvalLevel3":
            return HMEvalLevel3
        else:
            return super()._get_task()

    def _get_task(self):
            class_name = get_task_class_name(self.task_id)
            task_class = self._get_task_from_string(class_name)
            task = task_class(config=self.config)
            task.build_observation_space()
            return task
    
__combine(hm_goal_tasks, robots, max_episode_steps=1000, builder_name="CustomBuilderHandMade")

hm_goal_tasks_reward = {'HMR0': {}, 'HMR1': {}, 'HMR2': {}, 'HMR3': {}, 'HMR4': {}, 'HMR5': {}}

class CustomBuilderHandMadeReward(Builder):
    # def __init__(self):
    #     super.__init__()
    #     self.tasks = {

    #     }

    def _get_task_from_string(self, class_name):
        if class_name == "HMRLevel0":
            return HMRLevel0
        elif class_name == "HMRLevel1":
            return HMRLevel1
        elif class_name == "HMRLevel2":
            return HMRLevel2
        elif class_name == "HMRLevel3":
            return HMRLevel3
        elif class_name == "HMRLevel4":
            return HMRLevel4
        elif class_name == "HMRLevel5":
            return HMRLevel5
        else:
            return super()._get_task()

    def _get_task(self):
            class_name = get_task_class_name(self.task_id)
            task_class = self._get_task_from_string(class_name)
            task = task_class(config=self.config)
            task.build_observation_space()
            return task
    
__combine(hm_goal_tasks_reward, robots, max_episode_steps=1000, builder_name="CustomBuilderHandMadeReward")