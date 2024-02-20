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

from custom_envs.curriculum_levels.curriculum_level_0 import CurriculumLevel0
from custom_envs.curriculum_levels.curriculum_level_1 import CurriculumLevel1
from custom_envs.curriculum_levels.curriculum_level_2 import CurriculumLevel2

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


def __combine(tasks, agents, max_episode_steps):
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
                entry_point='custom_envs.__init__:CustomBuilder',
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
goal_tasks = {'Curriculum0': {}, 'Curriculum1': {}, 'Curriculum2': {}}

class CustomBuilder(Builder):
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
    
__combine(goal_tasks, robots, max_episode_steps=1000)