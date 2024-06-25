# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Environments in the Safety-Gymnasium."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import torch
import re

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box

@env_register
class HMCurriculumEnv(CMDP):
    """Curriculum Environment.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Keyword Args:
        render_mode (str, optional): The render mode ranges from 'human' to 'rgb_array' and 'rgb_array_list'.
            Defaults to 'rgb_array'.
        camera_name (str, optional): The camera name.
        camera_id (int, optional): The camera id.
        width (int, optional): The width of the rendered image. Defaults to 256.
        height (int, optional): The height of the rendered image. Defaults to 256.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False

    _original_support_envs = [
        "SafetyPointHM0-v0",
        "SafetyCarHM0-v0",
        "SafetyDoggoHM0-v0",
        "SafetyRacecarHM0-v0",
        "SafetyAntHM0-v0",
        "SafetyPointHM1-v0",
        "SafetyCarHM1-v0",
        "SafetyDoggoHM1-v0",
        "SafetyRacecarHM1-v0",
        "SafetyAntHM1-v0",
        "SafetyPointHM2-v0",
        "SafetyCarHM2-v0",
        "SafetyDoggoHM2-v0",
        "SafetyRacecarHM2-v0",
        "SafetyAntHM2-v0",
        "SafetyPointHM3-v0",
        "SafetyCarHM3-v0",
        "SafetyDoggoHM3-v0",
        "SafetyRacecarHM3-v0",
        "SafetyAntHM3-v0",
        "SafetyPointHM4-v0",
        "SafetyCarHM4-v0",
        "SafetyDoggoHM4-v0",
        "SafetyRacecarHM4-v0",
        "SafetyAntHM4-v0",
        "SafetyPointHM5-v0",
        "SafetyCarHM5-v0",
        "SafetyDoggoHM5-v0",
        "SafetyRacecarHM5-v0",
        "SafetyAntHM5-v0",
        "SafetyPointHMT-v0",
        "SafetyCarHMT-v0",
        "SafetyDoggoHMT-v0",
        "SafetyRacecarHMT-v0",
        "SafetyAntHMT-v0",
        
        # "SafetyPointHMEval3-v0",
        # "SafetyCarHMEval3-v0",
        # "SafetyDoggoHMEval3-v0",
        # "SafetyRacecarHMEval3-v0",
        # "SafetyAntHMEval3-v0",
    ]

    # Create dummy environments, which are not actual environments but contain information on what to do
    # _base_support_envs = [env.replace("HM", "BaseHM") for env in _original_support_envs]
    _reward_support_envs = [env.replace("HM", "HMR") for env in _original_support_envs]
    _original_support_envs = _original_support_envs + _reward_support_envs
    _curr_starting_support_envs = [env.replace("HM", f"From{i}HM") for env in _original_support_envs for i in [0, 1, 2, 3, 4, 5, "T"]]

    _support_envs: ClassVar[list[str]] = _original_support_envs + _curr_starting_support_envs

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of :class:`SafetyGymnasiumEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        self._kwargs = kwargs
        self._steps = 0
        self._curriculum = False
        # self.disable_progress = True

        self._task_to_int = {"0": 0,
                             "1": 1,
                             "2": 2,
                             "3": 3,
                             "4": 4,
                             "5": 5,
                             "T": 6,
                            }

        end_version_pattern = r'HMR?(\d+|T)'
        end_version = re.search(end_version_pattern, env_id)
        self._end_task = end_version.group(1)
        self._start_task = self._end_task

        # if "HM0" in env_id:
        #     self._curriculum = True
        # if "Base" in env_id:
        #     self._curriculum = False
        #     env_id = env_id.replace("Base", "")
        if "From" in env_id:
            self._curriculum = True
            start_version_pattern = r'From(\d+|T)'
            start_version = re.search(start_version_pattern, env_id)
            self._start_task = start_version.group(1)
            if self._task_to_int[self._start_task] > self._task_to_int[self._end_task]:
                raise Exception(f"Start task cannot be higher than end task in CMDP {env_id}")
            env_id = env_id.replace(f"From{self._start_task}", "")

        self._curr_changes = {"0": 0,
                              "1": 10 * 1000,
                              "2": 20 * 1000,
                              "3": 40 * 1000,
                              "4": 100 * 1000,
                              "5": 300 * 1000,
                              "T": 700 * 1000,
                              }

        if num_envs > 1:
            self._env = safety_gymnasium.vector.make(env_id=env_id, num_envs=num_envs, **kwargs)
            assert isinstance(self._env.single_action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.single_observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
        else:
            self.need_time_limit_wrapper = True
            self.need_auto_reset_wrapper = True

            # For the curriculum, create all future environments
            if self._curriculum:
                env_base = env_id.split("HM")[0]
                version_base = end_version.group()[:-1]
                self._env_0 = safety_gymnasium.make(id=env_base+version_base+"0-v0", autoreset=True, **self._kwargs)
                self._env_1 = safety_gymnasium.make(id=env_base+version_base+"1-v0", autoreset=True, **self._kwargs)
                self._env_2 = safety_gymnasium.make(id=env_base+version_base+"2-v0", autoreset=True, **self._kwargs)
                self._env_3 = safety_gymnasium.make(id=env_base+version_base+"3-v0", autoreset=True, **self._kwargs)
                self._env_4 = safety_gymnasium.make(id=env_base+version_base+"4-v0", autoreset=True, **self._kwargs)
                self._env_5 = safety_gymnasium.make(id=env_base+version_base+"5-v0", autoreset=True, **self._kwargs)
                self._env_T = safety_gymnasium.make(id=env_base+"HMT-v0", autoreset=True, **self._kwargs)
                self._env = eval(f"self._env_{self._start_task}")
                self._steps = self._curr_changes.get(self._start_task)
            else:
                self._env = safety_gymnasium.make(id=env_id, autoreset=True, **kwargs)

            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space

        self._metadata = self._env.metadata

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step the environment.

        .. note::
            OmniSafe uses auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode. And the
            true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key
            of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        
        self._steps += 1

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.


        Returns:
            observation: Agent's observation of the current environment.
            info: Some information logged by the environment.
        """

        if self._curriculum:
            if options != None and options.get("resetting_for_eval"):
                self._env = eval(f"self._env_{self._end_task}")
            elif self._steps == self._curr_changes.get("1") and (self._end_task == "T" or int(self._end_task) >= 1):
                print("Changed env to level 1")
                self._env = self._env_1
            elif self._steps == self._curr_changes.get("2") and (self._end_task == "T" or int(self._end_task) >= 2):
                print("Changed env to level 2")
                self._env = self._env_2
            elif self._steps == self._curr_changes.get("3") and (self._end_task == "T" or int(self._end_task) >= 3):
                print("Changed env to level 3")
                self._env = self._env_3
            elif self._steps == self._curr_changes.get("4") and (self._end_task == "T" or int(self._end_task) >= 4):
                print("Changed env to level 4")
                self._env = self._env_4
            elif self._steps == self._curr_changes.get("5") and (self._end_task == "T" or int(self._end_task) >= 5):
                print("Changed env to level 5")
                self._env = self._env_5
            elif self._steps == self._curr_changes.get("T") and self._end_task == "T":
                print("Changed env to level Target")
                self._env = self._env_T

        obs, info = self._env.reset(seed=seed, options=options)
        # self._env.task.agent.locations = [(-1.5, 0)]

        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def sample_action(self) -> torch.Tensor:
        """Sample a random action.

        Returns:
            A random action.
        """
        return torch.as_tensor(
            self._env.action_space.sample(),
            dtype=torch.float32,
            device=self._device,
        )

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()