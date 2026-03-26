# Copyright 2025-2026 Dimensional Inc.
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

import os
import traceback
from typing import Any

from dimos_lcm.std_msgs import Bool, String
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path
from dimos.navigation.base import NavigationInterface, NavigationState
from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Task names the nav module looks for in the coordinator (controller-agnostic)
_PATH_FOLLOWER_TASK_NAMES = ("path_follower", "reactive_path_follower")


class ReplanningAStarPlanner(Module, NavigationInterface):
    odom: In[PoseStamped]  # TODO: Use TF.
    global_costmap: In[OccupancyGrid]
    goal_request: In[PoseStamped]
    clicked_point: In[PointStamped]
    target: In[PoseStamped]

    goal_reached: Out[Bool]
    navigation_state: Out[String]  # TODO: set it
    cmd_vel: Out[Twist]
    path: Out[Path]
    navigation_costmap: Out[OccupancyGrid]

    _planner: GlobalPlanner

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._planner = GlobalPlanner(self.config.g)

        self._has_path_follower = False
        self._local_planner_cmd_vel_disposable: Disposable | None = None

    @rpc
    def start(self) -> None:
        super().start()

        self._disposables.add(Disposable(self.odom.subscribe(self._planner.handle_odom)))
        self._disposables.add(
            Disposable(self.global_costmap.subscribe(self._planner.handle_global_costmap))
        )
        self._disposables.add(
            Disposable(self.goal_request.subscribe(self._planner.handle_goal_request))
        )
        self._disposables.add(Disposable(self.target.subscribe(self._planner.handle_goal_request)))

        self._disposables.add(
            Disposable(
                self.clicked_point.subscribe(
                    lambda pt: self._planner.handle_goal_request(pt.to_pose_stamped())
                )
            )
        )

        self._disposables.add(self._planner.path.subscribe(self.path.publish))

        if not self._has_path_follower:
            self._local_planner_cmd_vel_disposable = self._planner.cmd_vel.subscribe(
                self.cmd_vel.publish
            )
            self._disposables.add(self._local_planner_cmd_vel_disposable)
            logger.warning("No path follower task detected — using LocalPlanner fallback")
        else:
            logger.info("Path follower task active — LocalPlanner cmd_vel disabled")

        self._disposables.add(self._planner.goal_reached.subscribe(self.goal_reached.publish))

        if "DEBUG_NAVIGATION" in os.environ:
            self._disposables.add(
                self._planner.navigation_costmap.subscribe(self.navigation_costmap.publish)
            )

        self._planner.start()

    @rpc
    def stop(self) -> None:
        self.cancel_goal()
        self._planner.stop()

        super().stop()

    @rpc
    def set_goal(self, goal: PoseStamped) -> bool:
        self._planner.handle_goal_request(goal)
        return True

    @rpc
    def get_state(self) -> NavigationState:
        return self._planner.get_state()

    @rpc
    def is_goal_reached(self) -> bool:
        return self._planner.is_goal_reached()

    @rpc
    def cancel_goal(self) -> bool:
        self._planner.cancel_goal()
        return True

    @rpc
    def set_replanning_enabled(self, enabled: bool) -> None:
        self._planner.set_replanning_enabled(enabled)

    @rpc
    def set_safe_goal_clearance(self, clearance: float) -> None:
        self._planner.set_safe_goal_clearance(clearance)

    @rpc
    def reset_safe_goal_clearance(self) -> None:
        self._planner.reset_safe_goal_clearance()

    @rpc
    def on_system_modules(self, modules: list) -> None:
        """Auto-detect ControlCoordinator and its path follower task.

        The nav module is **controller-agnostic** — it does not create or
        configure path follower tasks.  The blueprint declares the task via
        TaskConfig on the coordinator; this method simply discovers it by
        name and wires the GlobalPlanner to route paths through it.
        """
        from dimos.control.coordinator import ControlCoordinator as CoordinatorClass
        from dimos.core.rpc_client import RPCClient

        has_coordinator = False
        for module in modules:
            try:
                if issubclass(module.actor_class, CoordinatorClass):
                    has_coordinator = True
                    break
            except (AttributeError, TypeError):
                continue

        if not has_coordinator:
            logger.info("No ControlCoordinator in blueprint — LocalPlanner fallback active")
            return

        try:
            coordinator = RPCClient(None, CoordinatorClass)

            # Discover a path follower task already registered in the coordinator
            task_list = coordinator.list_tasks() if hasattr(coordinator, "list_tasks") else []
            task_name: str | None = None
            for name in _PATH_FOLLOWER_TASK_NAMES:
                if name in task_list:
                    task_name = name
                    break

            if task_name is None:
                logger.info(
                    "ControlCoordinator found but no path follower task registered — "
                    "LocalPlanner fallback active"
                )
                return

            logger.info(f"Found path follower task '{task_name}' in coordinator")

            self._planner.set_path_follower_task(coordinator, task_name)
            self._has_path_follower = True

            if self._local_planner_cmd_vel_disposable is not None:
                self._local_planner_cmd_vel_disposable.dispose()
                self._local_planner_cmd_vel_disposable = None
                logger.info("Disabled LocalPlanner cmd_vel — coordinator path follower is active")

        except Exception as e:
            logger.error(f"Failed to wire coordinator path follower: {e}")
            logger.error(traceback.format_exc())
