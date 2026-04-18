"""State-machine scripted oracle policies for PhysREPA manipulation tasks.

Uses IK RELATIVE control (command_type="pose", use_relative_mode=True).
Action space (7D): [dx, dy, dz, droll, dpitch, dyaw, gripper]
  - arm_action (6D): delta EEF pose in robot root frame
  - gripper_action (1D): negative = CLOSE, positive/zero = OPEN
"""

from __future__ import annotations

import torch


class ScriptedPolicy:
    """Base class for state-machine scripted policies."""

    P_GAIN = 3.0  # proportional gain for delta position control

    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.state = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.state_timer = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.state[env_ids] = 0
        self.state_timer[env_ids] = 0

    def get_action(self, obs: dict) -> torch.Tensor:
        raise NotImplementedError

    def _delta_action(self, ee_pos, target_pos, gripper):
        """Build (num_envs, 7) action: [dx,dy,dz, 0,0,0, gripper]."""
        action = torch.zeros(self.num_envs, 7, device=self.device)
        delta = self.P_GAIN * (target_pos - ee_pos)
        action[:, :3] = delta.clamp(-1.0, 1.0)
        if isinstance(gripper, (int, float)):
            action[:, 6] = float(gripper)
        else:
            action[:, 6] = gripper
        return action

    def _advance(self, mask):
        self.state[mask] += 1
        self.state_timer[mask] = 0

    def _tick(self):
        self.state_timer += 1


class LiftPolicy(ScriptedPolicy):
    """Lift: approach → descend → grasp → lift → hold."""

    ABOVE_Z = 0.08
    GRASP_Z = 0.0
    LIFT_Z = 0.15
    THRESH = 0.02
    DESCEND_T = 80
    GRASP_T = 30
    LIFT_T = 60

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        obj = obs["policy"]["object_position"]
        tgt = ee.clone()
        grip = torch.ones(self.num_envs, device=self.device)  # open

        for sid, (z_off, g, trans_fn) in enumerate([
            (self.ABOVE_Z, 1.0, lambda d, t: d < self.THRESH),
            (self.GRASP_Z, 1.0, lambda d, t: (d < self.THRESH) | (t > self.DESCEND_T)),
            (self.GRASP_Z, -1.0, lambda d, t: t > self.GRASP_T),
            (None, -1.0, lambda d, t: t > self.LIFT_T),
            (None, -1.0, lambda d, t: torch.zeros_like(t, dtype=torch.bool)),
        ]):
            s = self.state == sid
            if not s.any():
                continue
            if sid <= 2:
                t = obj.clone()
                t[:, 2] += z_off
                tgt[s] = t[s]
            elif sid == 3:
                t = ee.clone()
                t[:, 2] = obj[:, 2] + self.LIFT_Z
                tgt[s] = t[s]
            else:
                tgt[s] = ee[s]
            grip[s] = g
            dist = torch.norm(ee[s] - tgt[s], dim=-1)
            trans = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            trans[s] = trans_fn(dist, self.state_timer[s])
            self._advance(trans)

        self._tick()
        return self._delta_action(ee, tgt, grip)


class PickPlacePolicy(ScriptedPolicy):
    """Pick-and-Place: approach → descend → grasp → lift → move_to → descend → release → retreat."""

    ABOVE_Z = 0.08
    GRASP_Z = 0.0
    LIFT_Z = 0.08  # just enough to clear table, not too high
    PLACE_Z = 0.005  # almost touching table — minimal drop distance
    THRESH = 0.015  # tighter convergence for precision
    DESCEND_T = 80
    GRASP_T = 60  # more time to grip firmly
    LIFT_T = 60
    PLACE_T = 100  # more time for precise placement
    RELEASE_T = 20

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        obj = obs["policy"]["object_position"]
        target_raw = obs["physics_gt"]["target_position"]
        target_3d = target_raw[:, :3]

        tgt = ee.clone()
        grip = torch.ones(self.num_envs, device=self.device)  # open

        # State 0: APPROACH_ABOVE object
        s = self.state == 0
        if s.any():
            t = obj.clone(); t[:, 2] += self.ABOVE_Z; tgt[s] = t[s]; grip[s] = 1.0
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device); m[s] = d < self.THRESH
            self._advance(m)

        # State 1: DESCEND to object
        s = self.state == 1
        if s.any():
            t = obj.clone(); t[:, 2] += self.GRASP_Z; tgt[s] = t[s]; grip[s] = 1.0
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.DESCEND_T); self._advance(m)

        # State 2: GRASP
        s = self.state == 2
        if s.any():
            t = obj.clone(); t[:, 2] += self.GRASP_Z; tgt[s] = t[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.GRASP_T; self._advance(m)

        # State 3: LIFT + TRANSPORT — move to target XY at lift height (timer only, 200 steps = 4 sec)
        s = self.state == 3
        if s.any():
            t = target_3d.clone(); t[:, 2] = 0.08
            tgt[s] = t[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > 120  # 2.4 seconds to reach target (P_GAIN=5 converges faster)
            self._advance(m)

        # State 4: DESCEND above target — already above target, just hold XY and lower slowly
        s = self.state == 4
        if s.any():
            t = target_3d.clone(); t[:, 2] = 0.08  # hold above target briefly
            tgt[s] = t[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > 30
            self._advance(m)

        # State 5: DESCEND to place
        s = self.state == 5
        if s.any():
            t = target_3d.clone(); t[:, 2] = self.PLACE_Z; tgt[s] = t[s]; grip[s] = -1.0
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.PLACE_T); self._advance(m)

        # State 6: RELEASE — open gripper, hold position still
        s = self.state == 6
        if s.any():
            # Hold at current XY but keep low Z — don't move, just open gripper
            t = target_3d.clone(); t[:, 2] = self.PLACE_Z
            tgt[s] = t[s]; grip[s] = 1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.RELEASE_T; self._advance(m)

        # State 7: RETREAT — move straight UP from current position
        s = self.state == 7
        if s.any():
            t = ee.clone(); t[:, 2] = 0.15  # go straight up, don't move XY
            tgt[s] = t[s]; grip[s] = 1.0

        self._tick()
        return self._delta_action(ee, tgt, grip)


class PushPolicy(ScriptedPolicy):
    """Physics-adaptive push: adjusts push intensity based on friction.

    Low friction → gentle push (small overshoot distance, low gain)
    High friction → stronger push (larger overshoot, higher gain)
    """

    ABOVE_Z = 0.08
    PUSH_Z = 0.03
    BEHIND_DIST = 0.06
    THRESH = 0.02
    DESCEND_T = 80
    PUSH_T = 400  # 8 seconds max
    OBJ_TARGET_THRESH = 0.04

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        # Per-env adaptive push parameters (set after reset based on friction)
        self.push_overshoot = torch.full((num_envs,), 0.04, device=self.device)
        self.push_gain = torch.full((num_envs,), 3.0, device=self.device)

    def set_friction(self, surface_friction: float, object_friction: float):
        """Set adaptive push parameters based on friction values.

        Called from collection script after env.reset().
        """
        # Combined effective friction
        eff_friction = (surface_friction + object_friction) / 2.0

        # Low friction (< 0.3): very gentle push — small overshoot, low gain
        # Medium friction (0.3-0.5): moderate push
        # High friction (> 0.5): strong push — large overshoot, high gain
        self.push_overshoot[:] = 0.02 + eff_friction * 0.06  # 0.02 ~ 0.08
        self.push_gain[:] = 1.5 + eff_friction * 3.0  # 1.5 ~ 4.5

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        obj = obs["policy"]["object_position"]
        target_raw = obs["physics_gt"]["target_position"]
        target_3d = target_raw[:, :3]

        tgt = ee.clone()
        grip = torch.ones(self.num_envs, device=self.device)  # always open

        # Push direction (XY)
        push_dir = target_3d[:, :2] - obj[:, :2]
        push_norm = torch.norm(push_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        push_dir = push_dir / push_norm

        # Behind position
        behind = obj.clone()
        behind[:, :2] -= push_dir * self.BEHIND_DIST
        behind[:, 2] = self.PUSH_Z

        # State 0: APPROACH above behind position
        s = self.state == 0
        if s.any():
            t = behind.clone(); t[:, 2] = self.ABOVE_Z; tgt[s] = t[s]
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device); m[s] = d < self.THRESH
            self._advance(m)

        # State 1: DESCEND behind object
        s = self.state == 1
        if s.any():
            tgt[s] = behind[s]
            d = torch.norm(ee[s] - behind[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.DESCEND_T); self._advance(m)

        # State 2: PUSH — adaptive overshoot based on friction
        s = self.state == 2
        if s.any():
            push_target = obj.clone()
            # Overshoot distance adapts to friction
            overshoot = self.push_overshoot[s].unsqueeze(-1)  # (n, 1)
            push_target[s, :2] = obj[s, :2] + push_dir[s] * overshoot
            push_target[s, 2] = self.PUSH_Z
            tgt[s] = push_target[s]
            obj_dist = torch.norm(obj[s, :2] - target_3d[s, :2], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (obj_dist < self.OBJ_TARGET_THRESH) | (self.state_timer[s] > self.PUSH_T)
            self._advance(m)

        # State 3: DONE — hold position
        s = self.state == 3
        if s.any():
            tgt[s] = ee[s]

        self._tick()
        # Use adaptive gain for push action
        action = torch.zeros(self.num_envs, 7, device=self.device)
        delta = self.push_gain.unsqueeze(-1) * (tgt - ee)
        delta = delta.clamp(-1.0, 1.0)
        action[:, :3] = delta[:, :3]
        action[:, 6] = grip
        return action


class StackPolicy(ScriptedPolicy):
    """Stack: pick cube_a → place on cube_b."""

    ABOVE_Z = 0.08
    GRASP_Z = 0.0
    STACK_OFFSET = 0.065  # cube_a center above cube_b center
    PLACE_HEIGHT = 0.002  # almost touching cube_b before release
    THRESH = 0.015
    DESCEND_T = 80
    GRASP_T = 60
    TRANSPORT_T = 150  # more time to precisely reach cube_b
    PLACE_T = 120  # more time to descend precisely
    RELEASE_T = 20

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        ca = obs["policy"]["cube_a_position"]
        cb = obs["policy"]["cube_b_position"]

        tgt = ee.clone()
        grip = torch.ones(self.num_envs, device=self.device)  # open

        # State 0: APPROACH above cube_a
        s = self.state == 0
        if s.any():
            t = ca.clone(); t[:, 2] += self.ABOVE_Z; tgt[s] = t[s]; grip[s] = 1.0
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device); m[s] = d < self.THRESH
            self._advance(m)

        # State 1: DESCEND to cube_a
        s = self.state == 1
        if s.any():
            t = ca.clone(); t[:, 2] += self.GRASP_Z; tgt[s] = t[s]; grip[s] = 1.0
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.DESCEND_T); self._advance(m)

        # State 2: GRASP
        s = self.state == 2
        if s.any():
            t = ca.clone(); t[:, 2] += self.GRASP_Z; tgt[s] = t[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.GRASP_T; self._advance(m)

        # State 3: LIFT + TRANSPORT to above cube_b (combined, timer based)
        s = self.state == 3
        if s.any():
            t = cb.clone(); t[:, 2] += self.STACK_OFFSET + self.ABOVE_Z  # cube_b XY + stack height
            tgt[s] = t[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.TRANSPORT_T
            self._advance(m)

        # State 4: FINE ALIGN above cube_b (hold position)
        s = self.state == 4
        if s.any():
            t = cb.clone(); t[:, 2] += self.STACK_OFFSET + self.ABOVE_Z
            tgt[s] = t[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > 30  # hold for alignment
            self._advance(m)

        # State 5: DESCEND onto cube_b
        s = self.state == 5
        if s.any():
            t = cb.clone(); t[:, 2] += self.STACK_OFFSET + self.PLACE_HEIGHT
            tgt[s] = t[s]; grip[s] = -1.0
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.PLACE_T); self._advance(m)

        # State 6: RELEASE — hold position, open gripper
        s = self.state == 6
        if s.any():
            t = cb.clone(); t[:, 2] += self.STACK_OFFSET + self.PLACE_HEIGHT
            tgt[s] = t[s]; grip[s] = 1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.RELEASE_T; self._advance(m)

        # State 7: RETREAT — straight up
        s = self.state == 7
        if s.any():
            t = ee.clone(); t[:, 2] = 0.15; tgt[s] = t[s]; grip[s] = 1.0

        self._tick()
        return self._delta_action(ee, tgt, grip)
