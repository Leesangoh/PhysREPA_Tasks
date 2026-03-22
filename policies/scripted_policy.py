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

        # Increased overshoot — was undershooting on average 20cm
        # Low friction → object slides far → push target closer to object (small overshoot)
        # High friction → object stops quick → push past target (large overshoot)
        self.push_overshoot[:] = 0.04 + eff_friction * 0.12  # 0.04 ~ 0.16
        self.push_gain[:] = 2.5 + eff_friction * 3.5  # 2.5 ~ 6.0

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


class ReachPolicy(ScriptedPolicy):
    """Reach: move EE to target position. Trivial — no contact, no object.

    Action is 6D (no gripper) for reach env.
    """

    P_GAIN = 3.0
    THRESH = 0.02  # 2cm success threshold

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        # target_position from ee_pose command: [x, y, z, qw, qx, qy, qz]
        target_raw = obs["physics_gt"]["target_position"]
        target_3d = target_raw[:, :3]

        # 6D action: [dx, dy, dz, droll, dpitch, dyaw]
        action = torch.zeros(self.num_envs, 6, device=self.device)

        # State 0: MOVE to target
        s = self.state == 0
        if s.any():
            delta = self.P_GAIN * (target_3d[s] - ee[s])
            action[s, :3] = delta.clamp(-1.0, 1.0)
            d = torch.norm(ee[s] - target_3d[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = d < self.THRESH
            self._advance(m)

        # State 1: HOLD at target
        s = self.state == 1
        if s.any():
            delta = self.P_GAIN * (target_3d[s] - ee[s])
            action[s, :3] = delta.clamp(-1.0, 1.0)

        self._tick()
        return action


class StrikePolicy(ScriptedPolicy):
    """Strike: position behind ball, swing through to hit, retract, wait for ball to settle.

    Gripper always closed (fist).
    """

    WINDUP_DIST = 0.15  # increased distance behind ball
    SWING_Z = 0.04  # ball center height (ball radius = 0.04)
    ABOVE_Z = 0.10
    THRESH = 0.02
    POSITION_T = 100
    SWING_GAIN = 12.0  # higher gain for faster swing
    SWING_T = 30  # slightly more time for swing
    RETRACT_T = 30

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.swing_speed = torch.full((num_envs,), 0.5, device=self.device)

    def set_friction(self, surface_friction: float, object_friction: float):
        """Adjust swing gain based on friction (higher friction = harder swing needed)."""
        eff_friction = (surface_friction + object_friction) / 2.0
        # Adjust swing gain: low friction = moderate swing, high friction = very hard swing
        self.swing_speed[:] = 10.0 + eff_friction * 8.0  # 10 ~ 18

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        ball = obs["policy"]["object_position"]
        target_raw = obs["physics_gt"]["target_position"]
        target_3d = target_raw[:, :3]

        tgt = ee.clone()
        grip = -torch.ones(self.num_envs, device=self.device)  # always closed

        # Push direction (ball → target, XY only)
        push_dir = target_3d[:, :2] - ball[:, :2]
        push_norm = torch.norm(push_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        push_dir = push_dir / push_norm

        # Windup position: behind ball, opposite push direction
        windup = ball.clone()
        windup[:, :2] -= push_dir * self.WINDUP_DIST
        windup[:, 2] = self.SWING_Z

        # State 0: POSITION above windup point
        s = self.state == 0
        if s.any():
            t = windup.clone(); t[:, 2] = self.ABOVE_Z; tgt[s] = t[s]
            d = torch.norm(ee[s, :2] - t[s, :2], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = d < self.THRESH
            self._advance(m)

        # State 1: DESCEND to swing height behind ball
        s = self.state == 1
        if s.any():
            tgt[s] = windup[s]
            d = torch.norm(ee[s] - windup[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.POSITION_T)
            self._advance(m)

        # State 2: SWING — fast move through ball toward target
        s = self.state == 2
        if s.any():
            swing_target = ball.clone()
            swing_target[:, :2] += push_dir * 0.15  # 15cm past ball for more velocity
            swing_target[:, 2] = self.SWING_Z
            tgt[s] = swing_target[s]
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            passed_ball = (ee[s, 0] - ball[s, 0]) * push_dir[s, 0] + \
                          (ee[s, 1] - ball[s, 1]) * push_dir[s, 1]
            m[s] = (passed_ball > 0.02) | (self.state_timer[s] > self.SWING_T)
            self._advance(m)

        # State 3: RETRACT — lift up quickly
        s = self.state == 3
        if s.any():
            t = ee.clone(); t[:, 2] = self.ABOVE_Z + 0.05; tgt[s] = t[s]
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.RETRACT_T
            self._advance(m)

        # State 4: WAIT — hold position while ball settles
        s = self.state == 4
        if s.any():
            tgt[s] = ee[s]

        self._tick()

        # Use adaptive high gain for SWING phase, default for others
        action = torch.zeros(self.num_envs, 7, device=self.device)
        gain = torch.full((self.num_envs, 1), self.P_GAIN, device=self.device)
        swing_mask = self.state == 2
        gain[swing_mask] = self.swing_speed[swing_mask].unsqueeze(-1)
        delta = gain * (tgt - ee)
        action[:, :3] = delta.clamp(-1.0, 1.0)
        action[:, 6] = grip
        return action


class DrawerPolicy(ScriptedPolicy):
    """Drawer: approach handle → grasp → pull open → release.

    Saves handle position at first step to avoid feedback loop.
    """

    P_GAIN = 3.0
    THRESH = 0.03
    APPROACH_T = 150
    ALIGN_T = 80
    GRASP_T = 60  # longer grasp for firm grip
    PULL_T = 250
    PULL_GAIN = 3.0  # stronger pull

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.handle_pos_saved = torch.zeros(num_envs, 3, device=self.device)
        self.handle_saved = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.handle_saved[env_ids] = False

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        rel_dist = obs["policy"]["rel_ee_drawer_distance"]

        # Save handle position once at first step (avoid feedback loop during pull)
        needs_save = ~self.handle_saved
        if needs_save.any():
            handle_pos = ee + rel_dist
            self.handle_pos_saved[needs_save] = handle_pos[needs_save]
            self.handle_saved[needs_save] = True

        handle = self.handle_pos_saved.clone()

        tgt = ee.clone()
        grip = torch.ones(self.num_envs, device=self.device)  # open

        # State 0: APPROACH — move to pre-grasp (5cm BEYOND handle in +x, from cabinet side)
        s = self.state == 0
        if s.any():
            t = handle.clone()
            t[:, 0] += 0.05  # 5cm beyond handle (toward cabinet)
            tgt[s] = t[s]; grip[s] = 1.0
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.APPROACH_T)
            self._advance(m)

        # State 1: REACH_HANDLE — move to handle position
        s = self.state == 1
        if s.any():
            tgt[s] = handle[s]; grip[s] = 1.0
            d = torch.norm(ee[s] - handle[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.ALIGN_T)
            self._advance(m)

        # State 2: CLOSE_GRIPPER
        s = self.state == 2
        if s.any():
            tgt[s] = handle[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.GRASP_T
            self._advance(m)

        # State 3: PULL — fixed target: handle x - 0.15, same y/z
        s = self.state == 3
        if s.any():
            t = handle.clone()
            t[:, 0] -= 0.15  # pull 15cm toward robot
            tgt[s] = t[s]; grip[s] = -1.0
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.PULL_T
            self._advance(m)

        # State 4: DONE
        s = self.state == 4
        if s.any():
            tgt[s] = ee[s]; grip[s] = 1.0

        self._tick()

        action = torch.zeros(self.num_envs, 7, device=self.device)
        gain = torch.full((self.num_envs, 1), self.P_GAIN, device=self.device)
        gain[self.state == 3] = self.PULL_GAIN
        delta = gain * (tgt - ee)
        action[:, :3] = delta.clamp(-1.0, 1.0)
        action[:, 6] = grip
        return action


class PegInsertPolicy(ScriptedPolicy):
    """PegInsert: approach above hole → align XY → insert (descend) → hold.

    Pre-grasped peg, gripper always closed.
    """

    ABOVE_Z_OFFSET = 0.06  # 6cm above hole
    ALIGN_THRESH = 0.005  # 5mm XY alignment precision
    THRESH = 0.02
    APPROACH_T = 100
    ALIGN_T = 100
    INSERT_GAIN = 1.5  # gentle insertion
    INSERT_T = 300  # 6 seconds for insertion

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        hole_pos = obs["policy"]["hole_position"]  # in robot base frame

        tgt = ee.clone()
        grip = -torch.ones(self.num_envs, device=self.device)  # always closed

        # State 0: APPROACH — move above hole
        s = self.state == 0
        if s.any():
            t = hole_pos.clone()
            t[:, 2] += self.ABOVE_Z_OFFSET
            tgt[s] = t[s]
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.APPROACH_T)
            self._advance(m)

        # State 1: ALIGN — fine-tune XY directly above hole
        s = self.state == 1
        if s.any():
            t = hole_pos.clone()
            t[:, 2] = ee[:, 2]  # keep current Z, only move XY
            tgt[s] = t[s]
            xy_dist = torch.norm(ee[s, :2] - hole_pos[s, :2], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (xy_dist < self.ALIGN_THRESH) | (self.state_timer[s] > self.ALIGN_T)
            self._advance(m)

        # State 2: INSERT — slowly descend while maintaining XY alignment
        s = self.state == 2
        if s.any():
            t = hole_pos.clone()
            t[:, 2] -= 0.01  # target slightly below hole surface
            tgt[s] = t[s]
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.INSERT_T
            self._advance(m)

        # State 3: HOLD
        s = self.state == 3
        if s.any():
            tgt[s] = ee[s]

        self._tick()

        # Use gentle gain for INSERT phase
        action = torch.zeros(self.num_envs, 7, device=self.device)
        gain = torch.full((self.num_envs, 1), self.P_GAIN, device=self.device)
        gain[self.state == 2] = self.INSERT_GAIN
        delta = gain * (tgt - ee)
        action[:, :3] = delta.clamp(-1.0, 1.0)
        action[:, 6] = grip
        return action


class NutThreadPolicy(ScriptedPolicy):
    """NutThread: approach above bolt → descend → engage → rotate → hold.

    Pre-grasped nut, gripper always closed.
    Rotation via dyaw action component.
    """

    ABOVE_Z_OFFSET = 0.04  # 4cm above bolt
    THRESH = 0.02
    APPROACH_T = 100
    DESCEND_T = 80
    ENGAGE_T = 50
    ROTATE_T = 400  # 8 seconds for rotation
    ROTATE_GAIN = 1.0  # gentle rotation
    ROTATE_SPEED = 0.3  # radians per step for yaw

    def get_action(self, obs):
        ee = obs["physics_gt"]["ee_position_b"]
        bolt_pos = obs["policy"]["bolt_position"]  # in robot base frame

        tgt = ee.clone()
        grip = -torch.ones(self.num_envs, device=self.device)  # always closed

        # State 0: APPROACH — move above bolt
        s = self.state == 0
        if s.any():
            t = bolt_pos.clone()
            t[:, 2] += self.ABOVE_Z_OFFSET
            tgt[s] = t[s]
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.APPROACH_T)
            self._advance(m)

        # State 1: DESCEND — lower onto bolt, align XY
        s = self.state == 1
        if s.any():
            t = bolt_pos.clone()
            t[:, 2] += 0.015  # just above bolt top
            tgt[s] = t[s]
            d = torch.norm(ee[s] - t[s], dim=-1)
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = (d < self.THRESH) | (self.state_timer[s] > self.DESCEND_T)
            self._advance(m)

        # State 2: ENGAGE — push down gently to initiate thread contact
        s = self.state == 2
        if s.any():
            t = bolt_pos.clone()
            t[:, 2] += 0.005  # push onto bolt
            tgt[s] = t[s]
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.ENGAGE_T
            self._advance(m)

        # State 3: ROTATE — apply rotation (dyaw) while maintaining downward pressure
        s = self.state == 3
        if s.any():
            t = bolt_pos.clone()
            t[:, 2] -= 0.005  # slight downward pressure
            tgt[s] = t[s]
            m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m[s] = self.state_timer[s] > self.ROTATE_T
            self._advance(m)

        # State 4: HOLD
        s = self.state == 4
        if s.any():
            tgt[s] = ee[s]

        self._tick()

        # Build action with rotation for ROTATE phase
        action = torch.zeros(self.num_envs, 7, device=self.device)
        gain = torch.full((self.num_envs, 1), self.P_GAIN, device=self.device)
        gain[self.state >= 2] = self.ROTATE_GAIN
        delta = gain * (tgt - ee)
        action[:, :3] = delta.clamp(-1.0, 1.0)
        # Add yaw rotation during ROTATE phase
        rotate_mask = self.state == 3
        action[rotate_mask, 5] = -self.ROTATE_SPEED  # negative yaw for clockwise
        action[:, 6] = grip
        return action
