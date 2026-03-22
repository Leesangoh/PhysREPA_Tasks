# PhysREPA Task Pipeline Status

## Last Updated: 2026-03-19 ~21:00 UTC

## Environment Status

### All 4 environments created and tested:

| Task | Env Config | Policy | Camera | Contact | Physics Rand | Grasp Works |
|------|-----------|--------|--------|---------|-------------|-------------|
| Lift | ✅ | ✅ LiftPolicy | ✅ 256x256 | ✅ | ✅ mass+friction | ✅ VERIFIED |
| Pick-Place | ✅ | ✅ PickPlacePolicy | ✅ | ✅ | ✅ | Testing... |
| Push | ✅ | ✅ PushPolicy | ✅ | ✅ | ✅ | Testing... |
| Stack | ✅ | ✅ StackPolicy | ✅ | ✅ | ✅ | Testing... |

### Key Design Decisions
- **IK Relative control** (not absolute) — Franka can't reach table level with IK absolute
- **Gripper convention**: negative = close, positive = open (Isaac Lab's BinaryJointPositionAction)
- **Action space**: 7D = [dx, dy, dz, droll, dpitch, dyaw, gripper]
- **Cube scale**: 1.0 (60mm side, original DexCube size)
- **All observations in robot root frame** for policy, world frame for physics_gt

### Lift Task Verified Results
- Object successfully grasped and lifted from z=0.055 to z=0.67
- Contact detected at step 42
- Finger position ~0.027 (gripping cube of ~0.06m width)
- Full state machine: APPROACH → DESCEND → GRASP → LIFT → HOLD

### Running: Full test of all 4 tasks
- Command: `PYTHONPATH=/home/solee:$PYTHONPATH ./isaaclab.sh -p test_all_envs.py --task all --num_envs 2`
- Output: `/tmp/test_all_output.log`
- Expected completion: ~60-80 minutes from start (~20:37 UTC)

## File Structure
```
/home/solee/physrepa_tasks/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── lift_env_cfg.py        # Task 1: Lift
│   ├── pick_place_env_cfg.py  # Task 2: Pick-and-Place
│   ├── push_env_cfg.py        # Task 3: Push
│   └── stack_env_cfg.py       # Task 4: 2-Cube Stack
├── mdp/
│   ├── __init__.py
│   └── observations.py        # Custom physics GT observations
├── policies/
│   ├── __init__.py
│   └── scripted_policy.py     # Oracle policies for all 4 tasks
├── test_all_envs.py           # Environment test script
├── test_lift_env.py           # Lift-only test
└── STATUS.md                  # This file
```

## Next Steps (after test completes)
1. Tune policy parameters for each task (success rate >80%)
2. Data collection pipeline (HDF5/LeRobot format)
3. Pilot collection: 100 episodes per task
4. Sanity check: verify data quality
