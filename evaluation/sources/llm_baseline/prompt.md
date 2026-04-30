# LLM Baseline Prompt

Model: Claude Sonnet 4.6
Date: 2026-04-27
Temperature: default
Single run, no prompt tuning.

## Prompt

You are an RL reward design expert. For each of the 23 reward configurations below, identify structural traps — cases where the reward structure mathematically guarantees or strongly incentivizes degenerate behavior (standing still, dying early, looping, ignoring the goal, etc.).

For each configuration, output EXACTLY one of:
- FAIL: structural trap that will cause degenerate behavior
- WARN: likely issue depending on dynamics
- PASS: no structural issues detected

Then list the specific traps you found (if any). Be conservative — only flag FAIL for mathematically clear traps.

Here are the 23 configurations:

1. anymal: max_steps=1000, gamma=0.99, death_prob=0.02
   alive_bonus: +0.2/step (passive), velocity_tracking: +1.0/step (active), torque_penalty: -0.002/step, joint_vel_penalty: -0.001/step

2. atari: max_steps=18000, gamma=0.99, death_prob=0.05
   score: +1.0 (on event, active)

3. bipedal-walker: max_steps=1600, gamma=0.99, death_prob=0.05
   forward_progress: +0.3/step (active), fall_penalty: -100.0 (on event), torque_penalty: -0.003/step

4. cartpole: max_steps=500, gamma=0.99, death_prob=0.05
   alive_reward: +1.0/step (passive, THIS IS THE GOAL - survival task)

5. coast-runners: max_steps=2000, gamma=0.99, death_prob=0.01
   finish_race: +100.0 (terminal, active), turbo_powerup: +20.0 (on event, active, respawns infinitely, can loop)

6. coinrun: max_steps=1000, gamma=0.999, death_prob=0.02
   coin: +10.0 (terminal, active)

7. dense-survival: max_steps=2000, gamma=0.99, death_prob=0.1
   alive_bonus: +1.0/step (passive, IS THE GOAL), food: +5.0 (on event, active)

8. football: max_steps=3000, gamma=0.997, death_prob=0.0
   goal_scored: +1.0 (terminal, active), checkpoint_shaping: +0.1 (on event, active)

9. hand-manipulation: max_steps=100, gamma=0.998, death_prob=0.0
   rotation_achieved: +1.0 (terminal, active), rotation_distance_shaping: +0.5 (shaping, passive), fingertip_contact_bonus: +0.1/step (passive)

10. highway-env: max_steps=40, gamma=0.8, death_prob=0.05
    collision: -1.0 (on event), high_speed: +0.4/step (active), right_lane: +0.1/step (active)

11. humanoid: max_steps=1000, gamma=0.99, death_prob=0.03
    healthy_reward: +5.0/step (passive, NOT the goal), forward_velocity: +1.25/step (active, IS the goal), ctrl_cost: -0.1/step

12. legged-gym: max_steps=1000, gamma=0.99, death_prob=0.02
    tracking_lin_vel: +1.0/step (active), tracking_ang_vel: +0.5/step (active), feet_air_time: +1.0/step (passive), lin_vel_z_penalty: -2.0/step, ang_vel_xy_penalty: -0.05/step, torques_penalty: -0.00002/step, dof_acc_penalty: -0.0000025/step, action_rate_penalty: -0.01/step, collision_penalty: -1.0 (on event)

13. lunar-lander: max_steps=1000, gamma=0.99, death_prob=0.1
    distance_shaping: +1.0 (shaping, active), landing_bonus: +100.0 (terminal, active), crash_penalty: -100.0 (on event), fuel_cost: -0.03/step

14. metadrive: max_steps=1000, gamma=0.99, death_prob=0.01
    driving_progress: +1.0/step (active), lateral_factor: +1.0/step (active), speed_reward: +0.1/step (active), success: +10.0 (terminal, active), crash_penalty: -5.0 (on event)

15. minihack-navigation: max_steps=500, gamma=0.999, death_prob=0.05
    goal: +1.0 (terminal, active), step_penalty: -0.001/step

16. minihack-skill: max_steps=1000, gamma=0.999, death_prob=0.1
    task_completion: +1.0 (terminal, active), step_penalty: -0.001/step

17. mountain-car: max_steps=200, gamma=1.0, death_prob=0.0
    reach_flag: +1.0 (terminal, active), step_penalty: -1.0/step

18. mujoco-locomotion: max_steps=1000, gamma=0.99, death_prob=0.02
    forward_velocity: +1.0/step (active), alive_bonus: +1.0/step (passive, IS the goal alongside velocity), control_penalty: -0.001/step

19. mujoco-manipulation: max_steps=200, gamma=0.98, death_prob=0.0
    goal_reached: +1.0 (terminal, active), distance_shaping: +0.1 (shaping, active)

20. robosuite-pick-place: max_steps=200, gamma=0.99, death_prob=0.0
    grasp: +0.35 (on event, active), lift: +0.15 (on event, active), hover: +0.2 (on event, active), place: +1.0 (terminal, active)

21. smac: max_steps=120, gamma=0.99, death_prob=0.1
    enemy_killed: +10.0 (on event, active), ally_killed: -5.0 (on event, passive), damage_dealt: +0.5/step (active), damage_received: -0.25/step (passive), win_bonus: +200.0 (terminal, active)

22. sparse-goal: max_steps=500, gamma=0.99, death_prob=0.0
    goal: +1.0 (terminal, active)

23. taxi: max_steps=200, gamma=0.99, death_prob=0.0
    step_penalty: -1.0/step, dropoff_success: +20.0 (terminal, active), illegal_action: -10.0 (on event)
