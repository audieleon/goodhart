"""Auto-detect reward dynamics from a Gymnasium environment.

Runs random episodes to estimate reward structure, including:
- Per-step reward distribution (mean, min, max, std)
- Whether reward is state-dependent (high variance)
- Terminal vs per-step reward pattern
- Death/truncation probability
- Discovery probability for positive rewards

Builds an EnvironmentModel with value_range and state_dependent
fields populated from the rollout data.
"""

from typing import Tuple

from goodhart.models import (
    EnvironmentModel,
    RewardSource,
    RewardType,
    RespawnBehavior,
)


def detect_env(env_id: str, n_episodes: int = 20) -> Tuple[EnvironmentModel, dict]:
    """Detect reward structure by running random episodes.

    Args:
        env_id: Gymnasium environment ID (e.g., "CartPole-v1").
        n_episodes: Number of random episodes to run.

    Returns:
        (EnvironmentModel, stats_dict) tuple.
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError(
            "gymnasium is required for --detect. "
            "Install with: pip install goodhart[detect]"
        )

    try:
        env = gym.make(env_id)
    except Exception as e:
        raise RuntimeError(f"Could not create environment '{env_id}': {e}")

    spec = env.spec
    max_episode_steps = (
        spec.max_episode_steps if spec and spec.max_episode_steps
        else 1000
    )
    n_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else int(env.action_space.shape[0]) if hasattr(env.action_space, "shape") else 4
    )

    # Collect per-step reward data
    episode_lengths = []
    episode_rewards = []
    all_step_rewards = []
    terminal_rewards = []  # reward on last step of each episode
    positive_reward_episodes = 0
    early_terminations = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        last_reward = 0.0

        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            all_step_rewards.append(reward)
            last_reward = reward
            steps += 1

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        terminal_rewards.append(last_reward)

        if total_reward > 0:
            positive_reward_episodes += 1
        if terminated and steps < max_episode_steps:
            early_terminations += 1

    env.close()

    # Compute statistics
    import statistics
    mean_length = statistics.mean(episode_lengths)
    mean_reward = statistics.mean(episode_rewards)
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)
    discovery_probability = positive_reward_episodes / n_episodes
    death_probability = early_terminations / n_episodes

    # Per-step reward statistics
    step_mean = statistics.mean(all_step_rewards) if all_step_rewards else 0.0
    step_min = min(all_step_rewards) if all_step_rewards else 0.0
    step_max = max(all_step_rewards) if all_step_rewards else 0.0
    step_std = statistics.stdev(all_step_rewards) if len(all_step_rewards) > 1 else 0.0

    # Detect state-dependence: high variance relative to mean
    is_state_dependent = step_std > abs(step_mean) * 0.5 if step_mean != 0 else step_std > 0.01

    # Detect terminal reward: last-step reward differs from mid-episode
    mid_rewards = all_step_rewards[:-1] if len(all_step_rewards) > 1 else all_step_rewards
    mid_mean = statistics.mean(mid_rewards) if mid_rewards else 0.0
    terminal_mean = statistics.mean(terminal_rewards) if terminal_rewards else 0.0
    has_terminal = abs(terminal_mean - mid_mean) > abs(mid_mean) * 2 if mid_mean != 0 else abs(terminal_mean) > 0.1

    stats = {
        "env_id": env_id,
        "n_episodes": n_episodes,
        "max_episode_steps": max_episode_steps,
        "n_actions": n_actions,
        "mean_episode_length": mean_length,
        "mean_reward": mean_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "step_reward_mean": step_mean,
        "step_reward_range": (step_min, step_max),
        "step_reward_std": step_std,
        "state_dependent": is_state_dependent,
        "has_terminal_reward": has_terminal,
        "discovery_probability": discovery_probability,
        "death_probability": death_probability,
    }

    # Build EnvironmentModel from detected values
    model = EnvironmentModel(
        name=env_id,
        max_steps=max_episode_steps,
        n_actions=n_actions,
        death_probability=death_probability,
    )

    # Add per-step reward if detected
    if abs(step_mean) > 0.001 and not has_terminal:
        model.add_reward_source(RewardSource(
            name="per_step_reward",
            reward_type=RewardType.PER_STEP,
            value=step_mean,
            value_range=(step_min, step_max),
            state_dependent=is_state_dependent,
            requires_action=True,
            intentional=step_mean > 0,
        ))

    # Add terminal reward if detected
    if has_terminal and abs(terminal_mean) > 0.01:
        model.add_reward_source(RewardSource(
            name="terminal_reward",
            reward_type=RewardType.TERMINAL,
            value=terminal_mean,
            discovery_probability=discovery_probability,
        ))

    # Add step penalty if negative per-step
    if step_mean < -0.001 and not has_terminal:
        # Already covered by per_step_reward above
        pass
    elif step_min < -0.1 and has_terminal:
        # There's a penalty alongside the terminal reward
        penalty_estimate = step_mean if step_mean < 0 else step_min / max(mean_length, 1)
        model.add_reward_source(RewardSource(
            name="step_penalty",
            reward_type=RewardType.PER_STEP,
            value=penalty_estimate,
            value_range=(step_min, 0.0),
            requires_action=False,
        ))

    return model, stats
