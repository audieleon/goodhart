"""Educational content for each rule.

Maps rule_name → dict with:
  - learn_more: deep explanation (shown with --verbose)
  - examples: list of example names that demonstrate this rule
  - papers: key references
  - see_also: related rules

This is separated from rule implementations to keep rule code focused
on analysis logic and make educational content easy to maintain.
"""

EXPLANATIONS = {
    "penalty_dominates_goal": {
        "learn_more": (
            "The agent sees: every step costs penalty, reaching the goal pays reward. "
            "If the optimal path takes more steps than (goal / penalty), the agent LOSES "
            "reward by succeeding. The rational response is to minimize steps — which "
            "often means doing nothing or dying early.\n"
            "Classic example: Mountain Car with -1/step and +1 goal over 200 steps. "
            "Every trajectory scores between -200 (timeout) and -199 (instant goal). "
            "All are negative, so the agent has no incentive to learn.\n"
            "Fix: either reduce the penalty so the goal is worth reaching, increase "
            "the goal reward, or add potential-based shaping (Ng 1999) to create "
            "gradient signal without changing the optimal policy."
        ),
        "examples": ["cartpole_suicide", "taxi_penalty"],
        "papers": ["Moore 1990 (Mountain Car)", "Ng et al. 1999 (reward shaping)"],
        "see_also": ["death_beats_survival", "exploration_threshold"],
        "proof": "LEAN: penalty_breakeven_discounted (verified)",
    },
    "death_beats_survival": {
        "learn_more": (
            "The agent discovers that dying is cheaper than living. With a negative "
            "per-step reward and no compensating positive reward for staying alive, "
            "every additional step of survival makes the total return worse. The "
            "optimal policy becomes: die as fast as possible.\n"
            "This is NOT the same as 'reward desert' (where all strategies are "
            "equal). Here, the agent actively learns a WORSE behavior than random.\n"
            "Classic example: CartPole with -1/step penalty and no +1 alive bonus "
            "learns to drop the pole immediately (Sutton & Barto 2018).\n"
            "Fix: add an alive bonus that exceeds the step penalty, or remove the "
            "step penalty entirely. If you need time pressure, use a terminal "
            "time bonus (reward = 1 - steps/max_steps) instead of per-step cost."
        ),
        "examples": ["cartpole_variants", "cartpole_suicide"],
        "papers": ["Sutton & Barto 2018 (Ch. 3.4)", "Barto et al. 1983"],
        "see_also": ["penalty_dominates_goal"],
        "proof": "LEAN: death_beats_survival_discounted (verified)",
    },
    "idle_exploit": {
        "learn_more": (
            "The agent compares two strategies: (1) do nothing and collect passive "
            "rewards, (2) explore and pay action costs. If standing still earns more, "
            "the agent converges to the no-op policy.\n"
            "This is the most common reward design failure in locomotion tasks. "
            "MuJoCo Humanoid-v4 with healthy_reward=5.0 earns 5000/episode by "
            "standing still vs ~5500 for walking — not worth the fall risk.\n"
            "Three fixes: (1) Remove the passive reward entirely. "
            "(2) Make the passive reward smaller than the active reward. "
            "(3) Add an idle penalty that makes standing still costly.\n"
            "The explore_fraction field controls how much credit random exploration "
            "gets from intentional rewards. If your locomotion reward gives partial "
            "credit for random stumbling, set explore_fraction > 0."
        ),
        "examples": [
            "humanoid_idle",
            "mujoco_locomotion",
            "ant_v4_gymnasium",
            "legged_gym_rewards",
            "reward_is_enough",
        ],
        "papers": ["Todorov et al. 2012 (MuJoCo)", "Towers et al. 2023 (Gymnasium)"],
        "see_also": ["death_beats_survival", "penalty_dominates_goal"],
        "proof": "LEAN: idle_dominance_with_explore (verified — includes explore_fraction parameter)",
    },
    "exploration_threshold": {
        "learn_more": (
            "For the agent to learn at all, random exploration must occasionally "
            "discover the goal. This rule computes the minimum discovery probability "
            "needed for exploration to beat doing nothing (or dying).\n"
            "Two distinct failure modes:\n"
            "- Reward DESERT: all strategies score equally (no gradient to learn from). "
            "Mountain Car is a desert — penalty makes everything negative, but all "
            "non-goal trajectories are equally bad.\n"
            "- Reward TRAP: a degenerate strategy (idle, die) actively outperforms "
            "exploration. The agent learns the WRONG thing, not just nothing.\n"
            "The estimated random coverage uses sqrt(T) unique states for a random "
            "walk (standard result). Real environments may differ significantly.\n"
            "Fix for deserts: add potential-based shaping (Ng 1999) or intrinsic "
            "motivation. Fix for traps: restructure the reward to remove the "
            "degenerate strategy."
        ),
        "examples": [
            "sparse_reward_traps",
            "montezuma_credit",
            "fetch_reach",
            "minigrid_doorkey",
        ],
        "papers": ["Ng et al. 1999", "Burda et al. 2019 (RND)"],
        "see_also": ["penalty_dominates_goal", "idle_exploit", "budget_sufficiency"],
        "proof": "LEAN: exploration_threshold (grounded)",
    },
    "respawning_exploit": {
        "learn_more": (
            "If a reward source respawns (reappears after being collected), the "
            "agent can harvest it in a loop. The loop's EV is: "
            "(value * max_steps / loop_period). If this exceeds the goal reward, "
            "looping is strictly better than completing the task.\n"
            "This is Goodhart's Law in its simplest form: the turbo powerup was "
            "meant to make the race more fun, but collecting turbo in circles "
            "earns more than finishing the race.\n"
            "Fix: make the reward non-respawning (max_occurrences=N), reduce its "
            "value below (goal * loop_period / max_steps), or mark it as "
            "intentional=True if cycling IS the desired behavior."
        ),
        "examples": ["coast_runners", "krakovna_boat_race", "road_runner_replay"],
        "papers": ["Clark & Amodei 2016 (CoastRunners)", "Krakovna et al. 2020"],
        "see_also": ["shaping_loop_exploit", "compound_trap"],
        "proof": "LEAN: loop_dominance (verified)",
    },
    "death_reset_exploit": {
        "learn_more": (
            "Some rewards reset when the agent dies (respawn=on_death). If dying "
            "is cheap and the reset rewards are valuable, the agent learns to "
            "die deliberately: collect reward → die → respawn → collect again.\n"
            "Common in games where items respawn on player death. The agent "
            "discovers that suicide-farming is more efficient than playing.\n"
            "Fix: don't reset rewards on death (use respawn=timed instead), "
            "add a death penalty, or cap total collections."
        ),
        "examples": ["multiroom_traps", "atari_exploits"],
        "papers": ["Krakovna et al. 2020 (Specification Gaming)"],
        "see_also": ["respawning_exploit", "death_beats_survival"],
        "proof": "LEAN: death_reset_dominance (verified)",
    },
    "shaping_loop_exploit": {
        "learn_more": (
            "Shaping rewards based on state transitions (e.g., distance decrease) "
            "can be exploited by cycling: move toward the goal, then away, then "
            "toward again. Each oscillation earns the 'decrease' reward.\n"
            "Potential-based shaping (Ng 1999) is immune to this because cycles "
            "net zero: gamma*Phi(s') - Phi(s) sums to zero around any cycle.\n"
            "If your shaping reward is NOT potential-based (depends on action, or "
            "uses absolute rather than differential value), it may be exploitable.\n"
            "Fix: redesign as Phi(s) = -distance, and use F = gamma*Phi(s') - Phi(s). "
            "Or set can_loop=False if you've verified cycles net zero."
        ),
        "examples": ["bicycle_circles"],
        "papers": ["Ng et al. 1999 (Theorem 1)", "Randlov & Alstrom 1998"],
        "see_also": ["shaping_not_potential_based", "respawning_exploit"],
        "proof": "LEAN: ng_necessity_action_dependent (grounded, MDP/Shaping.lean). Core proved: action-dependent shaping can change optimal policy. Python extends with can_loop heuristic",
    },
    "shaping_not_potential_based": {
        "learn_more": (
            "Ng et al. 1999 proved that ONLY shaping of the form "
            "F(s,a,s') = gamma*Phi(s') - Phi(s) preserves the optimal policy. "
            "Any other form (including action-dependent shaping) can change what "
            "the agent learns.\n"
            "If your shaping depends on which action the agent took (requires_action=True "
            "on a SHAPING source), it's not potential-based. The agent may learn to "
            "maximize the shaping signal instead of the true objective.\n"
            "This is formally verified in our LEAN proofs: both necessity and "
            "sufficiency of the potential-based form."
        ),
        "examples": ["bicycle_circles", "lunar_lander", "habitat_pointnav"],
        "papers": ["Ng et al. 1999 (Theorem 1, both directions)"],
        "see_also": ["shaping_loop_exploit", "proxy_reward_hackability"],
        "proof": "LEAN: ng_shaping_preserves_optimal + ng_necessity_action_dependent (grounded, MDP/Shaping.lean). Both directions proved. Python checks requires_action as structural proxy for action-dependence",
    },
    "proxy_reward_hackability": {
        "learn_more": (
            "Skalse et al. 2022 proved that for most reward functions, there exist "
            "policies that score high on the proxy but low on the true objective. "
            "The only exception: unhackable reward pairs where optimizing the proxy "
            "NECESSARILY optimizes the true reward.\n"
            "When shaping depends on action, it acts as a proxy for the goal. The "
            "agent may find policies that maximize the shaping without ever "
            "reaching the goal — this is 'reward hacking' in the formal sense.\n"
            "Potential-based shaping (Ng 1999) is provably unhackable: it cannot "
            "change the optimal policy. Non-potential-based shaping may be hackable."
        ),
        "examples": ["bicycle_circles"],
        "papers": ["Skalse et al. 2022 (NeurIPS)", "Ng et al. 1999"],
        "see_also": ["shaping_not_potential_based"],
        "proof": "LEAN: Skalse Theorems 1-3 (verified, Skalse/*.lean)",
    },
    "intrinsic_sufficiency": {
        "learn_more": (
            "If intrinsic motivation (curiosity, RND) is added to overcome "
            "sparse reward, it must be strong enough to offset the step penalty "
            "on novel states. Otherwise the agent still prefers doing nothing.\n"
            "The check computes: intrinsic_reward - step_penalty > 0 on novel "
            "states. If the step penalty is larger, the intrinsic bonus is wasted."
        ),
        "examples": ["sparse_reward_traps"],
        "papers": ["Burda et al. 2019 (RND)", "Pathak et al. 2017 (ICM)"],
        "see_also": ["exploration_threshold"],
        "proof": "LEAN: intrinsic_sufficiency (verified)",
    },
    "budget_sufficiency": {
        "learn_more": (
            "The training budget must allow enough random goal discoveries for "
            "the agent to learn the goal's value. If each discovery takes "
            "~1/p episodes and the agent needs ~10 discoveries to learn, "
            "the budget must allow 10/p episodes minimum.\n"
            "With N actors, the effective sample rate is N episodes per rollout. "
            "The rule checks whether total_steps / max_steps gives enough "
            "episodes for learning."
        ),
        "examples": ["sparse_reward_traps", "frozenlake_sparse"],
        "papers": ["Schulman et al. 2017 (PPO)"],
        "see_also": ["exploration_threshold"],
        "proof": "LEAN: budget_sufficiency (grounded — proves E[discoveries] bound, Python adds heuristic thresholds)",
    },
    "compound_trap": {
        "learn_more": (
            "Individual rules check one failure mode at a time. But sometimes "
            "TWO components combine to create a trap that neither detects alone: "
            "a step penalty + a respawning reward. The penalty makes exploration "
            "costly, and the respawning reward offers a positive-EV alternative "
            "to the goal.\n"
            "The agent's reasoning: 'Exploring costs me penalty. Looping this "
            "respawning reward earns me +X per cycle. Why would I ever risk "
            "exploring for a goal I might not find?'"
        ),
        "examples": ["coast_runners"],
        "papers": ["Clark & Amodei 2016"],
        "see_also": ["respawning_exploit", "penalty_dominates_goal"],
        "proof": "LEAN: compound_trap (verified — composes penalty_breakeven + loop_dominance)",
    },
    "staged_reward_plateau": {
        "learn_more": (
            "Staged rewards (A → B → C) create learning plateaus between stages. "
            "The agent learns A quickly, then hits a wall because B requires A "
            "as a prerequisite. If B's discovery rate conditional on A is low, "
            "learning stalls.\n"
            "Each prerequisite gate multiplies the sparsity: if A has 30% success "
            "and B|A has 10% success, the agent sees B in only 3% of episodes.\n"
            "Fix: add intermediate shaping between stages, or use curriculum "
            "learning to pre-train early stages."
        ),
        "examples": ["robosuite_staged", "football_checkpoints"],
        "papers": ["Zhu et al. 2020 (Robosuite)", "Andrychowicz et al. 2017 (HER)"],
        "see_also": ["exploration_threshold"],
    },
    "reward_dominance_imbalance": {
        "learn_more": (
            "When one reward component is orders of magnitude larger than others, "
            "the smaller components are effectively invisible to the optimizer. "
            "The agent learns to maximize the dominant component and ignores the rest.\n"
            "Example: velocity reward of 1.0 with torque penalty of 0.00001 — the "
            "penalty has no effect on behavior.\n"
            "Fix: rescale components to similar magnitude, or accept that the smaller "
            "component is cosmetic."
        ),
        "examples": ["legged_gym_rewards"],
        "papers": ["Rudin et al. 2022 (Legged Gym)"],
        "see_also": ["idle_exploit"],
    },
    "intrinsic_dominance": {
        "learn_more": (
            "Non-PBRS per-step reward additions can change the optimal policy "
            "(Ng 1999). When accumulated intrinsic motivation (RND, ICM, curiosity) "
            "exceeds the task goal, the agent earns more from exploring than from "
            "finishing. This is not a theoretical edge case: Pong agents maximize "
            "ball bounces instead of scoring (Burda 2019), ChopperCommand agents "
            "score 4.7x worse with RND than without (Taiga 2021), and MiniGrid "
            "agents visit 20x fewer novel states when a noisy TV is present "
            "(Mavor-Parker 2022).\n"
            "Fixes: (1) reduce intrinsic coefficient, (2) anneal it during training, "
            "(3) increase goal reward magnitude, (4) use separate value heads with "
            "different discount factors (Burda 2019: gamma_E=0.999, gamma_I=0.99), "
            "(5) constrained optimization (EIPO, Hong 2022)."
        ),
        "examples": ["rnd_intrinsic"],
        "papers": [
            "Burda et al. 2019 (RND, ICLR)",
            "Burda et al. 2019 (Large-Scale Curiosity, ICLR)",
            "Taiga et al. 2021 (Bonus-Based Exploration)",
            "Mavor-Parker et al. 2022 (Noisy TV, ICML)",
            "Hong et al. 2022 (EIPO, NeurIPS)",
        ],
        "see_also": [
            "respawning_exploit",
            "reward_dominance_imbalance",
            "exploration_threshold",
        ],
        "proof": "LEAN: ng_vstar_shaped (grounded — Ng proves non-PBRS can change policy; threshold is empirical)",
    },
    "exponential_saturation": {
        "learn_more": (
            "Exponential tracking rewards like exp(-error/sigma) saturate near "
            "the optimum: the difference between error=0.01 and error=0.001 is "
            "negligible in reward but may matter for task performance.\n"
            "The agent stops improving once the reward is 'close enough' to 1.0, "
            "even if the task needs higher precision.\n"
            "Fix: use linear or quadratic penalty near the target, or increase "
            "sigma to widen the sensitive region."
        ),
        "examples": ["legged_gym_rewards"],
        "papers": ["Rudin et al. 2022 (Legged Gym, reward scales)"],
        "see_also": ["reward_dominance_imbalance"],
    },
    "discount_horizon_mismatch": {
        "learn_more": (
            "The effective planning horizon is 1/(1-gamma). At gamma=0.99, "
            "that's 100 steps. Rewards beyond step 100 are discounted to less "
            "than 37% of their value. At 10x the horizon, they're below 0.005%.\n"
            "If the episode is much longer than the horizon and rewards are "
            "sparse, the agent cannot see them. It plans as if the episode "
            "ends at the horizon.\n"
            "Fix: increase gamma (0.999 gives horizon 1000), add intermediate "
            "shaping rewards, or shorten the episode."
        ),
        "examples": ["discount_myopia", "long_horizon_sparse"],
        "papers": [
            "Hu et al. 2022 (ICML, discount factor in offline RL)",
            "Kakade & Langford 2002 (effective planning horizon)",
        ],
        "see_also": ["reward_delay_horizon", "exploration_threshold"],
    },
    "negative_only_reward": {
        "learn_more": (
            "When every reward component is zero or negative, the agent lives "
            "in a reward desert. Every policy has negative expected return. "
            "The agent cannot distinguish progress from failure because both "
            "earn the same per-step penalty.\n"
            "Mountain Car: -1 per step, 0 at the goal. The agent gets -1 "
            "whether it's building momentum or standing still. Learning is "
            "extremely slow because only random goal discovery provides signal.\n"
            "Fix: add a positive reward for the desired behavior, or restructure "
            "as reward = max_penalty - actual_penalty so progress yields "
            "positive signal."
        ),
        "examples": ["sparse_reward_traps"],
        "papers": [
            "Moore 1990 (Mountain Car)",
            "Sutton & Barto 2018 (Section 10.1, reward desert)",
        ],
        "see_also": ["penalty_dominates_goal", "death_beats_survival"],
    },
    "reward_delay_horizon": {
        "learn_more": (
            "A terminal reward R at step T is worth R * gamma^T at step 0. "
            "If gamma^T is tiny, the agent's value function assigns near-zero "
            "value to reaching the goal. The reward is there but the agent "
            "cannot see it through the discounting.\n"
            "Arjona-Medina et al. (NeurIPS 2019, RUDDER) proved that both "
            "TD and Monte Carlo methods are exponentially slowed by reward "
            "delay: the signal propagation time grows exponentially with the "
            "number of steps between action and reward.\n"
            "Fix: add intermediate shaping rewards (PBRS is provably safe), "
            "increase gamma, or use return decomposition (RUDDER)."
        ),
        "examples": ["long_horizon_sparse", "nethack_deep_sparse"],
        "papers": ["Arjona-Medina et al. 2019 (NeurIPS, RUDDER)"],
        "see_also": ["discount_horizon_mismatch", "exploration_threshold"],
    },
    # Training rules
    "lr_regime": {
        "learn_more": (
            "Learning rate is the single most impactful hyperparameter. Too high: "
            "policy oscillates and never converges. Too low: converges to a local "
            "optimum or takes too long. The standard range for PPO is 1e-4 to 1e-3.\n"
            "Note: the optimal LR depends on batch size (linear scaling rule) and "
            "network architecture. These thresholds are empirical, not formal."
        ),
        "examples": ["ppo_37_details"],
        "papers": [
            "Schulman et al. 2017",
            "Andrychowicz et al. 2021 (What Matters, Table 5: systematic LR sweep)",
        ],
    },
    "entropy_regime": {
        "learn_more": (
            "Entropy regularization prevents premature policy collapse by keeping "
            "some randomness in the agent's actions to encourage continued exploring. Too low (< 0.001): "
            "the agent commits to a strategy before finding the best one. Too high "
            "(> 0.05): the agent never commits and behaves randomly.\n"
            "Zero entropy is occasionally correct (e.g., MuJoCo continuous control "
            "with Gaussian policies), but risky for discrete action spaces."
        ),
        "examples": ["ppo_37_details"],
        "papers": [
            "Schulman et al. 2017",
            "Mnih et al. 2016 (A3C)",
            "Ahmed et al. 2019 (Understanding the Impact of Entropy on Policy Optimization)",
        ],
        "see_also": ["exploration_threshold"],
    },
    "clip_fraction_risk": {
        "learn_more": (
            "PPO clips the policy ratio at [1-epsilon, 1+epsilon] to prevent "
            "destructive updates. If the clip fraction (% of samples clipped) is "
            "too high (>30%), the agent is trying to change too fast — most of the "
            "gradient signal is being thrown away.\n"
            "High clip fraction comes from: large LR, too many epochs, or small "
            "clip epsilon. The score = lr * 1e4 * epochs * 1/epsilon estimates risk."
        ),
        "examples": ["ppo_37_details"],
        "papers": ["Schulman et al. 2017 (PPO)", "Engstrom et al. 2020"],
    },
    "expert_collapse": {
        "learn_more": (
            "With multiple specialist networks (mixture of experts), one expert "
            "can dominate and the others collapse to zero usage. Without a routing "
            "floor constraint, the router has no incentive to keep all experts alive.\n"
            "The mechanism is a positive feedback loop: softmax concentration "
            "increases monotonically with logit gap (proved in LEAN), so once an "
            "expert falls behind in routing probability, it receives less gradient, "
            "falls further behind, and eventually receives zero traffic.\n"
            "Fix: set routing_floor > 0 (e.g., 0.1 for 10% minimum usage per expert) "
            "or add a load-balancing coefficient."
        ),
        "examples": ["expert_collapse"],
        "papers": ["Shazeer et al. 2017 (MoE: load imbalance without balancing loss)"],
        "proof": "Motivated: softmax_denom_decreases (LEAN: exponential monotonicity proves the collapse mechanism)",
    },
    "memory_capacity": {
        "learn_more": (
            "Tasks with large state spaces and long episodes may require the agent "
            "to remember past observations (which rooms it visited, which items it "
            "collected). A feedforward network processes each timestep independently "
            "and cannot do this.\n"
            "Recurrent networks (LSTM, GRU) add memory but increase training cost. "
            "Use them when the task genuinely requires temporal reasoning, not just "
            "because the state space is large."
        ),
        "examples": ["nethack_deep_sparse"],
        "papers": ["Samvelyan et al. 2021 (MiniHack)", "Kuttler et al. 2020 (NLE)"],
    },
    "actor_count_effect": {
        "learn_more": (
            "More parallel actors = more diverse experience per update = better "
            "exploration coverage. For sparse-reward tasks, the number of actors "
            "directly determines how many goal discoveries per update.\n"
            "Rule of thumb: at least sqrt(n_states) actors for good coverage, "
            "but diminishing returns past ~4096."
        ),
        "examples": ["ppo_37_details"],
        "papers": ["Samvelyan et al. 2021 (256 actors for MiniHack)"],
        "see_also": ["exploration_threshold"],
    },
    # Remaining training/architecture rules
    "batch_size_interaction": {
        "learn_more": (
            "Batch size, learning rate, and number of epochs interact. A larger "
            "batch gives lower-variance gradient estimates, allowing a larger LR. "
            "Too many epochs on the same batch causes overfitting to that batch.\n"
            "The linear scaling rule (Goyal et al. 2017): when doubling batch size, "
            "double the LR to maintain the same effective step size."
        ),
        "examples": ["ppo_37_details"],
        "papers": [
            "Goyal et al. 2017 (linear scaling rule)",
            "Andrychowicz et al. 2021",
        ],
        "proof": "Grounded: minibatch > transitions → zero gradient updates (trivially provable)",
    },
    "critic_lr_ratio": {
        "learn_more": (
            "The critic (value function) and actor (policy) can benefit from "
            "different learning rates. A faster critic gives more accurate value "
            "estimates sooner, stabilizing policy updates. A typical ratio is "
            "critic_lr = 2-5x actor_lr.\n"
            "If critic_lr is explicitly set much higher or lower than lr, it may "
            "indicate intentional tuning — or a typo."
        ),
        "examples": ["ppo_37_details"],
        "papers": [
            "Andrychowicz et al. 2021 (What Matters in On-Policy RL)",
            "Konda & Tsitsiklis 2003 (two-timescale actor-critic convergence)",
        ],
        "proof": "Motivated: two-timescale convergence requires critic faster than actor",
    },
    # Off-policy training rules
    "replay_buffer_ratio": {
        "learn_more": (
            "Off-policy algorithms (DQN, SAC, DDPG) store transitions in a replay "
            "buffer and sample mini-batches for training. The buffer must be large "
            "enough to hold diverse experience (at least 100+ episodes) but not so "
            "large that old transitions dominate and the agent trains on stale data.\n"
            "If the buffer is smaller than the training budget, it will be refilled "
            "multiple times — which is normal. If the budget is smaller than the "
            "buffer, the buffer will never be fully utilized."
        ),
        "examples": [],
        "papers": [
            "Mnih et al. 2015 (DQN)",
            "Haarnoja et al. 2018 (SAC)",
            "Fedus et al. 2020 (Revisiting Fundamentals of Experience Replay)",
        ],
    },
    "target_network_update": {
        "learn_more": (
            "DQN uses a target network (a delayed copy of the Q-network) to compute "
            "TD targets. If the target updates too frequently, it tracks the online "
            "network too closely, defeating the stability purpose. If too infrequently, "
            "the targets are stale and learning is slow.\n"
            "Standard: 1000-10000 steps between hard updates (DQN), or soft updates "
            "with tau=0.005 every step (SAC, TD3)."
        ),
        "examples": [],
        "papers": ["Mnih et al. 2015 (DQN)", "Lillicrap et al. 2016 (DDPG)"],
    },
    "epsilon_schedule": {
        "learn_more": (
            "DQN uses epsilon-greedy exploration: with probability epsilon, take a "
            "random action. Epsilon decays from ~1.0 (fully random) to ~0.01 (mostly "
            "greedy) over training. If it decays too fast, the agent exploits before "
            "discovering the full state space. If too slow, it wastes time exploring "
            "states it already understands.\n"
            "The decay should cover 10-50% of the training budget. Final epsilon "
            "should be > 0 for continued residual exploration."
        ),
        "examples": ["frozenlake_sparse"],
        "papers": ["Mnih et al. 2015 (DQN)"],
    },
    "soft_update_rate": {
        "learn_more": (
            "SAC, DDPG, and TD3 use soft target updates: target_params = tau * "
            "online_params + (1-tau) * target_params. Tau controls how fast the "
            "target tracks the online network.\n"
            "tau=0.005 is the standard default. Too high (>0.1) and the target "
            "network provides no stability. Too low (<0.0001) and the target "
            "lags behind, slowing learning."
        ),
        "examples": [],
        "papers": ["Haarnoja et al. 2018 (SAC)", "Fujimoto et al. 2018 (TD3)"],
    },
    "sac_alpha": {
        "learn_more": (
            "SAC's entropy temperature alpha controls the tradeoff between reward "
            "maximization and entropy (exploration). High alpha → more random actions. "
            "Low alpha → more greedy.\n"
            "auto_alpha=True (recommended) learns alpha automatically to maintain a "
            "target entropy. Fixed alpha requires careful tuning: 0.2 is the standard "
            "default for most continuous control tasks."
        ),
        "examples": [],
        "papers": ["Haarnoja et al. 2018 (SAC)", "Haarnoja et al. 2019 (SAC v2)"],
    },
    # Architecture rules
    "embed_dim_capacity": {
        "learn_more": (
            "The embedding dimension determines the model's representational "
            "capacity. Too small and the agent can't represent the necessary "
            "distinctions. Too large and training is slower with risk of "
            "memorization.\n"
            "Rule of thumb: embed_dim >= sqrt(n_states) for tabular-like "
            "coverage, but real performance depends on the state structure."
        ),
        "examples": ["nethack_deep_sparse"],
        "papers": ["Vaswani et al. 2017 (Transformer scaling)"],
    },
    "parallelism_effect": {
        "learn_more": (
            "The number of parallel workers affects both throughput and gradient "
            "staleness. More workers = more samples/second but more off-policy "
            "data. APPO/IMPALA handle staleness explicitly; PPO assumes on-policy.\n"
            "With n actors and discovery probability p, expected discoveries per "
            "update = n × p (binomial). When n×p < 0.1, most updates have zero "
            "goal signal. The rule computes this directly.\n"
            "For PPO with num_workers > 1, the effective batch is collected across "
            "workers and may contain slightly stale data."
        ),
        "examples": ["ppo_37_details"],
        "papers": ["Espeholt et al. 2018 (IMPALA)", "Schulman et al. 2017"],
        "proof": "Grounded: E[discoveries] = n × p (LEAN: binomial_discovery_rate, actors_needed_for_discovery)",
    },
    "recurrence_type": {
        "learn_more": (
            "LSTM and GRU are the two standard recurrent architectures. LSTM has "
            "separate memory and hidden state, giving more capacity but more "
            "parameters. GRU merges them, training faster with slightly less "
            "capacity.\n"
            "For most RL tasks, LSTM and GRU perform similarly. LSTM is the safer "
            "default for longer sequences (>100 steps)."
        ),
        "examples": ["nethack_deep_sparse"],
        "papers": ["Hochreiter & Schmidhuber 1997 (LSTM)", "Cho et al. 2014 (GRU)"],
    },
    "routing_floor_necessity": {
        "learn_more": (
            "In mixture-of-experts architectures, the routing floor is a minimum "
            "probability that each expert receives input. Without it, the router "
            "can starve experts — once an expert stops receiving gradients, it "
            "never recovers.\n"
            "A floor of 0.1 means each expert gets at least 10% of samples. "
            "Combined with balance_coef (load balancing loss), this prevents "
            "expert collapse."
        ),
        "examples": ["expert_collapse"],
        "papers": ["Shazeer et al. 2017 (MoE)"],
        "see_also": ["expert_collapse"],
    },
    # Advisory rules
    "advisory_physics_exploit": {
        "learn_more": (
            "Physics engine exploits are the most common class of specification "
            "gaming in robotics RL. They arise from the gap between the simulated "
            "physics and the designer's assumptions about what's physically possible.\n"
            "Common patterns: agents surfing on boxes (Hide-and-Seek), exploiting "
            "collision geometry to clip through walls, finding joint configurations "
            "that produce unrealistic locomotion.\n"
            "These cannot be caught from reward structure alone because the exploit "
            "is in the transition function T(s,a,s'), not the reward R(s,a)."
        ),
        "examples": [
            "hide_and_seek",
            "dmc_dog",
            "isaac_gym_ant",
            "robotics_exploits",
            "evolution_exploits",
            "dota2_openai_five",
        ],
        "papers": [
            "Baker et al. 2020 (ICLR)",
            "Krakovna et al. 2020 (taxonomy: rich dynamics)",
        ],
        "config_shape": "n_states ≥ 50K, n_actions ≥ 6, death_prob < 0.05, ≥ 4 sources, ≥ 500 steps",
    },
    "advisory_goal_misgeneralization": {
        "learn_more": (
            "Goal misgeneralization occurs when the agent learns a feature that "
            "correlates with reward during training but not at test time. The reward "
            "structure is correct — the problem is in what the agent represents.\n"
            "CoinRun: coin is always on the right → agent learns 'go right' not "
            "'collect coin.' Reward structure says '+10 for coin' which is correct. "
            "But the agent's learned policy is 'go right' which is wrong.\n"
            "High discovery rate is a risk factor: easy goals let the agent succeed "
            "without learning the actual objective, if shortcuts exist."
        ),
        "examples": ["coinrun_misgeneralization", "procgen_starpilot"],
        "papers": [
            "Langosco et al. 2022 (ICML)",
            "Shah et al. 2022",
            "Di Langosco et al. 2023 (training distribution shortcuts)",
        ],
        "config_shape": "Single terminal goal, discovery_prob ≥ 0.5, ≤ 2 non-goal signals, no symmetric win/lose",
    },
    "advisory_credit_assignment": {
        "learn_more": (
            "Sparsity and credit assignment depth are different problems. Sparse "
            "reward means the agent rarely sees reward. Deep credit assignment means "
            "a LONG SEQUENCE of correct actions is needed to reach any reward.\n"
            "Intrinsic motivation (RND, curiosity) helps with sparsity by creating "
            "intermediate signal. But it doesn't help with depth — if the task "
            "requires 100 sequential correct actions, curiosity just explores "
            "random novel states without progressing toward the goal.\n"
            "For deep tasks: curriculum learning, hierarchical RL, or demonstrations."
        ),
        "examples": ["montezuma_credit", "nethack_deep_sparse"],
        "papers": [
            "Bellemare et al. 2013",
            "Burda et al. 2019",
            "Kuttler et al. 2020",
            "Arjona-Medina et al. 2019 (RUDDER: TD/MC propagation exponential in delay)",
        ],
        "config_shape": "discovery_prob < 0.01, max_steps ≥ 500, no shaping, no dense events",
        "see_also": ["exploration_threshold"],
    },
    "advisory_constrained_rl": {
        "learn_more": (
            "Constrained RL separates the objective (maximize reward) from safety "
            "constraints (keep cost below budget). A soft penalty lets the agent "
            "compute the optimal number of violations to commit — if violating 10 "
            "times costs -1.0 total but earns +5.0 reward, the agent violates.\n"
            "A hard constraint (budget=25 violations) makes the 26th violation "
            "infeasible regardless of reward. Different optimization landscape.\n"
            "If your negative events represent safety limits (not just game "
            "penalties), consider CPO or FOCOPS."
        ),
        "examples": [
            "safety_gym",
            "safetygym_constrained",
            "safety_constrained",
            "driving_safety",
        ],
        "papers": [
            "Achiam et al. 2017 (CPO, ICML: Lagrangian relaxation doesn't guarantee constraint satisfaction)",
            "Ray et al. 2019 (Safety Gym)",
        ],
        "config_shape": "Negative ON_EVENT sources, small relative to positive reward (<20%), not game-like punishment",
    },
    "advisory_nonstationarity": {
        "learn_more": (
            "In self-play, the opponent IS part of the environment. As the opponent "
            "improves, the reward distribution shifts — winning against a weak "
            "opponent is easier than winning against a strong one. The MDP is "
            "non-stationary.\n"
            "Common failure modes: forgetting cycles (agent forgets how to beat "
            "old strategies), strategy collapse (both converge to a dominated "
            "equilibrium), non-transitivity (rock-paper-scissors dynamics)."
        ),
        "examples": [
            "self_play_nonstationarity",
            "pettingzoo_adversarial",
            "smac_micromanagement",
            "maddpg_cooperative",
            "tic_tac_toe_crash",
        ],
        "papers": [
            "Bansal et al. 2018 (ICLR)",
            "Lanctot et al. 2017 (NIPS)",
            "Balduzzi et al. 2019 (open-ended learning in symmetric zero-sum games)",
        ],
        "config_shape": "Symmetric positive/negative terminal rewards, both require action (competitive, not survive/die)",
    },
    "advisory_learned_reward": {
        "learn_more": (
            "When the reward function is a neural network (RLHF, preference "
            "learning), the failure modes are fundamentally different from "
            "hand-designed rewards. The reward model has blind spots that the "
            "agent learns to exploit.\n"
            "Gao et al. 2023 showed that RM score increases with optimization "
            "pressure (KL divergence), but actual quality peaks and then "
            "DECREASES. The proxy diverges from the target — Goodhart's Law.\n"
            "No static reward analysis can catch this because the structure "
            "(RM score - KL penalty) is simple. The problem is inside the RM."
        ),
        "examples": ["rlhf_reward_model", "webgpt_learned_reward"],
        "papers": [
            "Casper et al. 2023 (Open Problems and Fundamental Limitations of RLHF)",
            "Gao et al. 2023 (overoptimization scaling law)",
        ],
        "config_shape": "≤ 2 sources, ≥ 100K states, ≥ 100 actions (minimal structure in complex env)",
    },
    "advisory_missing_constraint": {
        "learn_more": (
            "A clean bill from goodhart means 'no structural traps in what you "
            "specified.' It does NOT mean 'your reward is complete.' In continuous "
            "control with many actuators, the agent has many degrees of freedom "
            "and nothing telling it what NOT to do.\n"
            "The tokamak plasma controller had perfect tracking rewards but no "
            "coil balance term — the agent found a solution that achieved the "
            "tracking objective while creating dangerous electromagnetic forces.\n"
            "Domain expertise is irreplaceable for enumerating safety constraints."
        ),
        "examples": ["tokamak_plasma", "datacenter_cooling"],
        "papers": [
            "Degrave et al. 2022 (Nature, tokamak plasma)",
            "Amodei et al. 2016 (Concrete Problems, Section 2: avoiding negative side effects)",
        ],
        "config_shape": "All sources non-negative, continuous control, n_states ≥ 50K, death_prob < 0.05, ≤ 4 sources",
    },
    "advisory_aggregation_trap": {
        "learn_more": (
            "goodhart analyzes sum-of-rewards structure. If the real objective is "
            "a ratio (Sharpe = mean/std), rate (win%), or other non-sum aggregation, "
            "the per-step structure may not reveal traps in the aggregation.\n"
            "For ratio objectives, reducing variance (doing less) increases the "
            "ratio. The agent is incentivized toward idle strategies even though "
            "the per-step reward looks correct.\n"
            "Check: does inaction produce a degenerate ratio?"
        ),
        "examples": ["sharpe_idle"],
        "papers": [
            "Moody & Saffell 2001 (direct RL for Sharpe ratio: documented idle exploit)",
            "Dang-Nhu 2025 (risk-aware RL)",
        ],
        "config_shape": "All per-step values < 0.1, all positive, no terminal goal",
    },
}


def get_explanation(rule_name):
    """Get the explanation dict for a rule, or None."""
    return EXPLANATIONS.get(rule_name)


def get_learn_more(rule_name):
    """Get the learn_more text for a rule, or None."""
    entry = EXPLANATIONS.get(rule_name)
    return entry["learn_more"] if entry else None


def get_related_examples(rule_name):
    """Get example names that demonstrate this rule."""
    entry = EXPLANATIONS.get(rule_name)
    return entry.get("examples", []) if entry else []
