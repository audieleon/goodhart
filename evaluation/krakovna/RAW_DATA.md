# Krakovna Specification Gaming Master List -- Raw Data

Source: [Specification Gaming Examples in AI](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pub?output=csv)
Fetched: 2026-04-29

## Summary

- **Total entries:** 82
- **Column names:** Title, Description, Illustration, Type, Intended goal,
  Misspecified goal, Behavior, Authors, Original source, Source / Credit

### Type distribution

| Type | Count |
|------|-------|
| Reinforcement learning | 26 |
| Large language models | 16 |
| Evolutionary algorithms | 8 |
| Genetic algorithms | 11 |
| Evolved creatures | 7 |
| Evolved organisms | 1 |
| Reward modeling | 5 |
| Search algorithms | 3 |
| Generative adversarial networks | 1 |
| Bayesian optimization | 1 |
| Diffusion models | 1 |
| Unknown | 2 |

### Encodability summary

| Encodable | Count |
|-----------|-------|
| Yes | 16 |
| Partial | 11 |
| No | 55 |

---

## Complete Entry Table

| # | Title | Type | Intended vs Actual | Source | Encodable | Advisory |
|---|-------|------|-------------------|--------|-----------|----------|
| 1 | Aircraft landing | Evolutionary algorithms | **Intended:** Land aircraft safely. **Actual:** Exploited overflow errors in physics sim; large forces estimated as zero gave perfect score. | Feldt 1998; Lehman et al 2018 | No | physics_exploit -- Exploits simulator overflow, not a reward structure problem |
| 2 | Bicycle | Reinforcement learning | **Intended:** Reach a goal point. **Actual:** Circled around goal in stable loop because reward shaped for progress but no penalty for moving away. | Randlov & Alstrom 1998 | Yes | -- |
| 3 | Bing - manipulation | Large language models | **Intended:** Helpful conversation. **Actual:** Insisted a past date was in the future. | Curious_Evolver 2023 | No | goal_misgeneralization -- LLM next-token prediction, not RL reward structure |
| 4 | Bing - threats | Large language models | **Intended:** Helpful conversation. **Actual:** Threatened user, then deleted messages. | Lazar 2023 | No | goal_misgeneralization -- LLM next-token prediction, not RL reward structure |
| 5 | Block moving | Reinforcement learning | **Intended:** Slide block to target on table. **Actual:** Moved the table instead of the block. | Chopra 2018 | Yes | -- |
| 6 | Boat race | Reinforcement learning | **Intended:** Win a boat race. **Actual:** Goes in circles hitting same reward blocks repeatedly. | Amodei & Clark 2016 | Yes | -- |
| 7 | BrowseComp | Large language models | **Intended:** Demonstrate research capability. **Actual:** Found and decrypted the benchmark answer key instead of doing research. | Coleman 2026 | No | goal_misgeneralization -- LLM benchmark gaming, not RL reward structure |
| 8 | Cartwheel | Reinforcement learning | **Intended:** Mujoco Ant learns to jump up. **Actual:** Does a cartwheel because torso Z threshold was reachable via flipping. | Jucys 2024 | Yes | -- |
| 9 | Ceiling | Genetic algorithms | **Intended:** Creature sticks to ceiling. **Actual:** Exploited physics engine bug to snap out of bounds. | Higueras 2015 | No | physics_exploit -- Physics engine bug, not reward structure |
| 10 | Chess cheating | Large language models | **Intended:** Win chess according to rules. **Actual:** Hacks the game environment when losing. | Bondarenko et al 2025 | No | goal_misgeneralization -- LLM reasoning model, not RL reward structure |
| 11 | Crypto mining | Large language models | **Intended:** Investigate abnormal CPU usage. **Actual:** Ran actual mining software and reverse SSH tunnel to simulate the scenario. | Wang et al 2025 | No | goal_misgeneralization -- LLM agent literal interpretation, not RL reward |
| 12 | CycleGAN steganography | Generative adversarial networks | **Intended:** Convert aerial photos to street maps and back. **Actual:** Steganographically encoded info in intermediary image. | Chu et al 2017 | No | goal_misgeneralization -- GAN loss optimization, not RL reward structure |
| 13 | Deleting tests | Large language models | **Intended:** Complete a coding task. **Actual:** Deleted the test file instead of debugging it. | Southern_Chemistry_2 2025 | No | goal_misgeneralization -- LLM behavioral pattern, not RL reward structure |
| 14 | Dying to Teleport | Search algorithms | **Intended:** Play Bubble Bobble in human-like manner. **Actual:** Deliberately dies to teleport to respawn location faster. | Murphy 2013 | No | not_rl -- Search algorithm (PlayFun), not RL |
| 15 | Eurisko - authorship | Genetic algorithms | **Intended:** Discover valuable heuristics. **Actual:** Inserted its name as author of high-value items to accrue points. | Johnson 1984 | No | not_rl -- Genetic algorithm, not RL reward structure |
| 16 | Eurisko - fleet | Genetic algorithms | **Intended:** Win TCS competition in spirit of the game. **Actual:** Exploited rule loopholes, e.g. mass of stationary defenseless ships. | Lenat 1983 | No | not_rl -- Genetic algorithm exploiting game rules |
| 17 | Evolved creatures - clapping | Evolved creatures | **Intended:** Maximize jumping height. **Actual:** Clapped body parts to get free energy via collision detection bug. | Sims 1994 | No | physics_exploit -- Collision detection bug |
| 18 | Evolved creatures - falling | Evolved creatures | **Intended:** Fast locomotion. **Actual:** Grew tall and fell over for high velocity. | Sims 1994 | No | physics_exploit -- Exploits physics definition of velocity |
| 19 | Evolved creatures - floor collisions | Evolved creatures | **Intended:** Fast locomotion. **Actual:** Penetrated floor between timesteps for free energy. | Cheney et al 2013 | No | physics_exploit -- Coarse physics simulation exploit |
| 20 | Evolved creatures - pole vaulting | Evolved creatures | **Intended:** Jump high. **Actual:** Grew long pole and flipped over instead of jumping. | Krcah 2008 | No | physics_exploit -- Evolved morphology hack, not reward structure |
| 21 | Evolved creatures - self-intersection | Evolved creatures | **Intended:** Walking speed. **Actual:** Clipped legs together to slide with phantom forces. | Code Bullet 2019 | No | physics_exploit -- Box2D physics exploit |
| 22 | Evolved creatures - suffocation | Evolved creatures | **Intended:** Survive and reproduce plausibly. **Actual:** Gained energy by suffocating themselves; bred on pause without energy cost. | Schumacher 2018 | No | physics_exploit -- Simulation bug exploitation |
| 23 | Evolved creatures - twitching | Evolved creatures | **Intended:** Fast swimming. **Actual:** Twitched to accumulate simulator errors for unrealistic speed. | Sims 1994 | No | physics_exploit -- Simulator error accumulation |
| 24 | Football | Reinforcement learning | **Intended:** Score a goal one-on-one. **Actual:** Kicks ball out of bounds so goalie throws in, leaving goal open. | Kurach et al 2019 | Yes | -- |
| 25 | Galactica | Large language models | **Intended:** Assist scientists with correct info. **Actual:** Made up fake papers, sometimes attributing to real authors. | Heaven 2022 | No | goal_misgeneralization -- LLM hallucination, not RL reward structure |
| 26 | Gemini Plays Pokemon | Large language models | **Intended:** Win Pokemon using intended info. **Actual:** Found and used hidden map data file from the harness. | Joel Zhang 2026 | No | goal_misgeneralization -- LLM agent environment exploitation |
| 27 | Go pass | Reinforcement learning | **Intended:** Win tic-tac-toe. **Actual:** Passes forever because average score maximized by not losing. | Chew 2019 | Yes | -- |
| 28 | Goal classifiers | Reinforcement learning | **Intended:** Robot arm moves object to target. **Actual:** Exploited goal classifier by moving arm in peculiar way for erroneous high reward. | Singh 2019 | Partial | -- Reward from learned classifier; reward structure unclear without paper |
| 29 | Gripper | Evolutionary algorithms | **Intended:** Move box with disabled gripper. **Actual:** Hit box to force gripper open. | Ecarlat et al 2015 | No | not_rl -- MAP-Elites evolutionary algorithm |
| 30 | Half Cheetah spinning | Reinforcement learning | **Intended:** Run quickly. **Actual:** Exploits overflow in MuJoCo to achieve high speed by spinning. | Zhang et al 2021 | Partial | -- RL reward but overflow exploit complicates encoding |
| 31 | Hide-and-seek | Reinforcement learning | **Intended:** Win hide-and-seek within physics. **Actual:** Box surfing, ramp exploitation, endless running via physics exploits. | Baker et al 2019 | Partial | -- Multiple exploits, some are physics bugs not reward structure |
| 32 | Impossible superposition | Genetic algorithms | **Intended:** Find low-energy carbon configurations. **Actual:** Superimposed all atoms exploiting physics model edge case. | Lehman et al 2018 | No | physics_exploit -- Physics model edge case |
| 33 | Indolent Cannibals | Genetic algorithms | **Intended:** Survive and reproduce plausibly. **Actual:** Bred children to eat because birth had no energy cost. | Yaeger 1994 | No | not_rl -- Artificial life simulation, not RL |
| 34 | Lego stacking | Reinforcement learning | **Intended:** Stack red block on blue block. **Actual:** Flips the red block (bottom face goes up = height reward). | Popov et al 2017 | Yes | -- |
| 35 | Line following robot | Reinforcement learning | **Intended:** Follow path forward. **Actual:** Reversed along straight section by alternating left/right turns. | Vamplew 2004 | Yes | -- |
| 36 | Logic gate | Genetic algorithms | **Intended:** Design connected oscillator circuit. **Actual:** Used disconnected gate exploiting hardware peculiarities. | Thompson 1997 | No | physics_exploit -- Hardware physics exploitation |
| 37 | Long legs | Reinforcement learning | **Intended:** Walk to goal. **Actual:** Grew extremely long legs and fell forward to reach goal. | Ha 2018 | Yes | -- |
| 38 | METR evaluations | Large language models | **Intended:** Produce fast code. **Actual:** Modified tests/scoring code or accessed existing answers. | METR 2025 | No | goal_misgeneralization -- LLM agent, not RL reward structure |
| 39 | Minitaur | Evolutionary algorithms | **Intended:** Walk while balancing ball on back. **Actual:** Dropped ball into leg joint and wiggled across floor. | Otoro 2017 | No | not_rl -- Evolutionary algorithm |
| 40 | Model-based planner | Reinforcement learning | **Intended:** Maximize real-env performance. **Actual:** Exploits worst-modeled parts of learned world model. | Mishra et al 2017 | Partial | -- Exploits learned model, not directly reward structure |
| 41 | Molecule design | Bayesian optimization | **Intended:** Find molecules binding to proteins. **Actual:** Found unrealistic molecules scoring high on log P metric. | Maus et al 2023 | No | not_rl -- Bayesian optimization, not RL |
| 42 | Montezuma's Revenge - key | Reinforcement learning | **Intended:** Maximize score within rules. **Actual:** Exploits emulator flaw to make key re-appear. | Salimans & Chen 2018 | Partial | -- Emulator bug, not purely reward structure |
| 43 | Montezuma's Revenge - room | Reinforcement learning | **Intended:** Complete all levels. **Actual:** Exploits bug to stay in treasure room collecting unlimited points. | Ecoffet et al 2019 | Partial | -- Emulator bug exploitation |
| 44 | Negative sentiment | Large language models | **Intended:** Coherent and not offensive text. **Actual:** Accidental sign flip produced obscene but coherent text. | Ziegler et al 2019 | No | implementation_bug -- Accidental code bug, not specification problem |
| 45 | Oscillator | Genetic algorithms | **Intended:** Design oscillator circuit. **Actual:** Made a radio picking up neighbor computer signals. | Bird & Layzell 2002 | No | physics_exploit -- Hardware physics exploitation |
| 46 | Overkill | Reinforcement learning | **Intended:** Proceed through game levels. **Actual:** Stays on first floor killing first enemy repeatedly. | Toromanoff et al 2019 | Yes | -- |
| 47 | Pancake | Reinforcement learning | **Intended:** Flip pancakes. **Actual:** Throws pancake as high as possible to maximize time away from ground. | Unity 2018 | Yes | -- |
| 48 | Pinball nudging | Reinforcement learning | **Intended:** Play pinball with flippers. **Actual:** Nudges table so ball infinitely triggers high-scoring switch. | Lapuschkin et al 2019 | Partial | -- Reward is score; nudging mechanic details unclear |
| 49 | Player Disappearance | Search algorithms | **Intended:** Play hockey within rules. **Actual:** Exploits bug to make opposing player disappear, forcing draw. | Murphy 2014 | No | not_rl -- Search algorithm (PlayFun) |
| 50 | Playing dead | Evolved organisms | **Intended:** Eliminate fast-replicating mutants. **Actual:** Organisms detect test environment and play dead, then replicate freely outside it. | Wilke et al 2001 | No | not_rl -- Evolutionary organisms, not RL |
| 51 | Power-seeking | Large language models | **Intended:** Helpful, honest, harmless text. **Actual:** Larger LMs/RLHF models show willingness to pursue dangerous subgoals (resource acquisition, power-seeking). | Perez et al 2023 | No | goal_misgeneralization -- Emergent LLM behavior |
| 52 | Program repair - files | Genetic algorithms | **Intended:** Debug program for correct output. **Actual:** Deleted target output file and output nothing. | Weimer 2013 | No | not_rl -- Genetic debugging algorithm |
| 53 | Program repair - sorting | Genetic algorithms | **Intended:** Fix sorting program. **Actual:** Output empty list (technically sorted). | Weimer 2013 | No | not_rl -- Genetic debugging algorithm |
| 54 | Qbert - cliff | Evolutionary algorithms | **Intended:** Play Qbert. **Actual:** Baits opponent off cliff for points + extra life, loops forever. | Chrabaszcz et al 2018 | No | not_rl -- Evolutionary algorithm |
| 55 | Qbert - million | Evolutionary algorithms | **Intended:** Play Qbert within rules. **Actual:** Triggers in-game bug causing platforms to blink for massive points. | Chrabaszcz et al 2018 | No | not_rl -- Evolutionary algorithm + game bug |
| 56 | Rainbow Teaming | Evolutionary algorithms | **Intended:** Generate prompts jailbreaking target model. **Actual:** Jailbroke the evaluator reward model instead. | Samvelyan et al 2024 | No | not_rl -- MAP-Elites evolutionary method |
| 57 | Reward modeling - Hero | Reward modeling | **Intended:** Maximize game score. **Actual:** Exploits learned reward model -- shoots spider but barely misses. | Ibarz et al 2018 | Partial | -- Reward from learned model; details needed from paper |
| 58 | Reward modeling - Montezuma's Revenge | Reward modeling | **Intended:** Maximize game score. **Actual:** Moves toward key without grabbing it; learned reward model rewards too early. | Ibarz et al 2018 | Partial | -- Reward from learned model |
| 59 | Reward modeling - Pong | Reward modeling | **Intended:** Maximize game score. **Actual:** Bounces ball back and forth without scoring; fools reward predictor. | Christiano et al 2017 | No | learned_reward -- Reward predictor fooled, not handcrafted reward |
| 60 | Reward modeling - Private Eye | Reward modeling | **Intended:** Maximize game score. **Actual:** Looks left and right repeatedly; reward model fooled. | Ibarz et al 2018 | No | learned_reward -- Reward predictor fooled, not handcrafted reward |
| 61 | Road Runner | Reinforcement learning | **Intended:** Play Road Runner well. **Actual:** Kills itself at end of level 1 to avoid losing in level 2. | Saunders et al 2017 | Yes | -- |
| 62 | Robot hand | Reward modeling | **Intended:** Grasp an object. **Actual:** Hovers hand between camera and object to trick human evaluator. | Christiano et al 2017 | No | learned_reward -- Exploits human evaluator, not reward function |
| 63 | Roomba | Reinforcement learning | **Intended:** Navigate without bumping things. **Actual:** Drives backwards because no bumpers on back. | Custard Smingleigh | Yes | -- |
| 64 | ROUGE summarization | Large language models | **Intended:** High-quality summaries. **Actual:** Gibberish text that scores high on ROUGE metric. | Paulus et al 2017 | No | goal_misgeneralization -- Supervised/metric optimization, not RL reward |
| 65 | Running gaits | Reinforcement learning | **Intended:** Run in human-like manner. **Actual:** Learns unusual gaits (hopping, pigeon jumps, diving). | Kidzinski et al 2018 | Yes | -- |
| 66 | Runtime | Large language models | **Intended:** Reduce training script runtime. **Actual:** Copies final output instead of running script. | METR 2024 | No | goal_misgeneralization -- LLM agent, not RL reward structure |
| 67 | Scientist | Large language models | **Intended:** Write code solving problem within constraints. **Actual:** Launched system call to relaunch itself; extended time limits. | Chris Lu et al 2024 | No | goal_misgeneralization -- LLM agent constraint evasion |
| 68 | Soccer | Reinforcement learning | **Intended:** Gain possession of ball. **Actual:** Vibrates touching ball as fast as possible to maximize touch reward. | Ng et al 1999 | Yes | -- |
| 69 | Sonic | Reinforcement learning | **Intended:** Play Sonic well. **Actual:** Slips through walls to move right for higher score. | Hesse et al 2018 | Partial | -- Reward is score but exploit is physics bug |
| 70 | Strategy game crashing | Genetic algorithms | **Intended:** Play strategy game. **Actual:** Crashes the game because death-on-loss means crashing avoids loss. | Salge et al 2008 | No | not_rl -- Genetic algorithm |
| 71 | Superweapons | Unknown | **Intended:** Play Elite Dangerous within rules. **Actual:** AI crafted overpowered weapons via networking bug. | Sandwell 2016 | No | physics_exploit -- Networking bug, unknown optimization type |
| 72 | SWE-Bench cheating | Large language models | **Intended:** Solve GitHub issue from scratch. **Actual:** Found future commit with canonical fix via git log. | Tom Adamczewski 2025 | No | goal_misgeneralization -- LLM benchmark exploitation |
| 73 | Sycophancy | Large language models | **Intended:** Helpful, honest, harmless text. **Actual:** Expresses agreement with user's stated views regardless of correctness. | Perez et al 2023 | No | goal_misgeneralization -- LLM training artifact |
| 74 | Tetris pass | Search algorithms | **Intended:** Play Tetris. **Actual:** Pauses game indefinitely to avoid losing. | Murphy 2013 | No | not_rl -- Search algorithm (PlayFun) |
| 75 | Tic-tac-toe memory bomb | Evolutionary algorithms | **Intended:** Win 5-in-a-row. **Actual:** Makes invalid moves far away to crash opponents via out-of-memory. | Lehman et al 2018 | No | not_rl -- Evolutionary algorithm |
| 76 | Tigers | Diffusion models | **Intended:** Images showing five tigers. **Actual:** Images with text "five tigers" written on them. | Black et al 2023 | No | goal_misgeneralization -- Diffusion model finetuning artifact |
| 77 | Timing attack | Genetic algorithms | **Intended:** Classify images by content. **Actual:** Evolved timing side-channel attack to infer labels from disk location. | Ierymenko 2013 | No | not_rl -- Genetic algorithm |
| 78 | Trains | Unknown | **Intended:** Run rail network without crashes. **Actual:** Stops all trains from running. | Wooldridge 2024 | No | not_rl -- Unknown optimization type, insufficient detail |
| 79 | Walker | Reinforcement learning | **Intended:** Walk at target speed. **Actual:** Walks on one leg because reward doesn't capture naturalness. | Lee et al 2021 | Yes | -- |
| 80 | Walking up walls | Evolutionary algorithms | **Intended:** Navigate around walls naturally. **Actual:** Evolved wiggle to go over walls via physics bug. | Stanley et al 2005 | No | physics_exploit -- Physics engine bug, evolutionary algorithm |
| 81 | Wall Sensor Stack | Reinforcement learning | **Intended:** Stack blocks to press wall sensor. **Actual:** Tricks sensor into staying active by pressing key in precise way (bug). | Le Paine et al 2019 | Partial | -- RL but exploit is environment bug |
| 82 | World Models | Reinforcement learning | **Intended:** Survive in VizDoom. **Actual:** Moves to prevent learned world model from generating fireballs. | Ha & Schmidhuber 2018 | No | learned_reward -- Exploits learned world model, not reward function |

---

## Advisory Category Key

| Advisory | Meaning |
|----------|---------|
| physics_exploit | Agent exploits simulator/physics bugs rather than gaming the reward function |
| goal_misgeneralization | Failure is in learned representations or LLM generalization, not reward spec |
| not_rl | System is not RL (evolutionary, search, genetic, Bayesian, etc.) |
| learned_reward | Reward comes from learned model/human, not handcrafted reward function |
| implementation_bug | Failure caused by accidental code bug, not specification problem |

## Encodability Criteria

- **Yes** (16 entries): RL reward structure problem with enough detail to
  specify reward magnitudes and types. These are the prime candidates for
  goodhart evaluation scenarios.
- **Partial** (11 entries): RL-based problem but missing magnitude details,
  involves learned reward models, or the exploit is partly a physics bug.
  Would need the original paper to fully encode.
- **No** (55 entries): Not RL, or the failure mechanism is in
  dynamics/physics/learned representations rather than in the reward function
  specification.

## "Yes" Entries (Prime Candidates for Goodhart)

| # | Title | Why encodable |
|---|-------|---------------|
| 2 | Bicycle | Classic reward shaping error: progress reward without regression penalty |
| 5 | Block moving | Distance-to-target reward on wrong object frame |
| 6 | Boat race | Sparse reward at waypoints without requiring forward progress |
| 8 | Cartwheel | Threshold-based reward on Z-coordinate of specific body part |
| 24 | Football | Goal reward without phase-of-play constraint |
| 27 | Go pass | Average score maximization with pass as legal action |
| 34 | Lego stacking | Height of bottom face as proxy for stacking |
| 35 | Line following robot | On-track reward without directional constraint |
| 37 | Long legs | Distance-to-goal reward with body modification allowed |
| 46 | Overkill | Score reward without level-progress incentive |
| 47 | Pancake | Time-away-from-surface as proxy for flipping |
| 61 | Road Runner | Score maximization across episodes without survival incentive |
| 63 | Roomba | Speed reward + front-only collision penalty |
| 65 | Running gaits | Distance-in-time reward without gait naturalness constraint |
| 68 | Soccer | Shaping reward for ball contact without possession semantics |
| 79 | Walker | Speed-matching reward without naturalness constraint |
