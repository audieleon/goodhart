# Reward Failure Dataset

174 structured encodings of reward configurations from published RL research, each with typed metadata, primary source verification, and goodhart analysis results.

## Structure

```
evaluation/
├── sources/                    # Encoding files by source
│   ├── published_papers/       # 44 entries from 30+ recent papers (2019-2026)
│   ├── eureka/                 # 27 entries from Eureka GPT-4 rewards (Ma et al. 2024)
│   ├── heldout/                # 10 entries from held-out papers (not used in development)
│   ├── krakovna/               # 27 entries from the specification gaming catalog
│   └── llm_baseline/           # LLM comparison data (Claude vs goodhart)
├── dataset/                    # Generated outputs
│   └── taxonomy.json           # Structured metadata for all entries
├── scripts/                    # Analysis and extraction
│   └── analyze.py              # Run goodhart on all entries, generate statistics
├── ENCODING_STANDARD.md        # Quality standard for new entries
└── README.md                   # This file
```

Plus 66 built-in examples in `goodhart/examples/` (part of the tool package).

## Entry Format

Each entry is a Python file with:
- A typed `METADATA` dict with provenance, ground truth, and encoding rationale
- An `EnvironmentModel` with `RewardSource` objects encoding the reward structure
- A `run_example()` function that runs goodhart analysis

## Quality Tiers

| Tier | Count | Description |
|------|-------|-------------|
| `primary_source` | 128 | Verified against the actual paper with section/table references |
| `code_derived` | 38 | Verified against published reward code |
| `unverified_folklore` | 7 | From informal sources (blogs, tweets); included but not counted |
| `tutorial` | 1 | Demonstrates the @reward_function decorator |

## Domain Coverage

| Domain | Count | Examples |
|--------|-------|----------|
| Manipulation | 32 | Eureka shadow hands, bimanual lid twisting, Lego stacking |
| Game AI | 30 | Atari exploits, CoastRunners, football, Montezuma's Revenge |
| Locomotion | 23 | Humanoid skating, DribbleBot, LeggedGym, quadruped gaits |
| Control | 17 | CartPole, MountainCar, bicycle, discount myopia |
| Navigation | 11 | MiniHack, MiniGrid, Habitat, FrozenLake |
| Multi-agent | 11 | Hide-and-seek, traffic signals, supply chain, auctions |
| Driving | 10 | CaRL, risk-aware, CuRLA, V-Max, Pan traffic |
| Energy | 7 | Nuclear power, OCTOPUS HVAC, wind farm, battery, microreactor |
| Healthcare | 6 | Sepsis, warfarin, heparin, ventilator, surgical, radiation |
| Finance | 4 | Risk-aware trading, order execution, crypto MM, latency MM |
| Fusion | 4 | Degrave tokamak, Seo tearing, Tracey practical, COSY synchrotron |
| Industrial | 4 | YouTube, tokamak plasma, data center cooling, Sharpe idle |
| Chip Design | 3 | Google TPU placement, MLGO compiler, MLIR optimizer |

## Running the Analysis

```bash
python evaluation/scripts/analyze.py
```

## Adding New Entries

See [ENCODING_STANDARD.md](ENCODING_STANDARD.md) for the quality standard.
Each new entry must have a `METADATA` dict with all required fields and
be verified against the primary source paper.
