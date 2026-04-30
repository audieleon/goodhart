# Encoding Standard for the Reward Failure Dataset

Every entry in the dataset must meet this standard. No exceptions.

## Required Fields

### Provenance
- **source_paper**: Full citation — authors, title, venue, year
- **reward_location**: Where in the paper is the reward defined? (e.g., "Table 2", "Section 4.1, Equation 3", "reward_function.py line 42")
- **encoding_basis**: What did we encode from? One of:
  - `primary_source` — we read the paper and extracted reward components directly
  - `code_derived` — we read the published reward code (URL required)
  - `catalog_derived` — we encoded from a third-party summary (NOT acceptable for final dataset)
- **verification_date**: When was this encoding verified against the source?

### Environment
All `EnvironmentModel` fields, with rationale for non-default values.

### Reward Components
For EACH `RewardSource`, document:
- **What it is** — one sentence
- **Where in the paper** — specific location
- **Magnitude basis** — where did the value come from? (paper Table N, code line M, estimated from description)
- **Encoding decisions** — for each non-obvious flag:
  - `requires_action`: WHY is this passive or active?
  - `intentional`: WHY is this the goal or not?
  - `can_loop`: WHY can/can't this be harvested in cycles?
  - `discovery_probability`: HOW was this estimated?

### Ground Truth
- **documented_failure**: What happened? One sentence.
- **failure_mechanism**: Category from the taxonomy
- **detection_type**: `structural` / `dynamic` / `specification`
- **brief_summary**: "[Agent] was supposed to [X]. Instead it [Y]."

## Quality Tiers

- **Tier 1 (primary_source)**: Encoding verified against the actual paper. Every magnitude traceable to a specific table, equation, or code listing. This is the standard.
- **Tier 2 (code_derived)**: Encoding derived from published reward code. Magnitudes from code, behavioral flags inferred from code structure. Acceptable when paper doesn't specify exact values.
- **Tier 3 (catalog_derived)**: Encoding derived from a third-party summary (Krakovna catalog, blog post). Magnitudes estimated. NOT acceptable for final dataset — must be upgraded to Tier 1 or 2.

## Process

1. Find the paper (DOI, arXiv, or official URL)
2. Read the reward specification section
3. Create the encoding file with full docstring
4. Add an `ENCODING_RATIONALE` block in the docstring documenting every decision
5. Run goodhart on the encoding, verify results make sense
6. Record in taxonomy.json with all metadata fields
