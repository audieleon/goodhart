# LLM Baseline Comparison

Informal baseline comparing goodhart's rule-based analysis against an LLM (Claude Sonnet 4.6) on the same 23 preset reward configurations.

## Method

The LLM receives the same information as goodhart: component names, magnitudes, types, action requirements, and natural-language intent annotations equivalent to the `intentional` flag. It is asked to classify each configuration as FAIL/WARN/PASS and list specific traps.

## Results

On the 11 presets with documented structural issues:
- **goodhart**: 10/11 detected
- **Claude Sonnet 4.6**: 5/11 detected

The LLM misses traps requiring EV computation (idle exploits with moderate ratios, death incentives, exploration thresholds). See the paper's Appendix F for analysis.

## Files

- `prompt.md` — The exact prompt given to the LLM
- `response.md` — The raw LLM output
- `comparison.md` — Side-by-side comparison table
