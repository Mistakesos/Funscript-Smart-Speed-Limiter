# Funscript Speed Limiter v3.2

## Overview
A speed limiter for funscripts that preserves amplitude, rhythm, and total duration. Instead of reducing stroke amplitude, it detects periodic blocks, reduces repetition count, and stretches the remaining cycles to fill the original block duration. Non-periodic overspeed gaps are re-timed, followed by a global speed correction pass.

## Usage
`python limit_speed_deepseek.py <input.funscript> <output.funscript>`

## Key Features
- Preserves position amplitude whenever possible.
- Preserves rhythm and total script duration.
- Detects periodic blocks with period sizes `P=2..30`.
- Reduces repeat count `K` and stretches cycles to lower speed.
- Handles short non-periodic overspeed gaps up to `MAX_GAP_PTS`.
- Applies global overspeed correction only above `MAX_SPEED + SPEED_TOL`.
- Iterates global correction up to `MAX_ITER` times.
- Cleans timestamps, removes duplicates, and enforces strictly increasing time.
- Restores original first/last timestamps and positions.
- Provides optional debug validation for outliers and overspeed segments.

## How It Works
1. **Load and preprocess**  
   Sort actions by `at` and `pos`, drop near-duplicate points, and normalize same-timestamp conflicts.

2. **Detect periodic blocks**  
   Scan for repeated cycles where positions match within `POS_TOL` and relative timing matches within `TIME_RATIO_TOL`. Extend each match to count total repeats `K`. `P=2` requires more repeats to avoid false positives.

3. **Limit periodic blocks**  
   For each detected block, try reducing `K` to `K_new`. Rescale the first `K_new` cycles uniformly into the original block duration `T_block`. This lowers speed without changing stroke amplitude. If even one cycle cannot fit, keep the original block.

4. **Handle non-periodic gaps**  
   For uncovered overspeed segments with at most `MAX_GAP_PTS` points, compute minimum required time per segment using `MAX_SPEED`, then distribute remaining slack proportionally to original segment durations.

5. **Global speed correction**  
   After merging all segments, correct only speeds above `MAX_SPEED + SPEED_TOL`. Push later timestamps forward as needed, then rescale total duration back to the original duration. Repeat up to `MAX_ITER` times.

6. **Final cleanup and output**  
   Sort, deduplicate, enforce increasing timestamps, restore original first/last points, and write compact JSON.

## Parameters
| Parameter | Default | Purpose |
|---|---:|---|
| `MAX_SPEED` | `600` | Maximum speed in `pos/s` |
| `SPEED_TOL` | `10` | Global fix tolerance; only correct speeds above `MAX_SPEED + SPEED_TOL` |
| `TIME_RATIO_TOL` | `0.15` | Relative timing tolerance for period matching |
| `MIN_PERIOD_SEGS` | `2` | Minimum period segment count |
| `MAX_PERIOD_SEGS` | `30` | Maximum period segment count |
| `MIN_REPEATS` | `3` | Minimum repeats for normal periods |
| `MIN_REPEATS_P2` | `5` | Minimum repeats for `P=2` periods |
| `POS_TOL` | `0.5` | Position tolerance for period matching |
| `MAX_GAP_PTS` | `15` | Maximum gap size eligible for re-timing |
| `MAX_ITER` | `8` | Maximum global correction iterations |
| `DEBUG` | `False` | Enable verbose validation output |

## Notes
- The script cannot create time. If the original duration is insufficient, some overspeed may remain.
- Speeds up to `MAX_SPEED + SPEED_TOL` are treated as acceptable to avoid unnecessary micro-adjustments.
- Best suited for regular repetitive scripts, but designed to safely handle irregular segments.
- Output preserves the original first/last timestamps and positions.

## Comparison of Results

**Before**  
<img width="1287" height="185" alt="before" src="https://github.com/user-attachments/assets/dc46ff8c-4fcf-4bda-9853-9506ef67c5d6" />

**After**  
<img width="1296" height="192" alt="after" src="https://github.com/user-attachments/assets/a01ab302-a0f9-41b5-bbe9-aaa0d9de664a" />

**Compare**
<img width="1102" height="371" alt="compare" src="https://github.com/user-attachments/assets/457b773d-aa4d-46a0-9b2e-cf9f035628d5" />
