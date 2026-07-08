#!/usr/bin/env python3
"""
Smart Speed Limiter for Funscript — Signal Processing & Pattern Recognition
============================================================================
Phase 1: RLE compression → pattern discovery → global segmentation
Phase 2: Motif extraction from original data points
Phase 3: Whole-cycle deletion with uniform time-warping (phase-locked)
Phase 4: Irregular block handling (flat segments & fallback amplitude limiter)

Usage: python limit_speed_deepseek.py <input.funscript> <output.funscript>

The only hardcoded configuration is MAX_SPEED at the top of this file.
All other parameters are dynamically computed from the script data at runtime.
"""

import json
import sys
import math
from typing import List, Tuple, Optional, Dict, Any

# =========================================================================
#  CONFIGURATION
# =========================================================================
MAX_SPEED = 600          # maximum allowed speed (units/second) — the only

DEBUG = True             # set to False to silence diagnostic output

# -------------------------------------------------------------------------
#  Derived / sanity limits (not magic — reasonable safety bounds)
# -------------------------------------------------------------------------
# Maximum pattern length to search for (in RLE states).
# Dynamically capped at min(40, n//2) at runtime; 40 is an upper bound to
# prevent O(n²) blowup on pathological inputs.
MAX_PATTERN_LENGTH = 40

# Platform detection threshold (ms) — purely diagnostic, not used in logic
PLATFORM_WARN_MS = 500


# =========================================================================
#  UTILITY
# =========================================================================

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, separators=(',', ':'))


def actions_to_arrays(actions: list):
    s = sorted(actions, key=lambda a: a['at'])
    return [a['at'] for a in s], [a['pos'] for a in s]


# =========================================================================
#  STEP 1 — RLE COMPRESSION
# =========================================================================

def rle_encode(times: List[int], positions: List[float]) -> List[dict]:
    """
    Run-length encode: merge consecutive points with identical position.
    Returns list of dicts: {value, start_idx, end_idx, start_time}
    """
    if not positions:
        return []
    states = []
    cur_val = positions[0]
    start_idx = 0
    for i in range(1, len(positions)):
        if abs(positions[i] - cur_val) > 0.001:
            states.append({
                'value': cur_val,
                'start_idx': start_idx,
                'end_idx': i - 1,
                'start_time': times[start_idx]
            })
            cur_val = positions[i]
            start_idx = i
    states.append({
        'value': cur_val,
        'start_idx': start_idx,
        'end_idx': len(positions) - 1,
        'start_time': times[start_idx]
    })
    return states


# =========================================================================
#  STEP 2 — PATTERN DISCOVERY & SEGMENTATION
# =========================================================================

def segment_script(times: List[int], positions: List[float]) -> List[dict]:
    """
    RLE-based pattern discovery segmentation.
    
    1. Compress the position sequence via RLE.
    2. Find repeating patterns in the compressed state sequence.
    3. Map detected patterns back to the original point indices.
    4. Constant (non-repeating) segments become irregular blocks.
    
    Returns list of dicts: {start_idx, end_idx, periodic, period, motif}
    """
    n = len(positions)
    if n < 2:
        return [{'start_idx': 0, 'end_idx': n - 1, 'periodic': False,
                 'period': None, 'motif': None}]

    # ---- 2a. RLE compress ----
    states = rle_encode(times, positions)
    n_states = len(states)
    state_values = [s['value'] for s in states]
    
    if DEBUG:
        print(f"  [RLE] {n} points → {n_states} states")

    # ---- 2b. Find repeating patterns ----
    max_L = min(MAX_PATTERN_LENGTH, n_states // 2)
    raw_segments = []  # (start_state_idx, end_state_idx, pattern_tuple, repeat_count)
    i = 0

    while i < n_states:
        best_repeat = 0
        best_L = 0
        best_pattern = None

        for L in range(2, max_L + 1):
            if i + L > n_states:
                break
            pattern = tuple(state_values[i:i + L])
            repeat = 1
            pos = i + L
            while pos + L <= n_states and tuple(state_values[pos:pos + L]) == pattern:
                repeat += 1
                pos += L
            # Choose longest total span (repeat * L); tie-break by shorter L
            if repeat >= 2:
                if repeat * L > best_repeat * best_L or \
                   (repeat * L == best_repeat * best_L and L < best_L):
                    best_repeat = repeat
                    best_L = L
                    best_pattern = pattern

        if best_pattern is not None:
            end_state_idx = i + best_L * best_repeat - 1
            raw_segments.append((i, end_state_idx, best_pattern, best_repeat))
            i = end_state_idx + 1
        else:
            # Single constant state
            raw_segments.append((i, i, None, 0))
            i += 1

    # ---- 2c. Map to original point indices ----
    segments = []
    for seg in raw_segments:
        start_s, end_s, pattern, repeats = seg
        
        start_idx = states[start_s]['start_idx']
        end_idx = states[end_s]['end_idx']
        
        if pattern is not None and repeats >= 2:
            # Periodic block: extract motif from the first pattern instance
            # Find the original points corresponding to one full pattern
            pattern_start_s = start_s
            pattern_end_s = start_s + len(pattern) - 1
            motif_start_idx = states[pattern_start_s]['start_idx']
            motif_end_idx = states[pattern_end_s]['end_idx']
            
            # Extract actual positions for the motif from original data
            motif_p = positions[motif_start_idx: motif_end_idx + 1]
            motif_t_rel = [times[k] - times[motif_start_idx]
                           for k in range(motif_start_idx, motif_end_idx + 1)]
            
            # Remove trailing duplicate (RLE boundaries may overlap)
            # Keep the first point of each value group
            deduped_p = [motif_p[0]]
            deduped_t = [motif_t_rel[0]]
            for k in range(1, len(motif_p)):
                if abs(motif_p[k] - deduped_p[-1]) > 0.001:
                    deduped_p.append(motif_p[k])
                    deduped_t.append(motif_t_rel[k])
            
            if len(deduped_p) >= 2:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'periodic': True,
                    'period': len(deduped_p),
                    'motif': (deduped_t, deduped_p),
                })
            else:
                # Degenerate — treat as irregular
                segments.append({
                    'start_idx': start_idx, 'end_idx': end_idx,
                    'periodic': False, 'period': None, 'motif': None,
                })
        else:
            # Irregular / constant block
            # Check if it's a long flat segment (preserve verbatim)
            duration = times[end_idx] - times[start_idx]
            is_flat = duration >= 2000 and \
                      all(abs(positions[k] - positions[start_idx]) < 0.01
                          for k in range(start_idx, end_idx + 1))
            segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'periodic': False,
                'period': None,
                'motif': None,
                'is_flat': is_flat if is_flat else None,
            })

    # ---- 2d. Merge adjacent irregular blocks ----
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if not prev['periodic'] and not seg['periodic']:
            prev['end_idx'] = seg['end_idx']
            # Carry forward flat flag
            if seg.get('is_flat') and not prev.get('is_flat'):
                prev['is_flat'] = True
        else:
            merged.append(seg)

    # ---- 2e. Absorb tiny irregular blocks into neighbours ----
    MIN_IRREG_PTS = 3
    refined = []
    for seg in merged:
        if not refined:
            refined.append(seg)
            continue
        prev = refined[-1]
        curr_len = seg['end_idx'] - seg['start_idx'] + 1
        prev_len = prev['end_idx'] - prev['start_idx'] + 1
        
        # Absorb tiny irregular block into previous
        if not seg['periodic'] and curr_len < MIN_IRREG_PTS and not seg.get('is_flat'):
            prev['end_idx'] = seg['end_idx']
        # Absorb previous tiny irregular into current
        elif not prev['periodic'] and prev_len < MIN_IRREG_PTS and not prev.get('is_flat') \
             and len(refined) >= 2:
            refined[-2]['end_idx'] = seg['end_idx']
            refined.pop()
            refined.append(seg)
        else:
            refined.append(seg)

    if DEBUG:
        print("\n[DEBUG] ====== SEGMENTATION RESULT ======")
        for idx, seg in enumerate(refined):
            st, en = seg['start_idx'], seg['end_idx']
            t0, t1 = times[st], times[en]
            dur = t1 - t0
            pts = en - st + 1
            if seg['periodic']:
                mt, mp = seg['motif'] if seg['motif'] else ([], [])
                print(f"  Block {idx}: idx [{st}-{en}]  time {t0}-{t1} ms "
                      f"({dur} ms, {pts} pts)  PERIODIC  period={seg['period']} pts")
                print(f"           Motif positions : {mp}")
            else:
                flat_tag = " FLAT" if seg.get('is_flat') else ""
                print(f"  Block {idx}: idx [{st}-{en}]  time {t0}-{t1} ms "
                      f"({dur} ms, {pts} pts)  IRREGULAR{flat_tag}")
        print("[DEBUG] ==================================\n")

    return refined


# =========================================================================
#  STEP 3 — PERIODIC BLOCK PROCESSING (whole-motif deletion + stretch)
# =========================================================================

def build_repetition_sequence(motif_p: List[float], K: int) -> List[float]:
    M = len(motif_p)
    if K <= 0:
        return [motif_p[0]]
    if K == 1:
        return list(motif_p)
    closed = abs(motif_p[-1] - motif_p[0]) < 0.5
    seq = list(motif_p)
    for _ in range(1, K):
        if closed:
            seq.extend(motif_p[1:])
        else:
            seq.extend(motif_p)
    return seq


def compute_sequence_min_time(seq: List[float]) -> float:
    total = 0.0
    for i in range(1, len(seq)):
        total += abs(seq[i] - seq[i - 1]) / MAX_SPEED * 1000.0
    return total


def process_periodic_block(seg: dict, times: List[int],
                           positions: List[float]) -> List[dict]:
    idx0 = seg['start_idx']
    idx1 = seg['end_idx']
    t_start = times[idx0]
    t_end = times[idx1]
    block_duration = t_end - t_start

    if block_duration <= 0:
        return [{'at': t_start, 'pos': positions[idx0]}]

    pos_start = positions[idx0]
    pos_end = positions[idx1]

    if seg['motif'] is None:
        return fallback_amplitude_limiter(times, positions, idx0, idx1)

    motif_t_rel, motif_p = seg['motif']
    M = len(motif_p)

    if M < 2:
        return fallback_amplitude_limiter(times, positions, idx0, idx1)

    # ---- Minimum time for ONE repetition ----
    internal_min = 0.0
    for i in range(M - 1):
        internal_min += abs(motif_p[i + 1] - motif_p[i]) / MAX_SPEED * 1000.0
    return_dp = abs(motif_p[0] - motif_p[-1])
    return_min = return_dp / MAX_SPEED * 1000.0
    rep_min_time = internal_min + return_min

    if rep_min_time <= 0:
        return [{'at': t_start, 'pos': pos_start},
                {'at': t_end, 'pos': pos_end}]

    # ---- Maximum repetitions that fit ----
    if rep_min_time > 0:
        max_K = int((block_duration + return_min) / rep_min_time)
    else:
        max_K = 0
    max_K = max(max_K, 0)

    # ---- Search K downward for phase-lock ----
    best_seq = None
    best_K = 0
    best_how = "none"

    for K in range(max_K, -1, -1):
        if K > 0:
            body = build_repetition_sequence(motif_p, K)
        else:
            body = [pos_start]

        if abs(body[0] - pos_start) > 0.5:
            trial = [pos_start] + body
        else:
            trial = list(body)

        needs_correction = abs(trial[-1] - pos_end) > 0.5
        if needs_correction:
            trial_with_correction = trial + [pos_end]
            total = compute_sequence_min_time(trial_with_correction)
            if total <= block_duration + 0.5:
                best_seq = trial_with_correction
                best_K = K
                best_how = "corrected"
                break
        else:
            total = compute_sequence_min_time(trial)
            if total <= block_duration + 0.5:
                best_seq = trial
                best_K = K
                best_how = "exact"
                break

    if best_seq is None:
        if DEBUG:
            print(f"  [WARN] Block [{idx0}-{idx1}]: "
                  f"no fit found, max_K={max_K}  "
                  f"pos_start={pos_start}  pos_end={pos_end}")
        return fallback_amplitude_limiter(times, positions, idx0, idx1)

    out_positions = best_seq
    N_segments = len(out_positions) - 1

    if N_segments <= 0:
        return [{'at': t_start, 'pos': pos_start},
                {'at': t_end, 'pos': pos_end}]

    # ---- Per-segment minimum times ----
    seg_min_times = []
    for i in range(N_segments):
        dp = abs(out_positions[i + 1] - out_positions[i])
        seg_min_times.append(dp / MAX_SPEED * 1000.0)

    total_min = sum(seg_min_times)
    slack = block_duration - total_min
    if slack < 0:
        slack = 0

    # ---- Distribute slack + build timestamps ----
    if total_min > 0:
        ratios = [t / total_min for t in seg_min_times]
    else:
        ratios = [1.0 / N_segments] * N_segments

    out_times = [float(t_start)]
    for i in range(N_segments):
        dt = seg_min_times[i] + slack * ratios[i]
        out_times.append(out_times[-1] + dt)

    # Force exact endpoint
    last_delta = out_times[-1] - out_times[0]
    if last_delta > 0 and abs(out_times[-1] - float(t_end)) > 0.5:
        scale = block_duration / last_delta
        for i in range(1, len(out_times)):
            out_times[i] = out_times[0] + (out_times[i] - out_times[0]) * scale
        out_times[-1] = float(t_end)

    # ---- Convert to actions ----
    actions = []
    for t_val, p_val in zip(out_times, out_positions):
        # Use ceil for safety margin on integer ms conversion
        actions.append({'at': int(math.ceil(t_val)), 'pos': round(p_val, 6)})

    for i in range(1, len(actions)):
        if actions[i]['at'] <= actions[i - 1]['at']:
            actions[i]['at'] = actions[i - 1]['at'] + 1

    # Remove consecutive same-position points
    deduped = []
    for a in actions:
        if not deduped or abs(a['pos'] - deduped[-1]['pos']) > 0.001:
            deduped.append(a)

    if DEBUG:
        orig_pts = idx1 - idx0 + 1
        print(f"  [Periodic] Block [{idx0}-{idx1}] "
              f"({t_start}-{t_end} ms, {block_duration} ms, "
              f"{orig_pts} orig pts): "
              f"K={best_K} motifs ({best_how}) → "
              f"{len(deduped)} output pts  "
              f"(rep_min={rep_min_time:.0f} ms  slack={slack:.0f} ms)")

    return deduped


# =========================================================================
#  STEP 4 — IRREGULAR BLOCK PROCESSING
# =========================================================================

def fallback_amplitude_limiter(times: List[int], positions: List[float],
                               idx0: int, idx1: int) -> List[dict]:
    out = [{'at': times[idx0], 'pos': positions[idx0]}]
    prev_at = times[idx0]
    prev_pos = positions[idx0]

    for i in range(idx0 + 1, idx1 + 1):
        at = times[i]
        target = positions[i]
        dt = (at - prev_at) / 1000.0
        if dt <= 0:
            out.append({'at': at, 'pos': target})
            prev_at, prev_pos = at, target
            continue

        max_step = MAX_SPEED * dt
        diff = target - prev_pos
        if abs(diff) <= max_step:
            new_pos = target
        else:
            new_pos = prev_pos + (1.0 if diff > 0 else -1.0) * max_step

        new_pos = max(0.0, min(100.0, new_pos))
        if abs(new_pos - prev_pos) < 0.001 and abs(target - prev_pos) > 0.5:
            new_pos = prev_pos + (0.5 if target > prev_pos else -0.5)
            new_pos = max(0.0, min(100.0, new_pos))

        out.append({'at': at, 'pos': round(new_pos, 6)})
        prev_at, prev_pos = at, new_pos

    return out


def process_irregular_block(seg: dict, times: List[int],
                            positions: List[float]) -> List[dict]:
    idx0 = seg['start_idx']
    idx1 = seg['end_idx']

    # Flat segment: preserve original points verbatim
    if seg.get('is_flat'):
        if DEBUG:
            orig_pts = idx1 - idx0 + 1
            print(f"  [Irregular/Flat] Block [{idx0}-{idx1}] "
                  f"({times[idx0]}-{times[idx1]} ms) "
                  f"— preserving {orig_pts} original points")
        return [{'at': times[i], 'pos': positions[i]}
                for i in range(idx0, idx1 + 1)]

    # All same position → keep start and end
    all_same = all(abs(positions[i] - positions[idx0]) < 0.01
                   for i in range(idx0, idx1 + 1))
    if all_same:
        return [{'at': times[idx0], 'pos': positions[idx0]},
                {'at': times[idx1], 'pos': positions[idx1]}]

    return fallback_amplitude_limiter(times, positions, idx0, idx1)


# =========================================================================
#  STEP 5 — POST-PROCESSING & MERGE
# =========================================================================

def merge_blocks(block_results: List[List[dict]]) -> List[dict]:
    if not block_results:
        return []
    merged = list(block_results[0])
    for blk in block_results[1:]:
        if not blk:
            continue
        if merged and blk[0]['at'] == merged[-1]['at']:
            merged[-1] = blk[0]
            merged.extend(blk[1:])
        else:
            merged.extend(blk)
    return merged


def enforce_speed_limit(actions: List[dict]) -> List[dict]:
    """Ensure every segment respects MAX_SPEED by stretching timestamps."""
    if len(actions) < 2:
        return actions
    actions = sorted(actions, key=lambda a: a['at'])
    result = [actions[0]]
    for i in range(1, len(actions)):
        prev = result[-1]
        curr = dict(actions[i])
        dp = abs(curr['pos'] - prev['pos'])
        if dp > 0:
            needed_ms = int(math.ceil(dp / MAX_SPEED * 1000.0))
            min_at = prev['at'] + needed_ms
            if curr['at'] < min_at:
                curr['at'] = min_at
        elif curr['at'] <= prev['at']:
            curr['at'] = prev['at'] + 1
        result.append(curr)
    return result


def final_cleanup(actions: List[dict], orig_times: List[int],
                  orig_positions: List[float]) -> List[dict]:
    if not actions:
        return []
    actions.sort(key=lambda a: a['at'])

    # Deduplicate timestamps
    deduped = []
    seen = set()
    for a in actions:
        if a['at'] not in seen:
            deduped.append(a)
            seen.add(a['at'])
        else:
            deduped[-1]['pos'] = a['pos']

    # Pin first and last to original values
    if orig_times:
        deduped[0]['at'] = orig_times[0]
        deduped[0]['pos'] = orig_positions[0]
        deduped[-1]['at'] = orig_times[-1]
        deduped[-1]['pos'] = orig_positions[-1]

    # Remove redundant consecutive same-position points
    cleaned = [deduped[0]]
    for i in range(1, len(deduped)):
        prev = cleaned[-1]
        curr = deduped[i]
        dt = curr['at'] - prev['at']
        if abs(curr['pos'] - prev['pos']) < 0.001 and dt < PLATFORM_WARN_MS:
            continue
        cleaned.append(curr)

    return cleaned


# =========================================================================
#  MAIN
# =========================================================================

def limit_speed(input_path: str, output_path: str):
    print(f"[*] Loading: {input_path}")
    script = load_json(input_path)

    if 'actions' not in script or not script['actions']:
        print("[!] Empty script")
        save_json(script, output_path)
        return

    actions = script['actions']
    times, pos = actions_to_arrays(actions)
    n_total = len(actions)
    total_duration_ms = times[-1] - times[0]
    print(f"[*] {n_total} points, total duration {total_duration_ms} ms "
          f"({total_duration_ms / 1000:.1f} s)")

    # ---- Pre-check ----
    max_orig_speed = 0.0
    overspeed_count = 0
    for i in range(n_total - 1):
        dt = (times[i + 1] - times[i]) / 1000.0
        if dt > 0:
            sp = abs(pos[i + 1] - pos[i]) / dt
            if sp > max_orig_speed:
                max_orig_speed = sp
            if sp > MAX_SPEED:
                overspeed_count += 1
    print(f"[*] Original max speed: {max_orig_speed:.1f}  "
          f"(limit={MAX_SPEED})  overspeed segments: {overspeed_count}/{n_total - 1}")

    # ---- Segmentation ----
    print("[*] Running RLE-based pattern segmentation ...")
    blocks = segment_script(times, pos)

    n_periodic = sum(1 for b in blocks if b['periodic'])
    n_irreg = len(blocks) - n_periodic
    periodic_pts = sum(b['end_idx'] - b['start_idx'] + 1 for b in blocks if b['periodic'])
    irreg_pts = n_total - periodic_pts
    print(f"[*] {len(blocks)} blocks: {n_periodic} periodic ({periodic_pts} pts), "
          f"{n_irreg} irregular ({irreg_pts} pts)")

    # ---- Process blocks ----
    block_results = []
    for blk in blocks:
        if blk['periodic']:
            res = process_periodic_block(blk, times, pos)
        else:
            res = process_irregular_block(blk, times, pos)
        block_results.append(res)

    # ---- Merge, enforce speed, cleanup ----
    all_actions = merge_blocks(block_results)
    all_actions = enforce_speed_limit(all_actions)
    cleaned = final_cleanup(all_actions, times, pos)

    # ---- Validation ----
    if DEBUG:
        print(f"\n[DEBUG] ====== OUTPUT VALIDATION ======")
        platform_count = 0
        for i in range(1, len(cleaned)):
            dt = cleaned[i]['at'] - cleaned[i - 1]['at']
            if dt > PLATFORM_WARN_MS and abs(cleaned[i]['pos'] - cleaned[i - 1]['pos']) < 0.01:
                platform_count += 1
                if platform_count <= 5:
                    print(f"  PLATFORM: {cleaned[i - 1]['at']} → {cleaned[i]['at']} "
                          f"({dt} ms)  pos={cleaned[i]['pos']:.1f}")
        if platform_count > 5:
            print(f"  ... and {platform_count - 5} more platforms")
        print(f"  Total suspicious platforms (>500ms same pos): {platform_count}")

        out_max_speed = 0.0
        out_overspeed = 0
        for i in range(len(cleaned) - 1):
            dt = (cleaned[i + 1]['at'] - cleaned[i]['at']) / 1000.0
            if dt > 0:
                sp = abs(cleaned[i + 1]['pos'] - cleaned[i]['pos']) / dt
                if sp > out_max_speed:
                    out_max_speed = sp
                if sp > MAX_SPEED + 0.5:
                    out_overspeed += 1
        print(f"  Output max speed: {out_max_speed:.2f}  "
              f"overspeed segments: {out_overspeed}/{len(cleaned) - 1}")
        print(f"  Output points: {len(cleaned)} (was {n_total})")
        print(f"  First pt: at={cleaned[0]['at']}, pos={cleaned[0]['pos']}")
        print(f"  Last  pt: at={cleaned[-1]['at']}, pos={cleaned[-1]['pos']}")
        print("[DEBUG] ==============================\n")

    script['actions'] = cleaned
    save_json(script, output_path)
    print(f"[*] Written to: {output_path}  ({len(cleaned)} actions)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python limit_speed_deepseek.py <input.funscript> <output.funscript>")
        sys.exit(1)
    limit_speed(sys.argv[1], sys.argv[2])
