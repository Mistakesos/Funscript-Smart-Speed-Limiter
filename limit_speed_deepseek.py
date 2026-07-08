#!/usr/bin/env python3

"""
Smart Speed Limiter for Funscript — Signal Processing & Pattern Recognition
============================================================================
Phase 1: Pre-process flat segments (preserve intentional holds)
Phase 2: Global ACF-based periodicity detection & segmentation  
Phase 3: Motif extraction from periodic blocks
Phase 4: Whole-cycle deletion with uniform time-warping (phase-locked)
Phase 5: Irregular block amplitude clamping (last-resort safety net)

Usage: python limit_speed_deepseek.py <input.funscript> <output.funscript>

The only hardcoded configuration is MAX_SPEED at the top of this file.
All other parameters are dynamically computed from the script data at runtime.
"""

import json
import sys
import math
from typing import List, Tuple, Optional, Dict, Any

# =========================================================================
#  CONFIGURATION — the ONLY magic number you should ever need to change
# =========================================================================
MAX_SPEED = 600
DEBUG = True

# -------------------------------------------------------------------------
#  ACF / sliding‑window tuning knobs
# -------------------------------------------------------------------------
ACF_WINDOW_POINTS = 300
ACF_STEP = 60
ACF_PEAK_THRESHOLD = 0.55
MIN_PERIOD_POINTS = 2
MAX_PERIOD_POINTS = 200

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
#  STEP 1 — PRE-PROCESS: FLAT SEGMENT DETECTION
# =========================================================================

def extract_flat_segments(times: List[int], positions: List[float],
                          min_flat_duration_ms: int = 2000) -> List[dict]:
    """
    Identify contiguous ranges where position doesn't change for a
    significant duration. These are intentional holds / pauses and should
    be preserved as irregular blocks, not fed to the periodic processor.
    Returns a list of block dicts (periodic=False) for each flat segment.
    """
    n = len(positions)
    if n < 2:
        return []
    
    flats = []
    seg_start = 0
    # Tolerance: positions within 0.1 are considered "same"
    for i in range(1, n):
        if abs(positions[i] - positions[seg_start]) > 0.1:
            # End of this flat candidate
            duration = times[i - 1] - times[seg_start]
            if duration >= min_flat_duration_ms and i - seg_start >= 2:
                flats.append({
                    'start_idx': seg_start,
                    'end_idx': i - 1,
                    'periodic': False,
                    'period': None,
                    'motif': None,
                    'is_flat': True
                })
            seg_start = i
    
    # Check trailing flat
    if seg_start < n - 1:
        duration = times[-1] - times[seg_start]
        if duration >= min_flat_duration_ms and n - seg_start >= 2:
            flats.append({
                'start_idx': seg_start,
                'end_idx': n - 1,
                'periodic': False,
                'period': None,
                'motif': None,
                'is_flat': True
            })
    
    return flats


# =========================================================================
#  STEP 2 — AUTOCORRELATION & PERIOD DETECTION
# =========================================================================

def autocorrelation(series: List[float]) -> List[float]:
    n = len(series)
    if n < 2:
        return [1.0]
    mean = sum(series) / n
    centered = [x - mean for x in series]
    denom = sum(c * c for c in centered)
    if denom == 0:
        return [1.0] + [0.0] * (n - 1)
    r = []
    for lag in range(n):
        num = sum(centered[i] * centered[i + lag] for i in range(n - lag))
        r.append(num / denom)
    return r


def find_all_strong_peaks(acf: List[float], min_lag: int, max_lag: int,
                          threshold: float = None) -> List[int]:
    if threshold is None:
        threshold = ACF_PEAK_THRESHOLD
    max_lag = min(max_lag, len(acf) - 1)
    peaks = []
    for lag in range(min_lag, max_lag + 1):
        if acf[lag] >= threshold:
            left_ok = (lag == min_lag or acf[lag] >= acf[lag - 1])
            right_ok = (lag == max_lag or acf[lag] >= acf[lag + 1])
            if left_ok and right_ok:
                peaks.append((lag, acf[lag]))
    peaks.sort(key=lambda x: -x[1])
    return [p[0] for p in peaks]


# =========================================================================
#  MOTIF EXTRACTION
# =========================================================================

def minimal_period(p_seq: List[float]) -> int:
    L = len(p_seq)
    for k in range(1, L):
        if L % k != 0:
            continue
        ok = True
        for i in range(L):
            if abs(p_seq[i] - p_seq[i % k]) > 0.5:
                ok = False
                break
        if ok:
            return k
    return L


def extract_motif_at(times: List[int], positions: List[float],
                     start_idx: int, period: int) -> Tuple[List[int], List[float]]:
    end_idx = min(start_idx + period, len(positions))
    t0 = times[start_idx]
    rel_t = [times[i] - t0 for i in range(start_idx, end_idx)]
    p_seq = [positions[i] for i in range(start_idx, end_idx)]
    return rel_t, p_seq


def trim_flat_ends(p_seq: List[float], t_rel: List[int]) -> Tuple[List[float], List[int]]:
    if len(p_seq) <= 2:
        return p_seq, t_rel
    start = 0
    while start < len(p_seq) - 1 and abs(p_seq[start] - p_seq[start + 1]) < 0.001:
        start += 1
    end = len(p_seq)
    while end > start + 1 and abs(p_seq[end - 1] - p_seq[end - 2]) < 0.001:
        end -= 1
    if start > 0 or end < len(p_seq):
        p_seq = p_seq[start:end]
        t_rel = [t - t_rel[start] for t in t_rel[start:end]]
    return p_seq, t_rel


def motif_similarity(m1: Optional[Tuple], m2: Optional[Tuple],
                     corr_threshold: float = 0.85) -> bool:
    if m1 is None or m2 is None:
        return False
    _, p1 = m1
    _, p2 = m2
    if len(p1) != len(p2):
        return False
    mu1 = sum(p1) / len(p1)
    mu2 = sum(p2) / len(p2)
    num = sum((a - mu1) * (b - mu2) for a, b in zip(p1, p2))
    den = (sum((a - mu1) ** 2 for a in p1) * sum((b - mu2) ** 2 for b in p2)) ** 0.5
    if den == 0:
        return True
    return num / den >= corr_threshold


# =========================================================================
#  STEP 3 — GLOBAL SEGMENTATION
# =========================================================================

def segment_script(times: List[int], positions: List[float]) -> List[dict]:
    n = len(positions)
    if n < MIN_PERIOD_POINTS + 1:
        return [{'start_idx': 0, 'end_idx': n - 1, 'periodic': False,
                 'period': None, 'motif': None}]

    # ---- 3a. Pre-identify flat segments ----
    flat_segs = extract_flat_segments(times, positions)
    flat_idx_set = set()
    for fs in flat_segs:
        for k in range(fs['start_idx'], fs['end_idx'] + 1):
            flat_idx_set.add(k)

    if DEBUG:
        for fs in flat_segs:
            print(f"  [FLAT] idx [{fs['start_idx']}-{fs['end_idx']}] "
                  f"time {times[fs['start_idx']]}-{times[fs['end_idx']]} ms, "
                  f"pos={positions[fs['start_idx']]:.1f}")

    # ---- 3b. Sliding-window ACF voting (skip flat indices) ----
    vote_periodic = [0.0] * n
    vote_period = [None] * n

    for start in range(0, n - MIN_PERIOD_POINTS, ACF_STEP):
        end = min(start + ACF_WINDOW_POINTS, n)
        window_pos = positions[start:end]
        if len(window_pos) < MIN_PERIOD_POINTS + 1:
            continue

        acf = autocorrelation(window_pos)
        peaks = find_all_strong_peaks(
            acf, MIN_PERIOD_POINTS,
            min(MAX_PERIOD_POINTS, len(window_pos) // 2)
        )

        a = start + (end - start) // 4
        b = end - (end - start) // 4
        for i in range(a, b):
            if i in flat_idx_set:
                continue  # never override flat detection
            if peaks:
                vote_periodic[i] += 1.0
                vote_period[i] = peaks[0]
            else:
                vote_periodic[i] -= 0.6

    # ---- 3c. Fill unlabelled indices ----
    for i in range(n):
        if i in flat_idx_set:
            vote_periodic[i] = -1.0
            vote_period[i] = None
            continue
        if vote_periodic[i] == 0.0:
            for d in range(1, n):
                if i - d >= 0 and vote_periodic[i - d] != 0.0 and (i - d) not in flat_idx_set:
                    vote_periodic[i] = vote_periodic[i - d]
                    vote_period[i] = vote_period[i - d]
                    break
                if i + d < n and vote_periodic[i + d] != 0.0 and (i + d) not in flat_idx_set:
                    vote_periodic[i] = vote_periodic[i + d]
                    vote_period[i] = vote_period[i + d]
                    break
            else:
                vote_periodic[i] = -1.0

    # ---- 3d. Binarise & build raw blocks ----
    label = [1 if v > 0 else 0 for v in vote_periodic]
    # Flat segments are always irregular
    for i in flat_idx_set:
        label[i] = 0

    raw_blocks = []
    i = 0
    while i < n:
        state = label[i]
        per = vote_period[i] if state == 1 else None
        j = i
        while j < n and label[j] == state and (state == 0 or vote_period[j] == per):
            j += 1
        raw_blocks.append({
            'start_idx': i,
            'end_idx': j - 1,
            'periodic': state == 1,
            'period': per,
            'motif': None
        })
        i = j

    # ---- 3e. Split large periodic blocks by motif consistency ----
    # ACF sees period=4 across 900 points, but within that span the
    # pattern changes every ~16-20 points.  We slide a verification
    # window and split wherever the local pattern diverges from the
    # block's extracted motif.
    raw_blocks = split_by_motif_consistency(raw_blocks, times, positions)

    # ---- 3f. Extract motif for each periodic block ----
    for blk in raw_blocks:
        if not blk['periodic'] or blk['period'] is None:
            continue
        period = blk['period']
        idx0 = blk['start_idx']
        idx1 = blk['end_idx']
        block_len = idx1 - idx0 + 1

        if block_len < period:
            blk['periodic'] = False
            blk['period'] = None
            continue

        best_motif = None
        best_score = -1.0
        max_offset = min(period * 2, max(1, block_len - period))
        for offset in range(max_offset):
            cand_rel_t, cand_p = extract_motif_at(times, positions, idx0 + offset, period)
            mp = minimal_period(cand_p)
            if mp < len(cand_p):
                cand_p = cand_p[:mp]
                cand_rel_t = cand_rel_t[:mp]
            cand_p, cand_rel_t = trim_flat_ends(cand_p, cand_rel_t)
            if len(cand_p) < 2:
                continue
            score = 0.5
            if idx0 + offset + period < idx1:
                next_p = positions[idx0 + offset + period:
                                   min(idx0 + offset + 2 * period, idx1 + 1)]
                if len(next_p) >= len(cand_p):
                    next_p = next_p[:len(cand_p)]
                    mu1 = sum(cand_p) / len(cand_p)
                    mu2 = sum(next_p) / len(next_p)
                    num = sum((a - mu1) * (b - mu2) for a, b in zip(cand_p, next_p))
                    den = (sum((a - mu1) ** 2 for a in cand_p) *
                           sum((b - mu2) ** 2 for b in next_p)) ** 0.5
                    score = num / den if den > 0 else 1.0
            if abs(cand_p[0] - positions[idx0]) < 0.5:
                score += 0.1
            if score > best_score:
                best_score = score
                best_motif = (cand_rel_t, cand_p)

        if best_motif is not None and len(best_motif[1]) >= 2:
            blk['motif'] = best_motif
            blk['period'] = len(best_motif[1])
        else:
            blk['periodic'] = False
            blk['period'] = None

    # ---- 3g. Merge adjacent periodic blocks with similar motifs ----
    merged = []
    for blk in raw_blocks:
        if not merged:
            merged.append(blk)
            continue
        prev = merged[-1]
        if (prev['periodic'] and blk['periodic'] and
                prev['period'] == blk['period'] and
                motif_similarity(prev.get('motif'), blk.get('motif'))):
            prev['end_idx'] = blk['end_idx']
        elif (not prev['periodic'] and not blk['periodic']):
            prev['end_idx'] = blk['end_idx']
        else:
            merged.append(blk)

    # ---- 3h. Merge tiny irregular blocks into neighbours ----
    MIN_BLOCK_POINTS = 3
    refined = []
    for blk in merged:
        if not refined:
            refined.append(blk)
            continue
        prev = refined[-1]
        curr_len = blk['end_idx'] - blk['start_idx'] + 1
        prev_len = prev['end_idx'] - prev['start_idx'] + 1
        if not blk['periodic'] and curr_len < MIN_BLOCK_POINTS:
            prev['end_idx'] = blk['end_idx']
        elif not prev['periodic'] and prev_len < MIN_BLOCK_POINTS and len(refined) >= 2:
            refined[-2]['end_idx'] = blk['end_idx']
            refined.pop()
            refined.append(blk)
        else:
            refined.append(blk)

    if DEBUG:
        print("\n[DEBUG] ====== SEGMENTATION RESULT ======")
        for idx, blk in enumerate(refined):
            st, en = blk['start_idx'], blk['end_idx']
            t0, t1 = times[st], times[en]
            dur = t1 - t0
            pts = en - st + 1
            if blk['periodic']:
                mt, mp = blk['motif'] if blk['motif'] else ([], [])
                print(f"  Block {idx}: idx [{st}-{en}]  time {t0}-{t1} ms "
                      f"({dur} ms, {pts} pts)  PERIODIC  period={blk['period']} pts")
                print(f"           Motif positions : {mp}")
            else:
                print(f"  Block {idx}: idx [{st}-{en}]  time {t0}-{t1} ms "
                      f"({dur} ms, {pts} pts)  IRREGULAR")
        print("[DEBUG] ==================================\n")

    return refined


# =========================================================================
#  STEP 3e — MOTIF-CONSISTENCY SPLITTING
# =========================================================================

def split_by_motif_consistency(blocks: List[dict],
                               times: List[int],
                               positions: List[float]) -> List[dict]:
    """
    For large periodic blocks where ACF detected the same period but the
    actual shape changes, split at points where the delta fingerprint
    (differences between consecutive positions within a window) diverges.

    Delta fingerprinting is far more sensitive to pattern changes than
    position-based MSE comparisons, especially for long-period motifs.
    """

    def _fingerprint(win: List[float]) -> Tuple[float, ...]:
        """Return the delta sequence of a window."""
        return tuple(round(win[i + 1] - win[i], 2) for i in range(len(win) - 1))

    def _fingerprint_dist(fp1: Tuple, fp2: Tuple) -> float:
        """Euclidean distance between two fingerprints, normalised."""
        if len(fp1) != len(fp2):
            return float('inf')
        ssq = sum((a - b) ** 2 for a, b in zip(fp1, fp2))
        return (ssq / len(fp1)) ** 0.5  # RMS distance of deltas

    result = []
    for blk in blocks:
        if not blk['periodic'] or blk['period'] is None:
            result.append(blk)
            continue

        period = blk['period']
        idx0 = blk['start_idx']
        idx1 = blk['end_idx']
        block_len = idx1 - idx0 + 1

        # Only split blocks with ≥ 6 periods
        if block_len < period * 6:
            result.append(blk)
            continue

        # ---- Reference fingerprint from first period ----
        ref_win = positions[idx0: idx0 + period]
        ref_fp = _fingerprint(ref_win)

        # Threshold: RMS delta distance > 15 means patterns differ
        FP_THRESH = 12.0

        split_positions = [idx0]

        for cursor in range(idx0 + period, idx1 - period + 1, period):
            win = positions[cursor: cursor + period]
            fp = _fingerprint(win)
            dist = _fingerprint_dist(ref_fp, fp)

            if dist > FP_THRESH:
                split_positions.append(cursor)
                ref_fp = fp  # new reference

        split_positions.append(idx1 + 1)

        if len(split_positions) <= 2:
            result.append(blk)
            continue

        if DEBUG:
            print(f"  [SPLIT] Block [{idx0}-{idx1}] ({block_len} pts, "
                  f"period={period}) → {len(split_positions) - 2} splits")

        for k in range(len(split_positions) - 1):
            s = split_positions[k]
            e = split_positions[k + 1] - 1
            if e - s + 1 < period:
                if result and not result[-1]['periodic']:
                    result[-1]['end_idx'] = e
                else:
                    result.append({
                        'start_idx': s, 'end_idx': e,
                        'periodic': False, 'period': None, 'motif': None
                    })
                continue
            result.append({
                'start_idx': s, 'end_idx': e,
                'periodic': True,
                'period': period,
                'motif': None
            })

    return result


# =========================================================================
#  STEP 4 — PERIODIC BLOCK PROCESSING
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


def process_periodic_block(blk: dict, times: List[int],
                           positions: List[float]) -> List[dict]:
    idx0 = blk['start_idx']
    idx1 = blk['end_idx']
    t_start = times[idx0]
    t_end = times[idx1]
    block_duration = t_end - t_start

    if block_duration <= 0:
        return [{'at': t_start, 'pos': positions[idx0]}]

    pos_start = positions[idx0]
    pos_end = positions[idx1]

    if blk['motif'] is None:
        return fallback_amplitude_limiter(times, positions, idx0, idx1)

    motif_t_rel, motif_p = blk['motif']
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

        # Force pos_start prefix
        if abs(body[0] - pos_start) > 0.5:
            trial = [pos_start] + body
        else:
            trial = list(body)

        # Correction to pos_end if needed
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
        actions.append({'at': int(round(t_val)), 'pos': round(p_val, 6)})

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
#  STEP 5 — IRREGULAR BLOCK PROCESSING
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


def process_irregular_block(blk: dict, times: List[int],
                            positions: List[float]) -> List[dict]:
    idx0 = blk['start_idx']
    idx1 = blk['end_idx']

    # If flat segment: preserve original points exactly (no processing)
    if blk.get('is_flat'):
        if DEBUG:
            orig_pts = idx1 - idx0 + 1
            print(f"  [Irregular/Flat] Block [{idx0}-{idx1}] "
                  f"({times[idx0]}-{times[idx1]} ms) "
                  f"— preserving {orig_pts} original points")
        return [{'at': times[i], 'pos': positions[i]}
                for i in range(idx0, idx1 + 1)]

    # Quick check: all same position → keep as-is
    all_same = all(abs(positions[i] - positions[idx0]) < 0.01
                   for i in range(idx0, idx1 + 1))
    if all_same:
        return [{'at': times[idx0], 'pos': positions[idx0]},
                {'at': times[idx1], 'pos': positions[idx1]}]

    return fallback_amplitude_limiter(times, positions, idx0, idx1)


# =========================================================================
#  STEP 6 — POST-PROCESSING & MERGE
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


def fix_rounding_overspeed(actions: List[dict]) -> List[dict]:
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

    # Lock first and last to original EXACT values
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

    # Quick pre-check
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

    print("[*] Running ACF-based segmentation ...")
    blocks = segment_script(times, pos)

    n_periodic = sum(1 for b in blocks if b['periodic'])
    n_irreg = len(blocks) - n_periodic
    periodic_pts = sum(b['end_idx'] - b['start_idx'] + 1 for b in blocks if b['periodic'])
    irreg_pts = n_total - periodic_pts
    print(f"[*] {len(blocks)} blocks: {n_periodic} periodic ({periodic_pts} pts), "
          f"{n_irreg} irregular ({irreg_pts} pts)")

    block_results = []
    for blk in blocks:
        if blk['periodic']:
            res = process_periodic_block(blk, times, pos)
        else:
            res = process_irregular_block(blk, times, pos)
        block_results.append(res)

    all_actions = merge_blocks(block_results)
    all_actions = fix_rounding_overspeed(all_actions)
    cleaned = final_cleanup(all_actions, times, pos)

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
