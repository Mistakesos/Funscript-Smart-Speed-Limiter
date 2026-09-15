#!/usr/bin/env python3
"""
Funscript Speed Limiter v3.2
在 v3.1 基础上加全局速度微调：
- 对速度 > MAX_SPEED + 10 的段，推后后点时间戳
- 整体重缩放保持总时长
- 迭代直到收敛
其它逻辑与 v3.1 一致。


设计目标：保幅度、保节奏、保时长，用"减少周期重复次数 + 整体拉伸"
替代简单的降幅限速。适用于有规律重复的脚本，也能安全处理不规则段。

用法: python limit_speed_deepseek.py <input.funscript> <output.funscript>
"""

import json
import sys
import math

### -> 这里的参数可以按需调整
MAX_SPEED = 600             # 最大速度限制
DEBUG = False               # 调试模式（如果启用会输出更多日志内容）
### <-


### -> ！ 以下参数理论可调，但非必要不动
TIME_RATIO_TOL = 0.15       # 相对时间比例容差。两个周期的内部时间分布差异小于 15% 就视为同一模板。数据抖动大就调高，模板精确就调低。
### <-


### -> ！！！ 以下参数通常不应该被调整，除非你知道你在做什么 ！！！
MIN_PERIOD_SEGS = 2         # P=1 意味着“两点之间的重复”，比如 0→100→0→100，这不是一个完整往返周期，是半个。P=2 才是最小的完整往返。这是下界，不能更小。
MAX_PERIOD_SEGS = 30        # 防性能爆炸。周期太长，check_period_match 的复杂度上升，重复次数也少，置信度低。30 是保守上限。
MIN_REPEATS = 3             # 统计学直觉：2 次可能是巧合，3 次开始可信。低一点会误判，高一点会漏掉。3 是平衡点。
MIN_REPEATS_P2 = 5          # P=2 的周期很短，容易巧合。要求 5 次是为了压制假阳性。这是 v3.1 加 P=2 时配套设的。
POS_TOL = 0.5               # Funscript 的 pos 通常是整数，0.5 等价于“允许 ±0.5”，几乎是精确匹配。因为脚本数据里位置大多是整数，所以这个容差很紧。

MAX_GAP_PTS = 15            # 间隙点数上限，这里多轮测试后敲定 15

SPEED_TOL = 10              # 全局微调容差：超过 MAX_SPEED + SPEED_TOL 才修，默认为 10
MAX_ITER = 8                # 全局修正的迭代上限，默认为 8。防止某个脚本卡在边界反复推后-缩回。正常 1–2 次就收敛。这是防死循环，不是调优参数。
### <-


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, separators=(',', ':'))


def preprocess_actions(actions):
    s = sorted(actions, key=lambda a: (a['at'], a['pos']))
    result = [s[0]]
    for a in s[1:]:
        prev = result[-1]
        if a['at'] - prev['at'] < 10 and abs(a['pos'] - prev['pos']) < 0.5:
            continue
        if a['at'] == prev['at']:
            result[-1] = a
            continue
        result.append(a)
    return result


def seg_speed(pa, pb, ta, tb):
    dt = (tb - ta) / 1000.0
    return abs(pb - pa) / dt if dt > 0 else 0.0


def check_period_match(positions, times, i, j, P):
    T_i = times[i + P] - times[i]
    T_j = times[j + P] - times[j]
    if T_i <= 0 or T_j <= 0:
        return False
    for k in range(P + 1):
        if abs(positions[i + k] - positions[j + k]) > POS_TOL:
            return False
        if 0 < k < P:
            t_i = (times[i + k] - times[i]) / T_i
            t_j = (times[j + k] - times[j]) / T_j
            if abs(t_i - t_j) > TIME_RATIO_TOL:
                return False
    return True


def find_periodic_blocks(times, positions):
    n = len(positions)
    blocks = []
    i = 0
    while i < n:
        found = False
        for P in range(MIN_PERIOD_SEGS, MAX_PERIOD_SEGS + 1):
            min_reps = MIN_REPEATS_P2 if P == 2 else MIN_REPEATS
            if i + 2 * P >= n:
                break
            if not check_period_match(positions, times, i, i + P, P):
                continue
            K = 2
            while (i + (K + 1) * P < n
                   and check_period_match(positions, times, i, i + K * P, P)):
                K += 1
            if K >= min_reps:
                blocks.append((i, P, K))
                i += K * P
                found = True
                break
        if not found:
            i += 1
    return blocks


def compute_cycle_info(start, P, K, times, positions):
    cycles = []
    for r in range(K):
        cs = start + r * P
        ce = cs + P
        T_cycle = times[ce] - times[cs]
        if T_cycle <= 0:
            continue
        V_max = 0.0
        for k in range(P):
            s = seg_speed(positions[cs + k], positions[cs + k + 1],
                          times[cs + k], times[cs + k + 1])
            if s > V_max:
                V_max = s
        cycles.append((cs, ce, T_cycle, V_max))
    return cycles


def scaled_max_speed(cycles, K_new, T_block):
    m = 0.0
    for r in range(min(K_new, len(cycles))):
        _, _, T_cycle, V_max = cycles[r]
        T_out = T_block / K_new
        scale = T_out / T_cycle
        m = max(m, V_max / scale)
    return m


def process_periodic_block(start, P, K, times, positions):
    end = start + P * K
    if end >= len(times):
        end = len(times) - 1
    t_start = times[start]
    t_end = times[end]
    T_block = t_end - t_start
    if T_block <= 0:
        return [{'at': times[start], 'pos': positions[start]}]

    cycles = compute_cycle_info(start, P, K, times, positions)
    if not cycles:
        return [{'at': times[i], 'pos': positions[i]} for i in range(start, end + 1)]

    K_new = K
    while K_new >= 1:
        v_out = scaled_max_speed(cycles, K_new, T_block)
        if v_out <= MAX_SPEED + 0.5:
            break
        K_new -= 1
    if K_new < 1:
        if DEBUG:
            print(f"  [WARN] start={start}: cannot fit even 1 cycle, keeping original")
        return [{'at': times[i], 'pos': positions[i]} for i in range(start, end + 1)]

    T_out = T_block / K_new
    out = []
    for r in range(K_new):
        cs, ce, T_cycle, _ = cycles[r]
        scale = T_out / T_cycle
        r_start = t_start + r * T_out
        begin_j = 0 if r == 0 else 1
        for j in range(begin_j, P + 1):
            t_rel = times[cs + j] - times[cs]
            t_new = r_start + t_rel * scale
            out.append({'at': int(round(t_new)), 'pos': positions[cs + j]})

    if out:
        out[0]['at'] = int(t_start)
        out[0]['pos'] = positions[start]
        out[-1]['at'] = int(t_end)
        out[-1]['pos'] = positions[end]

    for i in range(1, len(out)):
        if out[i]['at'] <= out[i - 1]['at']:
            out[i]['at'] = out[i - 1]['at'] + 1

    if DEBUG:
        print(f"  [Periodic] start={start} P={P} K={K} -> K_new={K_new} "
              f"(T_block={T_block}ms, out_pts={len(out)})")
    return out


def process_gap(times, positions, gs, ge):
    if ge <= gs:
        return [{'at': times[gs], 'pos': positions[gs]}]
    t_start = times[gs]
    t_end = times[ge]
    T_total = t_end - t_start
    if T_total <= 0:
        return [{'at': times[i], 'pos': positions[i]} for i in range(gs, ge+1)]

    n_segs = ge - gs
    t_min_arr = []
    t_orig_arr = []
    for i in range(gs, ge):
        dp = abs(positions[i+1] - positions[i])
        t_min_arr.append(dp / MAX_SPEED * 1000.0)
        t_orig_arr.append(times[i+1] - times[i])
    sum_min = sum(t_min_arr)
    sum_orig = sum(t_orig_arr)

    if sum_min > T_total:
        scale = T_total / sum_min
        t_out_arr = [t * scale for t in t_min_arr]
    else:
        slack = T_total - sum_min
        if sum_orig > 0:
            t_out_arr = [t_min_arr[i] + slack * t_orig_arr[i] / sum_orig
                         for i in range(n_segs)]
        else:
            t_out_arr = [T_total / n_segs] * n_segs

    out = [{'at': int(t_start), 'pos': positions[gs]}]
    cur_t = float(t_start)
    for i in range(n_segs):
        cur_t += t_out_arr[i]
        out.append({'at': int(round(cur_t)), 'pos': positions[i + gs + 1]})
    out[-1]['at'] = int(t_end)

    for i in range(1, len(out)):
        if out[i]['at'] <= out[i-1]['at']:
            out[i]['at'] = out[i-1]['at'] + 1

    if DEBUG:
        print(f"  [Gap] idx {gs}-{ge} ({T_total}ms) -> {len(out)} pts")
    return out


def enforce_speed_global(actions, orig_times):
    """全局速度微调：只处理明显超速的段（> MAX_SPEED + SPEED_TOL），
       推后后点时间戳，整体重缩放保持总时长。"""
    if len(actions) < 2:
        return actions
    actions = sorted(actions, key=lambda a: a['at'])
    target_dur = orig_times[-1] - orig_times[0]
    threshold = MAX_SPEED + SPEED_TOL

    for iteration in range(MAX_ITER):
        adjusted = False
        for i in range(1, len(actions)):
            prev = actions[i - 1]
            curr = actions[i]
            dt = curr['at'] - prev['at']
            if dt <= 0:
                curr['at'] = prev['at'] + 1
                adjusted = True
                continue
            dp = abs(curr['pos'] - prev['pos'])
            if dp == 0:
                continue
            sp = dp / (dt / 1000.0)
            if sp > threshold:
                needed_ms = int(math.ceil(dp / MAX_SPEED * 1000.0))
                if curr['at'] < prev['at'] + needed_ms:
                    curr['at'] = prev['at'] + needed_ms
                    adjusted = True

        t_first = actions[0]['at']
        t_last = actions[-1]['at']
        actual_dur = t_last - t_first
        if actual_dur > target_dur + 1:
            scale = target_dur / actual_dur
            for a in actions:
                a['at'] = int(round(t_first + (a['at'] - t_first) * scale))
            adjusted = True

        # 检查收敛
        max_v = 0.0
        for i in range(1, len(actions)):
            dt = (actions[i]['at'] - actions[i-1]['at']) / 1000.0
            if dt > 0:
                v = abs(actions[i]['pos'] - actions[i-1]['pos']) / dt
                if v > max_v:
                    max_v = v
        if DEBUG:
            print(f"  [GlobalFix] iter={iteration} max_v={max_v:.1f} "
                  f"dur={actual_dur}ms")
        if max_v <= threshold + 0.5:
            break
        if not adjusted:
            break

    actions[0]['at'] = orig_times[0]
    actions[-1]['at'] = orig_times[-1]
    return actions


def final_cleanup(actions, orig_times, orig_positions):
    if not actions:
        return []
    actions.sort(key=lambda a: a['at'])
    deduped = []
    for a in actions:
        if deduped and a['at'] == deduped[-1]['at']:
            deduped[-1] = a
        else:
            deduped.append(a)
    for i in range(1, len(deduped)):
        if deduped[i]['at'] <= deduped[i - 1]['at']:
            deduped[i]['at'] = deduped[i - 1]['at'] + 1
    if orig_times:
        deduped[0]['at'] = orig_times[0]
        deduped[0]['pos'] = orig_positions[0]
        deduped[-1]['at'] = orig_times[-1]
        deduped[-1]['pos'] = orig_positions[-1]
    return deduped


def report_validation(cleaned, orig_positions):
    if not DEBUG:
        return
    orig_pos_set = set(round(p, 6) for p in orig_positions)
    outliers = [a for a in cleaned if round(a['pos'], 6) not in orig_pos_set]
    print(f"\n[OUTLIER POS] {len(outliers)} points with non-original pos")
    for a in outliers[:10]:
        print(f"  at={a['at']}  pos={a['pos']}")

    overspeed = []
    for i in range(len(cleaned) - 1):
        s = seg_speed(cleaned[i]['pos'], cleaned[i + 1]['pos'],
                      cleaned[i]['at'], cleaned[i + 1]['at'])
        if s > MAX_SPEED + SPEED_TOL + 1:
            overspeed.append((i, cleaned[i]['at'], cleaned[i + 1]['at'],
                              s, cleaned[i]['pos'], cleaned[i + 1]['pos']))
    print(f"[OVERSPEED SEGMENTS > {MAX_SPEED + SPEED_TOL}] {len(overspeed)}")
    for (i, ta, tb, s, pa, pb) in overspeed[:20]:
        print(f"  idx={i}  {ta}({pa}) -> {tb}({pb})  dt={tb-ta}ms  speed={s:.0f}")
    if len(overspeed) > 20:
        print(f"  ... and {len(overspeed) - 20} more")

    # 统计 600~610 之间的段
    borderline = 0
    for i in range(len(cleaned) - 1):
        s = seg_speed(cleaned[i]['pos'], cleaned[i + 1]['pos'],
                      cleaned[i]['at'], cleaned[i + 1]['at'])
        if MAX_SPEED < s <= MAX_SPEED + SPEED_TOL:
            borderline += 1
    print(f"[BORDERLINE] {borderline} segments in ({MAX_SPEED}, {MAX_SPEED + SPEED_TOL}]")


def limit_speed(input_path, output_path):
    print(f"[*] Loading: {input_path}")
    script = load_json(input_path)
    if 'actions' not in script or not script['actions']:
        print("[!] Empty script")
        save_json(script, output_path)
        return

    actions = preprocess_actions(script['actions'])
    times = [a['at'] for a in actions]
    positions = [a['pos'] for a in actions]
    n = len(actions)
    print(f"[*] {n} points, duration {times[-1] - times[0]} ms")

    orig_max = 0.0
    for i in range(n - 1):
        s = seg_speed(positions[i], positions[i + 1], times[i], times[i + 1])
        if s > orig_max:
            orig_max = s
    print(f"[*] Original max speed: {orig_max:.1f}")

    print("[*] Finding periodic blocks...")
    blocks = find_periodic_blocks(times, positions)
    print(f"[*] Found {len(blocks)} periodic blocks")

    covered = set()
    for start, P, K in blocks:
        end = start + P * K
        if end >= n:
            end = n - 1
        for i in range(start, end + 1):
            covered.add(i)
    covered_pts = len(covered)
    print(f"[*] Covered points: {covered_pts}/{n} ({100*covered_pts/n:.1f}%)")

    print(f"\n[*] Processing {len(blocks)} periodic blocks...")
    all_actions = []
    for start, P, K in blocks:
        all_actions.extend(process_periodic_block(start, P, K, times, positions))

    i = 0
    gap_count = 0
    while i < n:
        if i not in covered:
            j = i
            while j < n and j not in covered:
                j += 1
            has_speed = False
            for k in range(i, j):
                if k + 1 < n:
                    s = seg_speed(positions[k], positions[k + 1],
                                  times[k], times[k + 1])
                    if s > MAX_SPEED:
                        has_speed = True
                        break
            if has_speed and (j - i) <= MAX_GAP_PTS:
                all_actions.extend(process_gap(times, positions, i, j - 1))
                gap_count += 1
            else:
                for k in range(i, j):
                    all_actions.append({'at': times[k], 'pos': positions[k]})
            i = j
        else:
            i += 1
    print(f"[*] Processed {gap_count} gaps")

    cleaned = final_cleanup(all_actions, times, positions)

    print(f"\n[*] Running global speed fix...")
    cleaned = enforce_speed_global(cleaned, times)
    cleaned = final_cleanup(cleaned, times, positions)

    if DEBUG:
        print(f"\n[*] Output points: {len(cleaned)} (was {n})")
        print(f"[*] First: at={cleaned[0]['at']} pos={cleaned[0]['pos']}")
        print(f"[*] Last:  at={cleaned[-1]['at']} pos={cleaned[-1]['pos']}")
        report_validation(cleaned, positions)

    script['actions'] = cleaned
    save_json(script, output_path)
    print(f"\n[*] Written to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python limit_speed_deepseek.py <input.funscript> <output.funscript>")
        sys.exit(1)
    limit_speed(sys.argv[1], sys.argv[2])
