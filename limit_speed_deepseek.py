import json
import sys
import math

def find_speed_groups(actions):
    n = len(actions)
    if n < 2:
        return [], []

    is_fast = [False] * (n - 1)
    for i in range(n - 1):
        dt = actions[i+1]['at'] - actions[i]['at']
        if dt <= 0:
            continue
        speed = abs(actions[i+1]['pos'] - actions[i]['pos']) / (dt / 1000.0)
        if speed > 600:
            is_fast[i] = True

    groups = []
    i = 0
    while i < n - 1:
        if not is_fast[i]:
            i += 1
            continue
        start_idx = i
        while i < n - 1 and is_fast[i]:
            i += 1
        end_idx = i
        groups.append((start_idx, end_idx))
    
    isolated = []
    continuous = []
    for s, e in groups:
        if e - s + 1 <= 2:
            isolated.append((s, e))
        else:
            continuous.append((s, e))
    return isolated, continuous

def reduce_amplitude(actions, s, e):
    if e - s != 1: return
    a0 = actions[s]
    a1 = actions[e]
    dt = a1['at'] - a0['at']
    if dt <= 0: return
    diff = abs(a1['pos'] - a0['pos'])
    if diff == 0: return
    
    max_allowed_diff = int(600 * dt / 1000.0)
    if max_allowed_diff >= diff: return
    
    mid = (a0['pos'] + a1['pos']) / 2.0
    half = max_allowed_diff / 2.0
    if a1['pos'] > a0['pos']:
        new_pos0 = int(round(mid - half))
        new_pos1 = new_pos0 + max_allowed_diff
    else:
        new_pos0 = int(round(mid + half))
        new_pos1 = new_pos0 - max_allowed_diff
        
    a0['pos'] = new_pos0
    a1['pos'] = new_pos1

def process_continuous_block(actions, s, e):
    sub = actions[s:e+1]
    n = len(sub)
    if n <= 2: return

    start_at = sub[0]['at']
    end_at = sub[-1]['at']
    total_duration = end_at - start_at

    min_deltas = []
    for i in range(n-1):
        diff = abs(sub[i+1]['pos'] - sub[i]['pos'])
        min_deltas.append(diff * 1000.0 / 600.0)

    if all(d == 0 for d in min_deltas):
        if n > 1:
            avg = total_duration / (n-1)
            cur = start_at
            for i in range(1, n-1):
                cur += avg
                new_t = int(round(cur))
                if new_t <= sub[i-1]['at']:
                    new_t = sub[i-1]['at'] + 1
                sub[i]['at'] = new_t
            sub[-1]['at'] = end_at
        return

    while sum(min_deltas) > total_duration and n > 2:
        best_idx = 1
        best_red = -float('inf')
        for idx in range(1, n-1):
            d_prev = min_deltas[idx-1]
            d_next = min_deltas[idx]
            d_new = abs(sub[idx+1]['pos'] - sub[idx-1]['pos']) * 1000.0 / 600.0
            red = d_prev + d_next - d_new
            if red > best_red:
                best_red = red
                best_idx = idx
        
        # 检查删除 best_idx 后是否会导致剩余段首尾位置相同
        after_del = sub[:best_idx] + sub[best_idx+1:]
        if after_del[0]['pos'] == after_del[-1]['pos']:
            # 首尾相同，无法通过删点获得有意义的运动，改为整体缩放位移
            total_movement = sum(abs(sub[i+1]['pos'] - sub[i]['pos']) for i in range(n-1))
            if total_movement > 0:
                scale = (600 * total_duration / 1000.0) / total_movement
                base = sub[0]['pos']
                for i in range(1, n):
                    delta = sub[i]['pos'] - base
                    sub[i]['pos'] = base + int(round(delta * scale))
                # 重新计算 min_deltas
                min_deltas = [abs(sub[i+1]['pos'] - sub[i]['pos']) * 1000.0 / 600.0 for i in range(n-1)]
            break  # 跳出删点循环
        
        # 安全删除
        del sub[best_idx]
        n -= 1
        min_deltas = [abs(sub[i+1]['pos'] - sub[i]['pos']) * 1000.0 / 600.0 for i in range(n-1)]

    if n == 2:
        reduce_amplitude(sub, 0, 1)
        actions[s:e+1] = sub
        return

    # 核心修复：先压缩连续相同位置的动作
    compressed = [sub[0]]
    for act in sub[1:]:
        if act['pos'] != compressed[-1]['pos']:
            compressed.append(act)
    if len(compressed) != len(sub):
        sub = compressed
        n = len(sub)
        min_deltas = [abs(sub[i+1]['pos'] - sub[i]['pos']) * 1000.0 / 600.0 for i in range(n-1)]
        # 如果压缩后剩下2个点，直接降幅
        if n == 2:
            reduce_amplitude(sub, 0, 1)
            actions[s:e+1] = sub
            return

    sum_min = sum(min_deltas)
    factor = total_duration / sum_min if sum_min > 0 else 1.0
    
    cur = start_at
    prev_t = cur
    for i in range(1, n):
        min_dt = min_deltas[i-1] * factor
        cur += min_dt
        new_t = int(math.ceil(cur))
        # 只有在真正需要的时候才强制递增（避免制造无用冗余点）
        if new_t <= prev_t:
            if min_dt == 0:
                # 如果间隔本身不需要时间，那就老老实实停在 prev_t，最后会用 prev_t 做末尾对齐
                new_t = prev_t
            else:
                new_t = prev_t + 1
        if new_t > end_at - (n - 1 - i):
            new_t = end_at - (n - 1 - i)
            if new_t <= prev_t:
                new_t = prev_t + 1
        sub[i]['at'] = new_t
        prev_t = new_t
    sub[-1]['at'] = end_at
    actions[s:e+1] = sub

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    actions = data['actions']

    isolated, continuous = find_speed_groups(actions)

    for s, e in isolated:
        i = s
        while i <= e:
            if i+1 <= e:
                reduce_amplitude(actions, i, i+1)
            i += 1

    for s, e in continuous:
        process_continuous_block(actions, s, e)

    # 终极安全扫尾：确保没有任何跨界漏网之鱼
    for i in range(len(actions) - 1):
        dt = actions[i+1]['at'] - actions[i]['at']
        if dt <= 0: continue
        speed = abs(actions[i+1]['pos'] - actions[i]['pos']) / (dt / 1000.0)
        if speed > 600:
            reduce_amplitude(actions, i, i+1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    print(f"处理完成，输出文件：{output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法：python limit_speed_deepseek.py <输入JSON> <输出JSON>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
