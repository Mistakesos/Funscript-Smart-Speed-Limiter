## Introduce
**Smart Speed Limiter** is a lightweight Python script that automatically detects and corrects overspeed issues in JSON‑formatted action sequences. It keeps the movement speed between any two consecutive actions within a safe threshold of **600 units/second** by intelligently redistributing time or compressing amplitudes, while preserving the overall shape and rhythm of the original trajectory as much as possible.

> *Written by DeepSeek.*

## Usage
(Python 3.6+ required. If you're new to Python, just Google it or ask an AI.)

```bash
python limit_speed_deepseek.py <input.json> <output.json>
```

### Key Features
- **Automatic overspeed detection** – calculates instantaneous speed from timestamps (`at`) and positions (`pos`), and accurately flags all overspeed adjacent pairs.
- **Differentiates isolated vs. continuous overspeed blocks**
  - **Isolated (≤2 actions)** : applies a “mid‑point compression” strategy – directly reduces the displacement to the legal limit while keeping timestamps unchanged.
  - **Continuous (≥3 actions)** : uses a “time‑stretching + key‑point removal” approach – proportionally allocates the minimum required time across segments, and if the total minimum time exceeds the available duration, iteratively deletes intermediate points with the greatest time‑saving benefit.
- **Ultimate safety sweep** – after all processing, a final global check re‑examines every adjacent pair and forcibly compresses any remaining overspeed boundary, guaranteeing 100% compliance.

### How It Works
1. **Speed marking** – iterate over all adjacent pairs, compute `speed = |Δpos| / (Δat/1000)`, mark pairs where speed > 600.
2. **Grouping** – group consecutive marked indices into blocks, classify as isolated (length ≤2) or continuous (length ≥3).
3. **Isolated processing** – directly compress the displacement between the two actions to exactly hit 600 units/s.
4. **Continuous processing** –
   - Compute the minimum required time for each segment (based on 600 speed).
   - If total minimum time > actual duration, iteratively remove the intermediate point that yields the largest reduction in total minimum time.
   - Stretch the remaining segments proportionally, round timestamps (with ceiling and boundary protection) to ensure strictly increasing times while keeping endpoints fixed.
5. **Final sweep** – re‑scan all adjacent pairs and force‑compress any remaining overspeed.

## Comparison of Results

**Before**  
<img width="1287" height="185" alt="before" src="https://github.com/user-attachments/assets/dc46ff8c-4fcf-4bda-9853-9506ef67c5d6" />

**After**  
<img width="1296" height="192" alt="after" src="https://github.com/user-attachments/assets/a01ab302-a0f9-41b5-bbe9-aaa0d9de664a" />

**Compare**
<img width="1102" height="371" alt="compare" src="https://github.com/user-attachments/assets/457b773d-aa4d-46a0-9b2e-cf9f035628d5" />

**Random situation**
It appears that many adjustments have been made, but effectively, some points are discarded and the remaining ones are limited in range. This constitutes a mixed strategy. 
<img width="1113" height="376" alt="random" src="https://github.com/user-attachments/assets/e0cc316a-f391-4630-aaaf-850b6d7b097c" />

*The difference might be hard to see from some images, but in reality, for continuous overspeed blocks, the actual effect includes time redistribution and key‑point optimization, resulting in a better‑paced sequence overall.*
