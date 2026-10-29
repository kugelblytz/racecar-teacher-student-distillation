# Experiment log

> The original chronological lab log, kept as written. Commands and file references have been updated to the
> `scripts/{train,eval,verify,analysis}` layout (run them from the repository root with `python -m ...`); the
> "Project layout" section below predates that move. The summary is in the [README](../README.md).

Self-contained merged version of the **DM i AI 2025 race-car simulator** plus our full-throttle lane controller.

The simulator source and car assets come from the public challenge repository `amboltio/DM-i-AI-2025/race-car`. The controller/API in this repo are our implementation.

## What is implemented

- Full local Pygame race-car environment
- All three car sprites
- High-level policy restricted to `LEFT / STAY / RIGHT`
- Deterministic lane-change execution
- `ACCELERATE` whenever not steering
- TTC / closing-speed estimation from sensor history
- Emergency braking shield when no adjacent lane appears safe
- Short action batches (4) to reduce reaction latency
- Fixed FastAPI `/predict` endpoint
- Self-contained Python dependencies

The current `choose_lane()` implementation is heuristic. It is intentionally isolated so we can later replace it with PPO, DQN, behavior cloning, or a recurrent student without changing the low-level controller.

## Quick start

### 1. Clone

```bash
git clone git@github.com:kugelblytz/racecar-teacher-student-distillation.git
cd racecar-teacher-student-distillation
```

HTTPS alternative:

```bash
git clone https://github.com/kugelblytz/racecar-teacher-student-distillation.git
cd racecar-teacher-student-distillation
```

### 2. Create an environment and install dependencies

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the controller directly in the simulator

```bash
python run_controller.py
```

A Pygame window should open immediately. The yellow ego car is controlled by `controller.py`.

Run this command from the repository root because the simulator loads sprites from `public/assets/`.

## API mode

To run the competition-style prediction endpoint:

```bash
python api.py
```

Then open:

```text
http://localhost:9052/docs
```

The prediction endpoint is:

```text
POST /predict
```

## Project layout

```text
racecar-system/
├── controller.py
├── run_controller.py
├── api.py
├── dtos.py
├── requirements.txt
├── public/
│   └── assets/
│       ├── bluecar.png
│       ├── redcar.png
│       └── yellowcar.png
└── src/
    ├── elements/
    │   ├── car.py
    │   ├── road.py
    │   ├── sensor.py
    │   └── wall.py
    ├── game/
    │   └── core.py
    └── mathematics/
        ├── collision.py
        ├── randomizer.py
        └── vector.py
```

## Controller architecture

```text
16 sensors + velocity + sensor history
                 |
                 v
           lane policy
       LEFT / STAY / RIGHT
                 |
                 v
    deterministic lane controller
                 |
          safety shield
                 |
                 v
       simulator actions
```

## First test

For the first run, watch for:

- continuous acceleration on clear road
- lane changes before reaching slower traffic
- clean counter-steering into the new lane
- wall collisions
- side/rear collisions during lane changes
- final distance and tick count printed by the simulator

Do not add RL yet if the deterministic control layer itself is unstable. We should first use crashes from this baseline to identify which parts need to be fixed.

## Next experiments

1. Add a headless multi-seed benchmark runner.
2. Record distance, crash rate, maximum velocity and failure state.
3. Improve the deterministic safety/lane controller.
4. Build a privileged simulator-state teacher using exact car positions and rollouts.
5. Train a sensor-only student on teacher actions.
6. Fine-tune the high-level policy with RL if it adds value.


## Privileged PPO teacher

The heuristic controller remains available as the known-good baseline. The new
teacher is trained separately with exact simulator state and directly chooses all
five primitive actions:

```text
NOTHING
ACCELERATE
DECELERATE
STEER_LEFT
STEER_RIGHT
```

The PPO observation is compact privileged state rather than ray sensors:

- ego longitudinal speed
- ego lateral position and lateral speed
- time remaining
- active/spare traffic state
- for each of the 5 lanes: opponent presence, relative x, relative vx and vx

No RNG state or future spawn information is exposed.

### Smoke test training

First install the updated dependencies:

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run a short training job:

```powershell
.\.venv\Scripts\python.exe -m scripts.train.train_teacher --timesteps 100000 --envs 4 --eval-every 50000 --eval-episodes 8
```

Checkpoints and TensorBoard logs are written under `models/teacher/`.

### Full initial experiment

Once the smoke test runs correctly:

```powershell
.\.venv\Scripts\python.exe -m scripts.train.train_teacher --timesteps 5000000 --envs 8
```

The default PPO setup uses a 256-256-128 actor and critic, long-horizon
`gamma=0.9995`, 1-tick primitive actions, and fixed-seed evaluation every
500k training steps.

The reward is deliberately close to the challenge objective:

```text
reward = distance gained / 1000
         - 25 if the episode crashes
```

There are no lane-centering, steering, or hand-authored overtaking rewards.

### Evaluate a checkpoint

```powershell
.\.venv\Scripts\python.exe -m scripts.eval.evaluate_teacher models\teacher\best_teacher.zip --episodes 100 --csv models\teacher\eval_100.csv
```

The evaluator reports mean, median, 10th-percentile and min/max distance, crash
rate, and final speed over fixed seeds.

To continue training an existing checkpoint:

```powershell
.\.venv\Scripts\python.exe -m scripts.train.train_teacher --resume models\teacher\best_teacher.zip --timesteps 5000000 --envs 8
```

The first target is simple: demonstrate that privileged primitive-action PPO can
reliably beat the approximately 212k full-run heuristic baseline before we build
the recurrent sensor-only student.


## Fast training simulator

Privileged PPO training now uses a Pygame-free numeric simulator by default.
It reproduces the challenge's action dynamics, opponent random walk, spawn/retire
logic, collision rectangles and Python `random.Random` call order, but removes:

- Pygame rendering and surfaces
- sprite loading during each worker startup
- ray sensor updates
- `Car`, `Road` and `pygame.Rect` allocation in the hot step path

The original simulator remains available with `--sim original`.

### Verify exact-state parity first

After pulling a change to the fast simulator, run:

```bash
python -m scripts.verify.verify_fast_sim --seeds 50 --max-steps 2000
```

It feeds identical actions and seeds into the original and fast environments and
checks the privileged observation, reward, distance, speed, crash state and
termination at every tick.

Do not use a fast-simulator change for long training if this parity check fails.

### Compare PPO throughput

Fast simulator:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.train.train_teacher \
  --timesteps 500000 \
  --envs 24 \
  --device cpu \
  --sim fast \
  --crash-penalty 250 \
  --ent-coef 0 \
  --eval-every 10000000
```

Original simulator:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.train.train_teacher \
  --timesteps 500000 \
  --envs 24 \
  --device cpu \
  --sim original \
  --crash-penalty 250 \
  --ent-coef 0 \
  --eval-every 10000000
```

Compare the Stable-Baselines3 `fps` value. The fast version is now the default,
so `--sim fast` can normally be omitted.

This is still a scalar simulator running one environment per worker. If this
gives a large gain and parity is clean, the next optimization is a truly batched
NumPy/PyTorch simulator that advances hundreds or thousands of environments in a
single process.


## Fully GPU-native PPO teacher

For machines with a large CUDA GPU, `scripts/train/train_teacher_gpu.py` bypasses
Stable-Baselines3 and CPU Gym workers entirely.

The full hot path stays on CUDA:

```text
8192+ simultaneous races
        ↓
CUDA tensor physics / spawning / collisions
        ↓
CUDA privileged observations
        ↓
CUDA actor + critic
        ↓
CUDA rollout buffer
        ↓
CUDA GAE
        ↓
CUDA PPO updates
```

The GPU simulator matches the public dynamics and traffic distributions but does
not try to reproduce Python `random.Random` bit-for-bit. Checkpoints are
therefore evaluated on the original headless simulator using fixed seeds.

The policy still directly controls all five primitive actions. The default
initialization gives ACCELERATE / lane-holding a prior so primitive-action PPO
does not begin as a destructive uniform-random steering policy.

### RTX 5090 throughput smoke test

Pull the latest code and verify CUDA first:

```bash
git pull
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Then run two GPU PPO iterations with evaluation/checkpointing disabled:

```bash
python -m scripts.train.train_teacher_gpu \
  --timesteps 4194304 \
  --envs 8192 \
  --rollout-steps 256 \
  --device cuda \
  --eval-every 0 \
  --save-every 0
```

Each iteration contains `8192 × 256 = 2,097,152` environment transitions.
The printed `fps` is directly comparable to the old ~8.5k CPU/SB3 result,
although this new number includes both simulation and PPO optimization.

If VRAM usage is low, also try:

```bash
python -m scripts.train.train_teacher_gpu \
  --timesteps 8388608 \
  --envs 16384 \
  --rollout-steps 256 \
  --device cuda \
  --eval-every 0 \
  --save-every 0
```

If 8192 environments runs out of memory, retry with `--envs 4096`.

### Short learning + real-simulator validation

```bash
python -m scripts.train.train_teacher_gpu \
  --timesteps 20000000 \
  --envs 8192 \
  --rollout-steps 256 \
  --device cuda \
  --crash-penalty 250 \
  --ent-coef 0 \
  --eval-every 5000000 \
  --eval-episodes 32 \
  --save-every 5000000
```

Every evaluation is run with the original headless challenge simulator. Watch:

- original-simulator crash rate
- mean / median / p10 distance
- training entropy
- GPU training fps

### Longer run

Once original-simulator evaluation is improving:

```bash
python -m scripts.train.train_teacher_gpu \
  --timesteps 200000000 \
  --envs 8192 \
  --rollout-steps 256 \
  --device cuda \
  --crash-penalty 250 \
  --ent-coef 0 \
  --eval-every 10000000 \
  --eval-episodes 64 \
  --save-every 10000000
```

Checkpoints are written to `models/gpu_teacher/`.

Evaluate any checkpoint on 100 original-simulator seeds:

```bash
python -m scripts.eval.evaluate_teacher_gpu \
  models/gpu_teacher/best_teacher.pt \
  --episodes 100 \
  --csv models/gpu_teacher/eval_100.csv
```

Resume training:

```bash
python -m scripts.train.train_teacher_gpu \
  --resume models/gpu_teacher/latest_teacher.pt \
  --timesteps 400000000 \
  --envs 8192 \
  --device cuda
```


### Verify CUDA simulator fidelity

Before a long CUDA training run, validate the GPU simulator against the original
headless challenge simulator:

```bash
python -m scripts.verify.verify_gpu_sim --device cuda
```

The CUDA RNG stream is intentionally different from Python `random.Random`, so
same-seed traffic is not expected to be bit-for-bit identical. The verifier
instead checks:

- exact primitive-action ego dynamics and integration
- exact privileged observation encoding on controlled state
- exact pygame rectangle/wall collision boundaries
- statistical lane/front-back/spawn-speed parity against the original simulator

For an additional bit-for-bit check of the scalar numeric reference against the
original simulator, run:

```bash
python -m scripts.verify.verify_fast_sim --seeds 50 --max-steps 2000
```

Do not trust a long GPU training run if either deterministic fidelity check
fails.


### Policy-mode diagnostics

CUDA teacher evaluation now reports four modes at every evaluation point:

```text
ORIGINAL deterministic   # canonical deployment/checkpoint metric
ORIGINAL stochastic      # same fixed traffic seeds, sampled policy actions
GPU      deterministic   # CUDA simulator, argmax actions
GPU      stochastic      # CUDA simulator, sampled actions
```

The stochastic action RNG is separately seeded, so repeated evaluations are
reproducible. `best_teacher.pt` is still selected only by ORIGINAL
deterministic mean distance.

This distinction matters because the per-iteration training `crash=` metric is
based only on episodes that happened to finish inside that rollout and uses
sampled actions. It is not directly comparable to a complete deterministic
evaluation.

Standalone four-way evaluation:

```bash
python -m scripts.eval.evaluate_teacher_gpu \
  models/gpu_teacher/best_teacher.pt \
  --episodes 100 \
  --device cuda \
  --csv-prefix models/gpu_teacher/diagnostic
```

To keep the old single CSV behavior for the canonical original/deterministic
rows, use `--csv path.csv`.


## Direct legal-observation PPO baselines

Two CUDA-native baselines train directly from information available to the
challenge controller instead of privileged simulator state.

Legal observation (41 floats):

- 16 sensor distances, normalized to 0..1 with no-hit encoded as 1
- 16 sensor hit masks, preserving the API's Optional[float] distinction
- ego x velocity
- ego y velocity
- time remaining
- cumulative distance
- previous primitive action one-hot (controller-known history)

The CUDA ray caster uses the original 16 sensor angles, 1000 px range,
pygame-style integer rectangle geometry, opponent rectangles, and top/bottom
walls.

Verify CUDA legal observations against the exact original sensor implementation:

```bash
python -m scripts.verify.verify_legal_obs --device cuda
```

MLP smoke benchmark:

```bash
python -m scripts.train.train_direct_mlp_gpu --timesteps 8388608 --envs 32768 --rollout-steps 128 --device cuda --eval-every 0 --save-every 0
```

GRU smoke benchmark:

```bash
python -m scripts.train.train_direct_gru_gpu --timesteps 4194304 --envs 16384 --rollout-steps 128 --device cuda --eval-every 0 --save-every 0
```

Longer direct baselines:

```bash
python -m scripts.train.train_direct_mlp_gpu --timesteps 500000000 --envs 32768 --rollout-steps 128 --device cuda --eval-every 25000000 --eval-episodes 64 --eval-workers 24 --output models/direct_mlp

python -m scripts.train.train_direct_gru_gpu --timesteps 500000000 --envs 16384 --rollout-steps 128 --device cuda --eval-every 25000000 --eval-episodes 64 --eval-workers 24 --output models/direct_gru
```

Both trainers save separate `best_deterministic.pt` and
`best_stochastic.pt` checkpoints because PPO trains a sampled policy and
argmax behavior can differ substantially.

Standalone exact-original evaluation:

```bash
python -m scripts.eval.evaluate_direct_gpu models/direct_mlp/best_stochastic.pt --episodes 256 --seed-start 50000 --workers 24
python -m scripts.eval.evaluate_direct_gpu models/direct_gru/best_stochastic.pt --episodes 256 --seed-start 50000 --workers 24
```


### Direct-PPO traffic curriculum

Vanilla primitive-action PPO can fail to discover coherent lane changes because
the safe initialization samples steering very rarely and useful lane changes
require many coordinated steering/counter-steering ticks. The direct trainers
therefore support a training-only opponent-count curriculum:

```bash
python -m scripts.train.train_direct_mlp_gpu --timesteps 150000000 --envs 32768 --rollout-steps 128 --device cuda --curriculum-stage-steps 25000000 --eval-every 25000000 --eval-episodes 64 --eval-workers 24 --output models/direct_mlp_curriculum
```

This trains with 1 opponent for the first 25M environment steps, then 2, then 3,
then the competition's full 4 opponents from ~75M onward. Original-simulator
evaluation always uses the real four-opponent challenge, so curriculum affects
training only.

The same flag is available for `scripts/train/train_direct_gru_gpu.py`. Do not resume a
failed full-traffic run when testing the curriculum; start from a fresh model so
the early easy stages can shape exploration.


### Direct PPO action persistence

The direct trainers also support legal action persistence with
`--action-repeat N`. The policy still selects only the five challenge actions;
the chosen primitive action is held for N simulator ticks before the next policy
observation/decision. Training uses the corresponding multi-tick discounted
reward and discount factor, and exact-original evaluation repeats actions in the
same way.

This is intended as an exploration test for the unusual lateral dynamics, where
STEER changes lateral velocity rather than directly changing position. For
example, `--action-repeat 24` makes a hand-written 48-tick steer + 48-tick
counter-steer lane change representable as roughly LEFT, LEFT, RIGHT, RIGHT at
the policy-decision level.

`--timesteps` continues to count PPO policy decisions, not internal repeated
simulator ticks.


### MLP-warm-started residual GRU

`scripts/train/train_direct_residual_gru_gpu.py` turns a successful direct MLP checkpoint
into a recurrent policy without changing its initial behavior. The MLP
backbone/policy/value heads are copied exactly. A GRU reads the MLP's 128-D
features and a zero-initialized projection adds a recurrent residual back into
those features. Because that projection starts at exactly zero, logits and
values are bit-identical to the source MLP before the first optimizer update.

By default the copied MLP path is frozen (`--base-learning-rate 0`) and only
the recurrent branch trains at `5e-5`. This isolates the value of temporal
memory and protects the hard-won feed-forward behavior. Later runs can unfreeze
the base with a tiny nonzero learning rate.

Example:

```bash
python -m scripts.train.train_direct_residual_gru_gpu \
  --init-from-mlp models/direct_mlp_onecar/latest.pt \
  --timesteps 200000000 \
  --envs 16384 \
  --rollout-steps 128 \
  --training-opponents 1 \
  --device cuda \
  --eval-every 25000000 \
  --eval-episodes 64 \
  --eval-workers 24 \
  --output models/direct_residual_gru_onecar
```

The trainer automatically verifies exact MLP/residual-GRU equivalence before
training and performs an original four-opponent evaluation at step zero by
default.


### Legal observation V2: controller-tracked lateral position

Direct recurrent PPO can optionally append one controller-owned feature with
`--tracked-y`. V1 remains the original 41-D observation. V2 is exactly those
same 41 values plus one final feature:

```text
tracked_y_norm = (tracked_top_left_y - 510.5) / 470.5
```

so approximately -1 is the upper safe wall boundary, 0 is road center, and +1
is the lower safe wall boundary.

The value is not read from simulator-private ego position. A pure
`LateralPositionTracker` starts from the known reset state `y=510, vy=0`,
syncs `vy` from the legal API velocity field, and integrates only the actions
the controller itself issued. The GPU simulator maintains an independent copy
of the same action-only tracker so training can verify it against private state
without exposing private state to the policy.

A 41-D GRU can be resumed with `--tracked-y`. The trainer expands the first
encoder weight from 41 to 42 inputs, copies all 41 old columns unchanged, sets
the new tracked-y column to zero, and starts a fresh optimizer. Thus the new
feature is initially disconnected and the checkpoint's behavior is preserved
up to floating-point precision.

Verify the reconstruction and migration before training:

```bash
python -m scripts.verify.verify_tracked_y --device cuda
```

The verifier checks thousands of original-simulator steps, GPU batched dynamics,
episode resets, the normalized observation value, an explicit neighboring-car
sensor-occlusion case, and 41->42 GRU checkpoint migration.

Example migration/resume:

```bash
python -m scripts.train.train_direct_gru_gpu \
  --resume models/direct_gru_2p5b/latest.pt \
  --tracked-y \
  --timesteps 2000000000 \
  --envs 16384 \
  --rollout-steps 128 \
  --device cuda \
  --training-opponents 2 \
  --learning-rate 1e-4 \
  --eval-every 100000000 \
  --eval-episodes 64 \
  --eval-workers 24 \
  --save-every 100000000 \
  --output models/direct_gru_2p5b_tracked_y
```


### Direct-policy crash rewards

Direct MLP/GRU training now distinguishes wall crashes from opponent-car crashes:

```text
per tick distance reward = ego_vx / 1000
car crash penalty        = -50    (--crash-penalty)
wall crash penalty       = -100   (--wall-crash-penalty)
```

Crash penalties do not stack. If a wall and car overlap on the same terminal
tick, the wall penalty takes precedence, so the terminal penalty is -100 rather
than stacking both penalties. Shared teacher code remains backward compatible:
callers that do not
provide `wall_crash_penalty` fall back to the existing generic crash penalty.



The fresh GRU defaults also use `--ent-coef 0.01` and `--epochs 4`. The
entropy bonus is intended to resist premature collapse onto a low-action
coasting policy, while four PPO epochs make better use of each recurrent
rollout. `--clip-range 0.2`, `--target-kl 0.03`, and learning rate `3e-4`
remain unchanged.

Verify exact-original and GPU reward behavior with:

```bash
python -m scripts.verify.verify_wall_crash_reward --device cuda
```


### Privileged-teacher distillation into the legal GRU

`scripts/train/train_distilled_gru_gpu.py` pretrains the same deployable 42-D recurrent
student used by direct PPO, but the data-collection policy is the privileged
teacher. The teacher sees the 25-D exact simulator state; the student sees only
legal V2 observations (sensors, legal velocities/history, and controller-tracked
lateral y).

By default the teacher drives with the deterministic probability-credit
scheduler at temperature 1.0. This is the strongest teacher decoder found in
held-out original-simulator evaluation. The student is trained on both the
scheduler's executed primitive action (hard cross-entropy) and a smaller soft
cross-entropy term against the teacher's full action distribution.

The student value head is initialized to exactly zero and is not distilled,
because PPO fine-tuning uses a different direct-policy reward. Distillation
checkpoints intentionally store their optimizer under
`distill_optimizer_state`, not `optimizer_state`; therefore
`scripts/train/train_direct_gru_gpu.py --resume ...` starts fresh PPO Adam moments while
preserving the distilled policy weights.

The validated sequence-level MPC can now be used as an exact-simulator labeler
with `scripts/train/train_distilled_gru_mpc.py`. This path intentionally does not approximate
the planner inside the CUDA simulator: persistent CPU workers run the original
simulator and exact collision geometry, while the recurrent student update runs
on CUDA. Workers return only 42-D legal observations, MPC action labels, soft
targets, terminal boundaries, and aggregate diagnostics.

MPC trigger labels receive a default 2x loss weight and actual overrides receive
8x because the safety substitutions are rare. On an override, the soft target
defaults to the executed MPC action; on ordinary ticks it remains the privileged
PPO action distribution. The raw GPU distillation command remains unchanged.

Exact MPC distillation smoke/full examples:

```bash
python -m scripts.verify.verify_teacher_mpc_distill

python -m scripts.train.train_distilled_gru_mpc \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --transitions 10000000 \
  --workers 24 \
  --rollout-steps 256 \
  --device cuda \
  --eval-every 2000000 \
  --eval-episodes 128 \
  --eval-seed-start 400000 \
  --output models/distilled_gru_mpc
```

The command uses the promoted teacher settings by default: horizon 72,
strategic horizon 119 with one accepted intervention per episode, and 3-sigma
opponent uncertainty. Checkpoints record the complete MPC configuration,
teacher SHA-256, Git revision, collection counters, trigger reasons, label and
behavior action distributions, lane straddling, and held-out exact-simulator
results.

For dataset aggregation, a fixed prior student can drive part of collection
while the MPC still labels every visited state:

```bash
python -m scripts.train.train_distilled_gru_mpc \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --resume models/distilled_gru_mpc/best_deterministic.pt \
  --behavior-student-checkpoint models/distilled_gru_mpc/best_deterministic.pt \
  --student-behavior-fraction 0.5 \
  --behavior-mode stochastic \
  --transitions 20000000 \
  --output models/distilled_gru_mpc_dagger
```

This is a data-collection option, not a promotion claim. Safety-first held-out
evaluation remains the gate for deploying a student.

A stronger transfer mode starts from an already competent legal policy and
uses that same frozen policy for nominal proposals. The exact-state MPC then
changes only proposals it judges unsafe. This turns distillation into a safety
correction problem instead of asking a partially observed student to reproduce
the privileged PPO scheduler from scratch. Both 41-D V1 and 42-D V2 GRUs are
accepted; a 41-D initialization is expanded losslessly with a zero-weight
tracked-y input column.

```bash
python -m scripts.train.train_distilled_gru_mpc \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --init-student-checkpoint models/direct_gru_2p5b/best_deterministic.pt \
  --proposal-student-checkpoint models/direct_gru_2p5b/best_deterministic.pt \
  --proposal-mode deterministic \
  --transitions 1001472 \
  --workers 24 \
  --rollout-steps 256 \
  --data-seed-start 900000 \
  --learning-rate 1e-4 \
  --hard-weight 0 \
  --soft-weight 1 \
  --eval-every 251904 \
  --eval-episodes 64 \
  --eval-seed-start 400000 \
  --output models/direct_gru_mpc_corrected_v1
```

Non-overridden soft targets are the proposal policy's own distribution, so the
zero-hard-loss configuration above acts as behavior preservation plus weighted
MPC corrections. Checkpoint selection first minimizes the union of crashes and
episodes below 25k, then crash rate, and only then p10 and mean distance. This
prevents a stopped policy from winning merely because it avoids a collision.
Exact evaluation additionally supports `--modes scheduler` for the same
probability-credit decoder used elsewhere. `scripts/eval/evaluate_direct_gpu.py --output`
writes the aggregate report, while `--include-rows` adds per-seed collision
type, final lateral state, and action counts for failure analysis.

In the bounded 1,001,472-label experiment, deterministic performance on the
64-seed development block improved from 93.8% crashes for the source policy to
48.4%, while mean distance reached 49.3k and the below-25k rate fell from 57.8%
to 35.9%. On 256 untouched seeds beginning at 1,100,000, crashes fell from
91.4% to 47.3%, mean distance rose from 35.9k to 53.8k, and the below-25k rate
fell from 58.2% to 39.5%. P10 regressed from 7.0k to 6.3k, however, and nearly
half the episodes still crash. The resulting checkpoint is therefore a strong
experimental legal policy, not a promoted production policy.

One additional dataset-aggregation generation used that corrected policy as
the next frozen proposer. Its selected checkpoint reached 47.7% crashes on the
same untouched block but reduced p10 further to 5.4k, so it was rejected.

The next experiment added an auxiliary stay/left/right head trained from the
MPC's selected maneuver while retaining primitive-action correction loss:

```bash
python -m scripts.train.train_distilled_gru_mpc \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --init-student-checkpoint models/direct_gru_2p5b/best_deterministic.pt \
  --proposal-student-checkpoint models/direct_gru_2p5b/best_deterministic.pt \
  --proposal-mode deterministic \
  --transitions 1001472 \
  --workers 24 \
  --rollout-steps 256 \
  --data-seed-start 900000 \
  --learning-rate 1e-4 \
  --hard-weight 0 \
  --soft-weight 1 \
  --trigger-sample-weight 2 \
  --override-sample-weight 8 \
  --lane-option-loss-weight 1 \
  --lane-option-positive-weight 32 \
  --eval-every 251904 \
  --eval-episodes 64 \
  --eval-seed-start 400000 \
  --output models/direct_gru_mpc_options_v1
```

The separately decoded lane option remains experimental. Thresholds below 0.4
caused excessive or wall-crossing maneuvers; thresholds at or above 0.4 did not
change the chosen primitive actions on the development block. The selected
deployment decoder is therefore still `deterministic`, which uses only legal
observations and the five primitive actions. The option labels serve as an
auxiliary representation-learning target, not a privileged runtime input.

At 1,001,472 labels, the deterministic checkpoint reached 34.4% crashes, mean
66.5k, p10 7.0k, and 28.1% below 25k on the 64-seed development block. On the
untouched 256 seeds beginning at 1,100,000 it reached 39.8% crashes, mean 62.5k,
p10 6.0k, and 37.1% below 25k. This improves the previous corrected student's
47.3% crash rate, 53.8k mean, and 39.5% below-25k rate. All holdout crashes were
opponent collisions and none were wall collisions. Because p10 remains below
the original policy's 7.0k and 39.8% still crash, this is now the selected
**experimental** artifact (`best_deterministic.pt`), but it is not promoted as
a production-safe policy.

Smoke test:

```bash
python -m scripts.train.train_distilled_gru_gpu \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --transitions 4194304 \
  --envs 4096 \
  --rollout-steps 128 \
  --device cuda \
  --eval-every 0 \
  --save-every 0 \
  --output models/distilled_gru_smoke
```

Initial full distillation experiment:

```bash
python -m scripts.train.train_distilled_gru_gpu \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --transitions 200000000 \
  --envs 16384 \
  --rollout-steps 128 \
  --device cuda \
  --teacher-mode scheduler \
  --teacher-temperature 1.0 \
  --eval-every 25000000 \
  --eval-episodes 64 \
  --eval-workers 24 \
  --save-every 25000000 \
  --output models/distilled_gru_v1
```

Then fine-tune the best distilled student with direct PPO. This remains a fully
legal 42-D policy at deployment:

```bash
python -m scripts.train.train_direct_gru_gpu \
  --resume models/distilled_gru_v1/best_stochastic.pt \
  --tracked-y \
  --timesteps 500000000 \
  --envs 16384 \
  --rollout-steps 128 \
  --device cuda \
  --training-opponents 4 \
  --learning-rate 1e-4 \
  --ent-coef 0.002 \
  --eval-every 50000000 \
  --eval-episodes 64 \
  --eval-workers 24 \
  --save-every 50000000 \
  --output models/distilled_gru_v1_ppo
```

The distillation simulator always uses the full four-opponent traffic
distribution. Because the distilled student should already know how to steer
and avoid traffic, the first fine-tuning experiment should stay on the real
four-opponent distribution rather than dropping back to the one-opponent
exploration curriculum. A lower PPO learning rate and smaller entropy bonus are
used to avoid immediately destroying the distilled behavior.


### Safety-focused privileged teacher fine-tuning

The privileged teacher can now be fine-tuned specifically for reliability before
distillation. Two optional dense shaping terms are available only inside the
training reward:

- `--survival-bonus`: small reward on every non-crash tick.
- `--ttc-risk-penalty`: small penalty when an active opponent is
  longitudinally closing with low time-to-collision while inside or near the
  ego lateral corridor.

Both default to zero, preserving historical runs and direct-policy PPO rewards.

Teacher diagnostics now evaluate the probability-credit scheduler at T=1 on
both the exact original simulator and CUDA simulator. Best-teacher selection
defaults to the exact-original scheduler, with a 5% crash-rate gate:
checkpoints at or below the gate outrank checkpoints above it; above the gate,
crash rate receives a strong distance-equivalent penalty.

Recommended safety fine-tune from the existing best teacher:

```bash
python -m scripts.train.train_teacher_gpu \
  --resume models/gpu_teacher_65k_1b/best_teacher.pt \
  --no-resume-optimizer \
  --timesteps 500000000 \
  --envs 65536 \
  --rollout-steps 128 \
  --device cuda \
  --learning-rate 5e-5 \
  --crash-penalty 400 \
  --survival-bonus 0.005 \
  --ttc-risk-penalty 0.02 \
  --ttc-risk-horizon 120 \
  --ttc-lateral-margin 24 \
  --ent-coef 0.001 \
  --eval-every 25000000 \
  --eval-episodes 256 \
  --eval-workers 24 \
  --save-every 25000000 \
  --selection-mode scheduler \
  --selection-max-crash 0.05 \
  --selection-crash-weight 500000 \
  --output models/gpu_teacher_safe_v2
```

`--timesteps` remains the absolute total stored in the checkpoint, so a
checkpoint already at e.g. 175M steps will run until 500M, not for an
additional 500M.

After training, validate the selected checkpoint over a much larger untouched
seed set before using it for distillation.


### Teacher max-speed safety sweep

Before changing the teacher network, `scripts/eval/evaluate_teacher_speed_caps.py` can measure
the speed/safety Pareto frontier of an existing checkpoint on identical exact
original-simulator seeds. The governor is downstream of the policy: when the
decoded action is `ACCELERATE` and the +0.1 velocity step would cross the cap,
it executes `NOTHING` instead. Steering and braking are never overridden.

Recommended first sweep:

```bash
python -m scripts.eval.evaluate_teacher_speed_caps \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --caps 40 50 60 70 80 100 \
  --episodes 1024 \
  --seed-start 90000 \
  --workers 24 \
  --mode scheduler \
  --temperature 1.0 \
  --json models/teacher_speed_cap_sweep.json \
  --csv models/teacher_speed_cap_sweep.csv
```

The table reports mean/median/p10 distance, crash rate, <25k rate, average
episode speed, and how often the governor vetoed an acceleration. All caps and
the uncapped baseline use the same seeds. Tune on this seed block, then validate
the chosen cap on a separate larger untouched block before adopting it.


### Privileged teacher crash forensics

`scripts/analysis/analyze_teacher_crashes.py` runs the scheduler teacher on the exact original
simulator and retains the final history window for every failed seed. It
classifies crashes into same-lane front/rear, lane-change front/rear, wall, or
unknown failures and records the exact privileged traffic state leading into the
collision.

Recommended analysis run:

```bash
python -m scripts.analysis.analyze_teacher_crashes \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --episodes 2048 \
  --seed-start 120000 \
  --workers 24 \
  --history-ticks 180 \
  --temperature 1.0 \
  --output-prefix models/teacher_crash_analysis
```

Outputs:

- `teacher_crash_analysis_summary.json`: aggregate crash categories, timing,
  speed/lateral-speed statistics, and TTC warning lead times.
- `teacher_crash_analysis_crashes.csv`: one compact row per failed seed for
  sorting/filtering.
- `teacher_crash_analysis.json`: full final 180-tick trajectory for every
  crash, including teacher probabilities, actions, ego state, per-lane traffic,
  relative velocities, and front/rear TTC.

This should be run before further reward changes so teacher fixes target the
actual dominant failure mode rather than assuming excessive absolute speed is
the cause.


### Lane-change initiation forensics and privileged shield

Crash forensics now traces each lane-change failure back to the steering action
that began the maneuver from a settled lane state. At that exact start state it
projects a canonical 48-tick steer + 48-tick counter-steer trajectory while
freezing current opponent longitudinal velocities. The report distinguishes:

- `unsafe_at_start`: the canonical trajectory already intersects a current car
  or wall from information available when the teacher initiated the maneuver.
- `safe_at_start_but_failed`: the start-state projection was clear, so later
  execution or opponent random-walk dynamics caused the failure.
- target-lane front/rear side, start gap/TTC/relative velocity, projected
  longitudinal clearance at ticks 48 and 96, ego speed at initiation, and the
  previous 120 ticks of acceleration/braking.

Re-run the existing forensic command after pulling to obtain these fields.

An experimental privileged shield is also available. It does not retrain the
teacher and never touches legal deployment code. It only vetoes a steering
action that *initiates* a projected-unsafe lane change; counter-steering and
ongoing lateral control are left alone. The fallback is chosen from exact
current-lane TTC: brake for the nearer front threat, accelerate away from the
nearer rear threat, otherwise wait one tick.

Exact-geometry benchmark:

```bash
python -m scripts.eval.evaluate_teacher_lane_shield \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --episodes 1024 \
  --seed-start 140000 \
  --workers 24 \
  --temperature 1.0 \
  --x-margin 0 \
  --y-margin 0 \
  --output models/teacher_lane_shield_exact.json
```

If exact geometry helps, a second conservative benchmark can test the margins
already used by the local privileged controller:

```bash
python -m scripts.eval.evaluate_teacher_lane_shield \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --episodes 1024 \
  --seed-start 140000 \
  --workers 24 \
  --temperature 1.0 \
  --x-margin 55 \
  --y-margin 18 \
  --output models/teacher_lane_shield_margin.json
```


### Per-tick privileged teacher safety filter

The initiation-only lane shield is retained as a diagnostic, but it is not
recommended: on a 2,048-seed exact-original benchmark it intervened on 11.67%
of ticks and increased crash rate from 10.5% to 17.9%. The failure mode was a
blocked steering request repeatedly accumulating scheduler pressure.

`scripts/eval/evaluate_teacher_safety_filter.py` implements the replacement: a
short-horizon **receding-horizon primitive safety filter**.

At every real simulator tick:

1. The normal T=1 probability-credit scheduler proposes one primitive action.
2. That action is projected for a short horizon from exact current state.
3. Projected tick 1 executes the proposed primitive. Later projection ticks use
   a simple emergency recovery controller: damp lateral velocity first, then
   brake for an imminent front threat or accelerate away from an imminent rear
   threat.
4. Existing opponents keep their current vx in the projection. Future
   random-walk draws and future spawns are deliberately not used.
5. If the proposal remains recoverable, it executes unchanged.
6. Only if it projects a collision are the other four primitives projected.
   The highest-probability safe teacher alternative is substituted. If all five
   project unsafe, the action with the latest projected collision / largest
   clearance is used.
7. A vetoed scheduler request is consumed rather than restored, and the
   substituted fallback is not charged to scheduler credit. This prevents the
   blocked-action debt loop seen in the initiation shield.

The filter replans from exact state on the next tick, so it does not assume a
fixed 96-tick lane-change trajectory.

First run a small horizon sweep on identical exact-original seeds:

```bash
python -m scripts.eval.evaluate_teacher_safety_filter \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --horizons 12 24 36 \
  --episodes 512 \
  --seed-start 160000 \
  --workers 24 \
  --temperature 1.0 \
  --x-margin 0 \
  --y-margin 0 \
  --output models/teacher_safety_filter_sweep.json
```

Then validate the best horizon on a larger untouched block, for example:

```bash
python -m scripts.eval.evaluate_teacher_safety_filter \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --horizons 24 \
  --episodes 2048 \
  --seed-start 180000 \
  --workers 24 \
  --temperature 1.0 \
  --x-margin 0 \
  --y-margin 0 \
  --output models/teacher_safety_filter_h24_2k.json
```

The report includes crash/mean/p10/<25k, override rate, all-actions-unsafe rate,
which teacher actions were overridden, which fallbacks replaced them, projected
collision kind/side, action distributions, and time spent at vx=0.


### Sequence-level receding-horizon privileged MPC

The one-primitive short-horizon filter is kept for comparison, but its 512-seed
sweep showed that longer primitive projections became progressively worse. The
problem is structural: one primitive differs too little from the others in this
simulator's lateral-velocity dynamics.

`scripts/eval/evaluate_teacher_mpc.py` implements a maneuver-level replacement around the
same privileged scheduler teacher. A causal traffic trigger invokes planning for
immediate proximity, a closing trajectory reachable within the horizon, or
traffic relevant to an active lateral maneuver; settled traffic moving away from
ego does not trigger it. When triggered, the planner first projects a meaningful
continuation of the teacher's proposal. If that nominal continuation remains
collision-free with a small clearance buffer, the teacher is left unchanged.

Only when the nominal teacher continuation looks unhealthy does the controller
compare the following multi-tick plans:

- `teacher`: meaningful local continuation of the proposed primitive.
- `settle`: drive toward the nearest lane center and cancel lateral velocity.
- `left` / `right`: one-lane bang-bang lateral maneuver with dynamic
  counter-steering.
- `accelerate`: short acceleration burst, then lateral stabilization.
- `brake`: short deceleration burst, then lateral stabilization.
- `coast`: no longitudinal acceleration, with lateral stabilization.

The validated configuration also permits one accepted strategic intervention
per episode. When the normal 72-tick trigger sees lateral relevance but a
119-tick reach reveals closing traffic, the longer projection is accepted only
if the nominal teacher plan predicts a front collision. Long-range rear
predictions caused by a proposed lane move fall back to the normal horizon.
Reports distinguish strategic candidates from accepted interventions. This
catches an otherwise hidden approach early without repeatedly changing an
otherwise good trajectory or reacting prematurely to uncertain adjacent
traffic.

The lateral planner uses steering as acceleration rather than treating LEFT or
RIGHT as direct position commands. For counter-steering, it estimates the
lateral stopping position under the simulator's 0.1 acceleration and targets the
nearest lane center. This avoids discontinuous target changes after small lane
center overshoots.

Each plan starts from exact current ego/opponent state. Nominal opponent vx is
frozen, but collision rectangles are expanded by a configurable statistical
position envelope derived from the simulator's uniform `[-0.1, 0.1]` per-tick
opponent velocity changes. The default is 3 standard deviations. Future traffic
RNG and future spawns are not used. Plans are ranked lexicographically by safety
tier, worst clearance, stopped fraction, lane centering, residual lateral speed,
progress per tick, and teacher agreement. Only the first primitive is executed;
planning starts over from exact state one real tick later.

Validation/tuning sweep:

```bash
python -m scripts.eval.evaluate_teacher_mpc \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --horizons 72 96 120 \
  --burst-ticks 12 \
  --episodes 256 \
  --seed-start 200000 \
  --workers 24 \
  --temperature 1.0 \
  --x-margin 0 \
  --y-margin 0 \
  --trigger-x 900 \
  --trigger-lateral 300 \
  --trigger-immediate 30 \
  --strategic-horizon 119 \
  --strategic-trigger-limit 1 \
  --clearance-threshold 8 \
  --wall-threshold 4 \
  --opponent-uncertainty-sigma 3 \
  --bootstrap-samples 10000 \
  --output models/teacher_mpc_sweep.json
```

The evaluator includes an identical-seed baseline and reports trigger rate,
override rate, crash/mean/p10/<25k, selected plans, overridden teacher actions,
fallback primitives, teacher-projected collision type/side, all-plans-unsafe
rate, time spent at vx=0, lane-straddling rate, trigger reasons, and throughput.
It also reports paired crash rescues/regressions and reproducible 95% bootstrap
intervals for crash-rate, mean-distance, and p10-distance deltas. Reports embed
the Git revision, dirty-tree state, checkpoint SHA-256, dependency versions, and
complete command arguments so ignored files under `models/` remain traceable.

Run the focused deterministic verifier after planner changes:

```bash
python -m scripts.verify.verify_teacher_mpc
```

The validated default is horizon 72 with a 12-tick longitudinal burst, 3-sigma
opponent uncertainty, a 30-pixel immediate-proximity trigger, and the single
front-only strategic confirmation at horizon 119. On the 256 tuning seeds it
reduced crashes from 29 to 0 with no new crashes, raised p10 distance from 23.4k
to 74.0k, increased mean distance by 9.0k, and triggered on 18.2% of ticks.

On the separate 512-seed holdout beginning at seed 300000, crashes fell from 51
to 1 (50 rescued, zero introduced, one crash shared with baseline), with a
paired crash-rate delta of -9.77% (95% bootstrap interval -12.30% to -7.23%).
P10 increased from 32.5k to 64.4k, the below-25k rate fell from 9.6% to 3.5%,
and the trigger rate was 18.7%. Worker throughput was 2,516 ticks/second, about
3.0% below the previous 72-tick implementation and within the 10% promotion
budget. The remaining crash is one tick away when every modeled action is
already unsafe. The MPC remains outside teacher PPO training. It is available
to the separate exact-label distillation path described above; the original
CUDA distillation path still uses the raw PPO scheduler.

The first bounded integration experiment collected 2,002,944 exact MPC labels
and evaluated every 500k transitions on 64 untouched seeds beginning at 400000.
The best deterministic student was the 503,808-label checkpoint, but it still
crashed on 98.4% of those episodes (mean 10.9k, p10 2.9k). A further 1M-label
50% student-behavior dataset-aggregation pass did not improve that safety gate.
Those checkpoints are retained as training evidence but are **not promoted for
deployment**. The result separates two conclusions: the privileged MPC is a
strong exact-state safety teacher, while transferring it into the partially
observed legal policy remains unsolved and needs a stronger closed-loop student
training method.
On the same 64-seed block, the exact MPC teacher had 0% crashes, p10 81.6k,
and mean 162.6k, confirming that the failed gate is transfer quality rather than
the teacher trajectory source.

For deterministic failure inspection, `scripts/analysis/analyze_teacher_mpc_failures.py` replays
baseline and MPC runs and stores the final tick-level window, including ego and
traffic state, scheduler credit, trigger reason, selected plan, every projected
maneuver, and the actual collision geometry:

```bash
python -m scripts.analysis.analyze_teacher_mpc_failures \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --seeds 300397 300417 \
  --output models/teacher_mpc_failure_replays.json
```


### Offline causal replay of teacher crashes

`scripts/analysis/analyze_teacher_counterfactuals.py` turns the crash-forensics output into an
intervention study. It replays only seeds that the baseline teacher crashed on,
checks that the unmodified replay reproduces the original crash tick, and then
changes one mechanism inside a 160-tick window around the baseline fatal event.

The intervention trigger is intentionally taken from the already-observed
baseline failure: fatal maneuver start when available, otherwise TTC<=120 onset,
otherwise 120 ticks before the crash. This makes the tool an offline causal
diagnostic, **not** a deployable controller.

The default interventions are:

- `freeze_noise`: consume all opponent RNG draws but force their random-walk vx
  perturbation to zero during the window.
- `no_spawns`: suppress recycled-car placement during the window while still
  consuming the normal placement RNG draws.
- `speed_floor`: replace DECELERATE with NOTHING only when braking would cross
  a configurable vx floor (default 5).
- `lateral_sign`: preserve the PPO teacher's steering opportunities and
  longitudinal actions, but correct steering direction toward the same baseline
  target lane.
- `lateral_full`: stronger execution intervention; use the target-lane feedback
  controller every tick until the same baseline target lane is settled.
- `safer_target`: at the baseline trigger, choose the safest of
  settle/left/right under a frozen-vx maneuver projection, then remap only PPO
  steering opportunities toward that chosen target.

Start with a small deterministic smoke run. The baseline exact-tick reproduction
should be 100% before interpreting any intervention:

```bash
python -m scripts.analysis.analyze_teacher_counterfactuals \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --crash-analysis models/teacher_crash_analysis_v2.json \
  --max-crashes 20 \
  --workers 20 \
  --window 160 \
  --speed-floor 5 \
  --lane-horizon 96 \
  --output-prefix models/teacher_counterfactuals_smoke
```

Then run the full failed-seed set:

```bash
python -m scripts.analysis.analyze_teacher_counterfactuals \
  models/gpu_teacher_65k_1b/best_teacher.pt \
  --crash-analysis models/teacher_crash_analysis_v2.json \
  --workers 24 \
  --window 160 \
  --speed-floor 5 \
  --lane-horizon 96 \
  --output-prefix models/teacher_counterfactuals
```

The summary reports rescue rate for each intervention, how often the remaining
crashes move earlier/later, mean distance gain, category-specific rescue counts,
and overlap between interventions that rescue the same seeds. Because the input
contains only known baseline failures, rescue rate is a causal diagnostic and
must not be interpreted as a new global crash rate.
