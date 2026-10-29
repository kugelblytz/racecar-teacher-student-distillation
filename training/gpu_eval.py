"""Evaluation helpers for CUDA-trained privileged teachers.

Original-simulator evaluation is CPU-bound, so complete episodes are distributed
across spawn-based worker processes. Each worker owns one original simulator and
one CPU copy of the policy. Spawn is intentional: forking after CUDA has been
initialized is unsafe.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Literal, Sequence, Tuple

# Set these before importing pygame through teacher_env. Spawned evaluation
# workers import this module from scratch.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, MAX_TICKS, OBS_SIZE, GpuRaceCarBatch
from training.teacher_env import PrivilegedRaceCarEnv


PolicyMode = Literal["deterministic", "stochastic", "scheduler"]


def cpu_copy(model: ActorCritic) -> ActorCritic:
    clone = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False).cpu()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    clone.load_state_dict(state)
    clone.eval()
    return clone


def _summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    distances = np.asarray([float(r["distance"]) for r in rows], dtype=np.float64)
    speeds = np.asarray([float(r["final_speed"]) for r in rows], dtype=np.float64)
    ticks = np.asarray([int(r["ticks"]) for r in rows], dtype=np.float64)
    overrides = np.asarray(
        [int(r.get("speed_cap_overrides", 0)) for r in rows],
        dtype=np.float64,
    )
    crashes = sum(int(bool(r["crashed"])) for r in rows)
    n = max(1, len(rows))
    total_ticks = float(ticks.sum())

    return {
        "episodes": len(rows),
        "mean_distance": float(distances.mean()) if len(rows) else 0.0,
        "median_distance": float(np.median(distances)) if len(rows) else 0.0,
        "p10_distance": float(np.percentile(distances, 10)) if len(rows) else 0.0,
        "min_distance": float(distances.min()) if len(rows) else 0.0,
        "max_distance": float(distances.max()) if len(rows) else 0.0,
        "crash_rate": float(crashes / n),
        "mean_final_speed": float(speeds.mean()) if len(rows) else 0.0,
        "mean_episode_speed": (
            float(np.mean(distances / ticks)) if len(rows) else 0.0
        ),
        "speed_cap_overrides": int(overrides.sum()),
        "speed_cap_override_tick_rate": (
            float(overrides.sum() / total_ticks) if total_ticks > 0 else 0.0
        ),
        "below_10k_rate": float((distances < 10_000).mean()) if len(rows) else 0.0,
        "below_25k_rate": float((distances < 25_000).mean()) if len(rows) else 0.0,
        "below_50k_rate": float((distances < 50_000).mean()) if len(rows) else 0.0,
        "below_100k_rate": float((distances < 100_000).mean()) if len(rows) else 0.0,
        "rows": rows,
    }


def _cpu_action(
    model: ActorCritic,
    obs_t: torch.Tensor,
    *,
    mode: PolicyMode,
    generator: torch.Generator,
    temperature: float = 1.0,
    scheduler_credit: torch.Tensor | None = None,
) -> int:
    with torch.no_grad():
        logits, _ = model(obs_t)

    if mode == "deterministic":
        return int(logits.argmax(dim=-1).item())

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = torch.softmax(logits / float(temperature), dim=-1)

    if mode == "scheduler":
        if scheduler_credit is None:
            raise ValueError("scheduler mode requires scheduler_credit")
        scheduler_credit.add_(probs.squeeze(0))
        action = int(scheduler_credit.argmax().item())
        scheduler_credit[action] -= 1.0
        return action

    return int(torch.multinomial(probs, 1, generator=generator).item())


def _run_original_episode(
    model: ActorCritic,
    env: PrivilegedRaceCarEnv,
    *,
    mode: PolicyMode,
    sim_seed: int,
    action_seed: int,
    temperature: float = 1.0,
    speed_cap: float | None = None,
) -> Dict[str, object]:
    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(int(action_seed))

    # Upstream randomizer.seed() prints two lines on every reset.
    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(options={"sim_seed": int(sim_seed)})

    terminated = False
    truncated = False
    scheduler_credit = (
        torch.zeros(len(ACTIONS), dtype=torch.float32)
        if mode == "scheduler"
        else None
    )
    speed_cap_overrides = 0

    while not (terminated or truncated):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action = _cpu_action(
            model,
            obs_t,
            mode=mode,
            generator=action_generator,
            temperature=temperature,
            scheduler_credit=scheduler_credit,
        )

        # Diagnostic inference-time speed governor. Only suppress ACCELERATE;
        # steering/braking decisions are never overridden. Ego vx has no passive
        # drift, so preventing an acceleration step from crossing the cap gives
        # an exact maximum-speed limit without changing the teacher network.
        if speed_cap is not None:
            if speed_cap <= 0:
                raise ValueError("speed_cap must be positive")
            if (
                action == 1
                and float(info["speed"]) + 0.1 > float(speed_cap) + 1e-9
            ):
                # _cpu_action already debited the scheduler's proposed action.
                # If the governor executes NOTHING instead, make scheduler
                # credit reflect the action that actually reached the simulator.
                if mode == "scheduler" and scheduler_credit is not None:
                    scheduler_credit[action] += 1.0
                    scheduler_credit[0] -= 1.0
                action = 0  # NOTHING
                speed_cap_overrides += 1

        obs, _, terminated, truncated, info = env.step(action)

    return {
        "seed": int(sim_seed),
        "distance": float(info["distance"]),
        "ticks": int(info["ticks"]),
        "crashed": bool(info["crashed"]),
        "final_speed": float(info["speed"]),
        "speed_cap": None if speed_cap is None else float(speed_cap),
        "speed_cap_overrides": int(speed_cap_overrides),
    }


# Per-process state for original-simulator evaluation workers.
_WORKER_MODEL: ActorCritic | None = None
_WORKER_ENV: PrivilegedRaceCarEnv | None = None


def _original_worker_init(
    model_state: Dict[str, torch.Tensor],
    crash_penalty: float,
    reward_scale: float,
) -> None:
    """Initialize one CPU model and one original simulator per process."""
    global _WORKER_MODEL, _WORKER_ENV

    # Do not let each process create its own OpenMP thread team. Parallelism is
    # across evaluation processes, not inside the tiny MLP.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows setting this once in a process.
        pass

    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False).cpu()
    model.load_state_dict(model_state)
    model.eval()

    _WORKER_MODEL = model
    _WORKER_ENV = PrivilegedRaceCarEnv(
        base_seed=0,
        crash_penalty=float(crash_penalty),
        reward_scale=float(reward_scale),
    )


def _original_worker_task(
    task: Tuple[str, PolicyMode, int, int, float, float | None],
) -> Tuple[str, Dict[str, object]]:
    label, mode, sim_seed, action_seed, temperature, speed_cap = task

    if _WORKER_MODEL is None or _WORKER_ENV is None:
        raise RuntimeError("Original-evaluation worker was not initialized")

    row = _run_original_episode(
        _WORKER_MODEL,
        _WORKER_ENV,
        mode=mode,
        sim_seed=sim_seed,
        action_seed=action_seed,
        temperature=temperature,
        speed_cap=speed_cap,
    )
    return label, row


def _resolve_original_workers(requested: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if requested > 0:
        return max(1, min(int(requested), task_count))

    # Auto mode. Cap at 24 because spawning many Python+PyTorch processes beyond
    # that tends to add memory/startup cost without helping this small simulator.
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, task_count, 24))


def _evaluate_original_variants(
    model: ActorCritic,
    *,
    episodes: int,
    seed_start: int,
    variants: Sequence[Tuple[str, PolicyMode, float, float | None]],
    crash_penalty: float,
    reward_scale: float,
    action_seed_start: int,
    workers: int,
) -> Dict[str, Dict[str, object]]:
    """Evaluate multiple policy variants in one shared CPU process pool."""
    for label, mode, temperature, speed_cap in variants:
        if mode not in ("deterministic", "stochastic", "scheduler"):
            raise ValueError(f"Unknown policy mode: {mode}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive for {label}")
        if speed_cap is not None and speed_cap <= 0:
            raise ValueError(f"speed_cap must be positive for {label}")

    episode_count = int(episodes)
    if episode_count <= 0:
        raise ValueError("episodes must be positive")

    tasks: List[
        Tuple[str, PolicyMode, int, int, float, float | None]
    ] = []
    for label, mode, temperature, speed_cap in variants:
        for i in range(episode_count):
            tasks.append(
                (
                    label,
                    mode,
                    int(seed_start) + i,
                    int(action_seed_start) + i,
                    float(temperature),
                    None if speed_cap is None else float(speed_cap),
                )
            )

    worker_count = _resolve_original_workers(int(workers), len(tasks))
    grouped: Dict[str, List[Dict[str, object]]] = {
        label: [] for label, _, _, _ in variants
    }

    if worker_count == 1:
        eval_model = cpu_copy(model)
        env = PrivilegedRaceCarEnv(
            base_seed=0,
            crash_penalty=crash_penalty,
            reward_scale=reward_scale,
        )
        try:
            for (
                label,
                mode,
                sim_seed,
                action_seed,
                temperature,
                speed_cap,
            ) in tasks:
                grouped[label].append(
                    _run_original_episode(
                        eval_model,
                        env,
                        mode=mode,
                        sim_seed=sim_seed,
                        action_seed=action_seed,
                        temperature=temperature,
                        speed_cap=speed_cap,
                    )
                )
        finally:
            env.close()
    else:
        model_state = {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        }
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_original_worker_init,
            initargs=(model_state, crash_penalty, reward_scale),
        ) as executor:
            for label, row in executor.map(
                _original_worker_task,
                tasks,
                chunksize=1,
            ):
                grouped[label].append(row)

    results: Dict[str, Dict[str, object]] = {}
    for label, mode, temperature, speed_cap in variants:
        rows = sorted(grouped[label], key=lambda row: int(row["seed"]))
        result = _summarize_rows(rows)
        result["simulator"] = "original"
        result["policy_mode"] = mode
        result["temperature"] = float(temperature)
        result["speed_cap"] = (
            None if speed_cap is None else float(speed_cap)
        )
        result["workers"] = worker_count
        results[label] = result

    return results


def _evaluate_original_modes(
    model: ActorCritic,
    *,
    episodes: int,
    seed_start: int,
    modes: Sequence[PolicyMode],
    crash_penalty: float,
    reward_scale: float,
    action_seed_start: int,
    workers: int,
) -> Dict[PolicyMode, Dict[str, object]]:
    variants = [(mode, mode, 1.0, None) for mode in modes]
    results = _evaluate_original_variants(
        model,
        episodes=episodes,
        seed_start=seed_start,
        variants=variants,
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
        action_seed_start=action_seed_start,
        workers=workers,
    )
    return {mode: results[mode] for mode in modes}


def evaluate_original_temperature_sweep(
    model: ActorCritic,
    *,
    temperatures: Sequence[float],
    episodes: int,
    seed_start: int,
    crash_penalty: float = 250.0,
    reward_scale: float = 1000.0,
    action_seed_start: int = 1_000_000,
    workers: int = 0,
    include_deterministic: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Evaluate stochastic inference temperatures on identical original seeds."""
    variants: List[
        Tuple[str, PolicyMode, float, float | None]
    ] = []
    if include_deterministic:
        variants.append(("deterministic", "deterministic", 1.0, None))

    for temperature in temperatures:
        t = float(temperature)
        variants.append((f"temperature_{t:g}", "stochastic", t, None))

    return _evaluate_original_variants(
        model,
        episodes=episodes,
        seed_start=seed_start,
        variants=variants,
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
        action_seed_start=action_seed_start,
        workers=workers,
    )



def evaluate_original_speed_cap_sweep(
    model: ActorCritic,
    *,
    speed_caps: Sequence[float],
    episodes: int,
    seed_start: int,
    crash_penalty: float = 250.0,
    reward_scale: float = 1000.0,
    action_seed_start: int = 1_000_000,
    workers: int = 0,
    policy_mode: PolicyMode = "scheduler",
    temperature: float = 1.0,
    include_uncapped: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Evaluate inference-time speed caps on identical exact-original seeds.

    The cap is a controller wrapper, not a retrained policy: if the decoded
    action is ACCELERATE and that +0.1 step would exceed the cap, execute
    NOTHING instead. Steering and braking are never overridden.
    """
    if policy_mode not in ("deterministic", "stochastic", "scheduler"):
        raise ValueError(f"Unknown policy mode: {policy_mode}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    variants: List[
        Tuple[str, PolicyMode, float, float | None]
    ] = []
    if include_uncapped:
        variants.append(("uncapped", policy_mode, float(temperature), None))

    seen = set()
    for cap in speed_caps:
        cap_f = float(cap)
        if cap_f <= 0:
            raise ValueError("all speed caps must be positive")
        if cap_f in seen:
            continue
        seen.add(cap_f)
        variants.append(
            (f"cap_{cap_f:g}", policy_mode, float(temperature), cap_f)
        )

    if not variants:
        raise ValueError("provide at least one speed cap or include uncapped")

    return _evaluate_original_variants(
        model,
        episodes=episodes,
        seed_start=seed_start,
        variants=variants,
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
        action_seed_start=action_seed_start,
        workers=workers,
    )


def evaluate_original_scheduler_sweep(
    model: ActorCritic,
    *,
    temperatures: Sequence[float],
    episodes: int,
    seed_start: int,
    crash_penalty: float = 250.0,
    reward_scale: float = 1000.0,
    action_seed_start: int = 1_000_000,
    workers: int = 0,
    include_stochastic_t1: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Compare argmax, normal stochastic sampling, and deterministic scheduling.

    Scheduler state is reset to zero at the start of every episode. At each tick:

        credit += softmax(logits / temperature)
        action = argmax(credit)
        credit[action] -= 1

    This preserves the policy's requested long-run action proportions without
    injecting action-sampling randomness.
    """
    variants: List[
        Tuple[str, PolicyMode, float, float | None]
    ] = [
        ("deterministic", "deterministic", 1.0, None),
    ]
    if include_stochastic_t1:
        variants.append(("stochastic_t1", "stochastic", 1.0, None))

    for temperature in temperatures:
        t = float(temperature)
        variants.append((f"scheduler_{t:g}", "scheduler", t, None))

    return _evaluate_original_variants(
        model,
        episodes=episodes,
        seed_start=seed_start,
        variants=variants,
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
        action_seed_start=action_seed_start,
        workers=workers,
    )

def evaluate_on_original(
    model: ActorCritic,
    *,
    episodes: int,
    seed_start: int,
    crash_penalty: float = 250.0,
    reward_scale: float = 1000.0,
    policy_mode: PolicyMode = "deterministic",
    action_seed_start: int = 1_000_000,
    workers: int = 0,
) -> Dict[str, object]:
    """Evaluate one policy mode on the exact original headless simulator.

    workers=0 automatically uses up to 24 logical CPUs. workers=1 forces the
    previous serial behavior.
    """
    return _evaluate_original_modes(
        model,
        episodes=episodes,
        seed_start=seed_start,
        modes=(policy_mode,),
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
        action_seed_start=action_seed_start,
        workers=workers,
    )[policy_mode]


@torch.no_grad()
def evaluate_on_gpu_sim(
    model: ActorCritic,
    *,
    episodes: int,
    seed: int,
    crash_penalty: float = 250.0,
    reward_scale: float = 1000.0,
    policy_mode: PolicyMode = "deterministic",
    action_seed: int = 2_000_000,
) -> Dict[str, object]:
    """Evaluate one first episode per CUDA environment.

    This is distributional validation, not a same-seed comparison with the
    original simulator because the CUDA environment intentionally uses torch RNG.
    """
    if policy_mode not in ("deterministic", "stochastic", "scheduler"):
        raise ValueError(f"Unknown policy mode: {policy_mode}")

    device = next(model.parameters()).device
    eval_model = model
    eval_model.eval()

    sim = GpuRaceCarBatch(
        num_envs=int(episodes),
        device=device,
        seed=int(seed),
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
    )
    obs = sim.observation()

    pending = torch.ones(int(episodes), dtype=torch.bool, device=device)
    distances = torch.zeros(int(episodes), dtype=torch.float64, device=device)
    ticks = torch.zeros(int(episodes), dtype=torch.int32, device=device)
    crashed = torch.zeros(int(episodes), dtype=torch.bool, device=device)
    speeds = torch.zeros(int(episodes), dtype=torch.float64, device=device)

    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(int(action_seed))
    scheduler_credit = torch.zeros(
        (int(episodes), len(ACTIONS)),
        dtype=torch.float32,
        device=device,
    )

    # Every initial episode must finish by MAX_TICKS. Environments that finish
    # earlier auto-reset internally; pending makes us record only their first run.
    for _ in range(MAX_TICKS):
        logits, _ = eval_model(obs)
        if policy_mode == "deterministic":
            actions = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            if policy_mode == "scheduler":
                scheduler_credit.add_(probs)
                actions = scheduler_credit.argmax(dim=-1)
                scheduler_credit.sub_(
                    torch.nn.functional.one_hot(
                        actions, num_classes=len(ACTIONS)
                    ).to(scheduler_credit.dtype)
                )
            else:
                actions = torch.multinomial(
                    probs, 1, replacement=True, generator=action_generator
                ).squeeze(1)

        obs, _, _, episode = sim.step(actions)
        scheduler_credit *= (~episode.done).to(torch.float32).unsqueeze(1)

        first_done = pending & episode.done
        distances = torch.where(first_done, episode.distance, distances)
        ticks = torch.where(first_done, episode.ticks, ticks)
        crashed = torch.where(first_done, episode.crashed, crashed)
        speeds = torch.where(first_done, episode.final_speed, speeds)
        pending &= ~first_done

    if bool(pending.any().item()):
        raise RuntimeError("Some GPU evaluation episodes did not finish by MAX_TICKS")

    distances_np = distances.cpu().numpy()
    ticks_np = ticks.cpu().numpy()
    crashed_np = crashed.cpu().numpy()
    speeds_np = speeds.cpu().numpy()

    rows: List[Dict[str, object]] = [
        {
            "seed": int(seed) + i,
            "distance": float(distances_np[i]),
            "ticks": int(ticks_np[i]),
            "crashed": bool(crashed_np[i]),
            "final_speed": float(speeds_np[i]),
        }
        for i in range(int(episodes))
    ]

    result = _summarize_rows(rows)
    result["simulator"] = "gpu"
    result["policy_mode"] = policy_mode
    result["workers"] = 0
    return result


def evaluate_policy_diagnostics(
    model: ActorCritic,
    *,
    episodes: int,
    seed_start: int,
    crash_penalty: float = 250.0,
    reward_scale: float = 1000.0,
    action_seed_start: int = 1_000_000,
    gpu_seed: int = 3_000_000,
    original_workers: int = 0,
) -> Dict[str, Dict[str, object]]:
    """Run exact-original and CUDA diagnostics for all deployment decoders."""
    original = _evaluate_original_modes(
        model,
        episodes=episodes,
        seed_start=seed_start,
        modes=("deterministic", "stochastic", "scheduler"),
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
        action_seed_start=action_seed_start,
        workers=original_workers,
    )

    return {
        "original_deterministic": original["deterministic"],
        "original_stochastic": original["stochastic"],
        "original_scheduler": original["scheduler"],
        "gpu_deterministic": evaluate_on_gpu_sim(
            model,
            episodes=episodes,
            seed=gpu_seed,
            crash_penalty=crash_penalty,
            reward_scale=reward_scale,
            policy_mode="deterministic",
            action_seed=action_seed_start,
        ),
        "gpu_stochastic": evaluate_on_gpu_sim(
            model,
            episodes=episodes,
            seed=gpu_seed,
            crash_penalty=crash_penalty,
            reward_scale=reward_scale,
            policy_mode="stochastic",
            action_seed=action_seed_start,
        ),
        "gpu_scheduler": evaluate_on_gpu_sim(
            model,
            episodes=episodes,
            seed=gpu_seed,
            crash_penalty=crash_penalty,
            reward_scale=reward_scale,
            policy_mode="scheduler",
            action_seed=action_seed_start,
        ),
    }
