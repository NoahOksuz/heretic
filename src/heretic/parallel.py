# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import atexit
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path

import optuna
import torch
from optuna import Trial, TrialPruned
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from torch import Tensor

from .config import Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model
from .system import empty_cache
from .utils import format_duration, get_trial_parameters, print, print_memory_usage


@dataclass(frozen=True)
class OptimizationArtifacts:
    refusal_directions_path: Path
    base_logprobs_path: Path
    base_refusals_path: Path

    def exists(self) -> bool:
        return (
            self.refusal_directions_path.exists()
            and self.base_logprobs_path.exists()
            and self.base_refusals_path.exists()
        )

    def to_study_attrs(self) -> dict[str, str]:
        return {
            "refusal_directions_path": str(self.refusal_directions_path),
            "base_logprobs_path": str(self.base_logprobs_path),
            "base_refusals_path": str(self.base_refusals_path),
        }

    @classmethod
    def from_study_attrs(cls, attrs: dict[str, str]) -> "OptimizationArtifacts":
        return cls(
            refusal_directions_path=Path(attrs["refusal_directions_path"]),
            base_logprobs_path=Path(attrs["base_logprobs_path"]),
            base_refusals_path=Path(attrs["base_refusals_path"]),
        )


def get_model_checkpoint_slug(model: str) -> str:
    return "".join(
        [(character if (character.isalnum() or character in ["_", "-"]) else "--") for character in model]
    )


def get_artifact_paths(study_checkpoint_dir: str, model: str) -> OptimizationArtifacts:
    slug = get_model_checkpoint_slug(model)
    directory = Path(study_checkpoint_dir)
    return OptimizationArtifacts(
        refusal_directions_path=directory / f"{slug}_refusal_directions.pt",
        base_logprobs_path=directory / f"{slug}_base_logprobs.pt",
        base_refusals_path=directory / f"{slug}_base_refusals.json",
    )


def delete_optimization_artifacts(artifacts: OptimizationArtifacts) -> None:
    for path in (
        artifacts.refusal_directions_path,
        artifacts.base_logprobs_path,
        artifacts.base_refusals_path,
        artifacts.refusal_directions_path.parent / "gpu_assignments.json",
        artifacts.refusal_directions_path.parent / "gpu_claim.lock",
    ):
        if path.exists():
            os.unlink(path)


def resolve_n_workers(settings: Settings) -> int:
    if settings.evaluate_model is not None:
        return 1

    if not torch.cuda.is_available():
        return 1

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return 1

    if settings.n_workers == 0:
        return gpu_count

    if settings.n_workers > gpu_count:
        print(
            f"[yellow]Requested [bold]{settings.n_workers}[/] workers but only "
            f"[bold]{gpu_count}[/] CUDA GPUs are available. Using [bold]{gpu_count}[/] workers.[/]"
        )
        return gpu_count

    return settings.n_workers


def prep_device_map() -> dict[str, int]:
    return {"": 0}


def save_optimization_artifacts(
    artifacts: OptimizationArtifacts,
    refusal_directions: Tensor,
    base_logprobs: Tensor,
    base_refusals: int,
) -> None:
    artifacts.refusal_directions_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refusal_directions.cpu(), artifacts.refusal_directions_path)
    torch.save(base_logprobs.cpu(), artifacts.base_logprobs_path)
    artifacts.base_refusals_path.write_text(json.dumps(base_refusals))


def load_refusal_directions(artifacts: OptimizationArtifacts) -> Tensor:
    return torch.load(artifacts.refusal_directions_path, weights_only=True)


def load_base_logprobs(artifacts: OptimizationArtifacts) -> Tensor:
    return torch.load(artifacts.base_logprobs_path, weights_only=True)


def load_base_refusals(artifacts: OptimizationArtifacts) -> int:
    return json.loads(artifacts.base_refusals_path.read_text())


_gpu_claim_state_path: Path | None = None
_gpu_claim_lock_path: Path | None = None


def _configure_gpu_claim(checkpoint_dir: str) -> None:
    global _gpu_claim_state_path, _gpu_claim_lock_path
    directory = Path(checkpoint_dir)
    _gpu_claim_state_path = directory / "gpu_assignments.json"
    _gpu_claim_lock_path = directory / "gpu_claim.lock"
    directory.mkdir(parents=True, exist_ok=True)
    if not _gpu_claim_state_path.exists():
        _gpu_claim_state_path.write_text("{}")


def _read_gpu_assignments() -> dict[str, int]:
    assert _gpu_claim_state_path is not None
    return json.loads(_gpu_claim_state_path.read_text() or "{}")


def _write_gpu_assignments(assignments: dict[str, int]) -> None:
    assert _gpu_claim_state_path is not None
    _gpu_claim_state_path.write_text(json.dumps(assignments))


def _gpu_claim_lock():
    assert _gpu_claim_lock_path is not None
    return open(_gpu_claim_lock_path, "a+")


def claim_gpu_id(checkpoint_dir: str, n_gpus: int) -> int:
    import fcntl

    _configure_gpu_claim(checkpoint_dir)
    pid = str(os.getpid())

    with _gpu_claim_lock() as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        assignments = _read_gpu_assignments()

        if pid in assignments:
            return assignments[pid]

        used_gpus = set(assignments.values())
        for gpu_id in range(n_gpus):
            if gpu_id not in used_gpus:
                assignments[pid] = gpu_id
                _write_gpu_assignments(assignments)
                atexit.register(release_gpu_claim, checkpoint_dir)
                return gpu_id

    raise RuntimeError(f"No free GPU available for worker (requested {n_gpus} workers).")


def release_gpu_claim(checkpoint_dir: str) -> None:
    import fcntl

    _configure_gpu_claim(checkpoint_dir)
    pid = str(os.getpid())

    try:
        with _gpu_claim_lock() as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            assignments = _read_gpu_assignments()
            if pid in assignments:
                del assignments[pid]
                _write_gpu_assignments(assignments)
    except OSError:
        pass


def print_worker(gpu_id: int, message: str) -> None:
    print(f"[grey50][GPU {gpu_id}][/] {message}")


class WorkerContext:
    def __init__(self, trial: Trial, n_workers: int):
        study = trial.study
        self.settings = Settings.model_validate_json(study.user_attrs["settings"])
        self.n_workers = n_workers
        self.n_trials = self.settings.n_trials
        self.artifacts = OptimizationArtifacts.from_study_attrs(
            study.user_attrs["optimization_artifacts"]
        )
        self.start_time = study.user_attrs["optimize_start_time"]

        checkpoint_dir = str(self.artifacts.refusal_directions_path.parent)
        self.gpu_id = claim_gpu_id(checkpoint_dir, n_workers)
        self.settings.device_map = {"": self.gpu_id}

        print_worker(self.gpu_id, "Loading model...")
        self.model = Model(self.settings)
        self.refusal_directions = load_refusal_directions(self.artifacts)
        self.evaluator = Evaluator(
            self.settings,
            self.model,
            base_logprobs=load_base_logprobs(self.artifacts),
            base_refusals=load_base_refusals(self.artifacts),
        )

    def run_trial(self, trial: Trial) -> tuple[float, float]:
        from dataclasses import asdict

        trial_number = trial.number + 1
        trial.set_user_attr("index", trial_number)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(self.model.get_layers()) - 1

        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in self.model.get_abliterable_components():
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer_index,
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {key: asdict(value) for key, value in parameters.items()})

        completed_trials = sum(
            1 for study_trial in trial.study.trials if study_trial.state == TrialState.COMPLETE
        )

        print_worker(
            self.gpu_id,
            f"Running trial [bold]{trial_number}[/] ([bold]{completed_trials + 1}[/] completed, "
            f"[bold]{self.n_trials}[/] target)...",
        )
        print_worker(self.gpu_id, "* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print_worker(self.gpu_id, f"  * {name} = [bold]{value}[/]")
        print_worker(self.gpu_id, "* Resetting model...")
        self.model.reset_model()
        print_worker(self.gpu_id, "* Abliterating...")
        self.model.abliterate(self.refusal_directions, direction_index, parameters)
        print_worker(self.gpu_id, "* Evaluating...")
        score, kl_divergence, refusals = self.evaluator.get_score()

        elapsed_time = time.perf_counter() - self.start_time
        print_worker(
            self.gpu_id,
            f"Elapsed time: [bold]{format_duration(elapsed_time)}[/]",
        )
        print_memory_usage()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)
        trial.set_user_attr("base_refusals", self.evaluator.base_refusals)
        trial.set_user_attr("n_bad_prompts", len(self.evaluator.bad_prompts))

        return score


_worker_ctx: WorkerContext | None = None
_worker_n_workers: int = 1


def configure_parallel_workers(n_workers: int) -> None:
    global _worker_ctx, _worker_n_workers
    _worker_ctx = None
    _worker_n_workers = n_workers


def parallel_objective(trial: Trial) -> tuple[float, float]:
    global _worker_ctx
    if _worker_ctx is None:
        _worker_ctx = WorkerContext(trial, _worker_n_workers)
    return _worker_ctx.run_trial(trial)


def parallel_objective_wrapper(trial: Trial) -> tuple[float, float]:
    try:
        return parallel_objective(trial)
    except KeyboardInterrupt:
        trial.study.stop()
        raise TrialPruned()


def _optimization_worker(args: tuple[str, int]) -> None:
    study_checkpoint_file, n_workers = args
    configure_parallel_workers(n_workers)

    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)
    worker_study = optuna.load_study(study_name="heretic", storage=storage)
    target = Settings.model_validate_json(worker_study.user_attrs["settings"]).n_trials

    while len(worker_study.trials) < target:
        try:
            worker_study.optimize(parallel_objective_wrapper, n_trials=1, n_jobs=1)
        except KeyboardInterrupt:
            worker_study.stop()
            break


def run_parallel_optimize(
    study: optuna.Study,
    study_checkpoint_file: str,
    n_trials: int,
    n_workers: int,
) -> None:
    if n_trials <= 0:
        return

    configure_parallel_workers(n_workers)

    if n_workers <= 1:
        try:
            study.optimize(parallel_objective_wrapper, n_trials=n_trials, n_jobs=1)
        except KeyboardInterrupt:
            pass
        return

    ctx = multiprocessing.get_context("spawn")
    try:
        with ctx.Pool(processes=n_workers) as pool:
            pool.map(
                _optimization_worker,
                [(study_checkpoint_file, n_workers)] * n_workers,
            )
    except KeyboardInterrupt:
        study.stop()


def reload_model_for_ui(settings: Settings, artifacts: OptimizationArtifacts) -> tuple[Model, Tensor, Evaluator]:
    ui_settings = settings.model_copy(deep=True)
    ui_settings.device_map = prep_device_map()
    model = Model(ui_settings)
    refusal_directions = load_refusal_directions(artifacts)
    evaluator = Evaluator(
        ui_settings,
        model,
        base_logprobs=load_base_logprobs(artifacts),
        base_refusals=load_base_refusals(artifacts),
    )
    return model, refusal_directions, evaluator


def free_parallel_workers() -> None:
    global _worker_ctx
    if _worker_ctx is not None:
        del _worker_ctx.model, _worker_ctx.evaluator, _worker_ctx.refusal_directions
        _worker_ctx = None
    empty_cache()
