import csv
import time
import importlib.util
import os
import random
from pathlib import Path
import json

import torch
from torch.utils.data import Subset

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from sampler import (
    random_sample_indices,
    carbon_proportional_random_indices,
    carbon_proportional_entropy_indices,
    carbon_proportional_entropy_low_indices,
    carbon_proportional_gradient_indices,
    carbon_proportional_scheduled_indices,
    carbon_proportional_protected_indices,
    _carbon_proportional_count,
    carbon_proportional_gradient_pool,
    carbon_proportional_entropy_low_pool,
    carbon_proportional_entropy_pool,
    carbon_proportional_rarity_indices,
    fixed_count_indices,
    
)
from task import Net, load_partition_datasets, make_dataloader
from task import train as train_fn
from task import test


# Load private carbon intensities from project directory (not bundled by Flower)
_carbon_path = os.path.join(os.path.expanduser("~"), "tu-fl-thesis-carbon-aware-data-sampling", "carbon_intensities.py")
_spec = importlib.util.spec_from_file_location("carbon_intensities", _carbon_path)
_carbon_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_carbon_module)
CARBON_INTENSITY_SCENARIOS = _carbon_module.CARBON_INTENSITY_SCENARIOS
_skew_path = os.path.join(os.path.expanduser("~"), "tu-fl-thesis-carbon-aware-data-sampling", "carbon_intensities_skew.py")
_skew_spec = importlib.util.spec_from_file_location("carbon_intensities_skew", _skew_path)
_skew_module = importlib.util.module_from_spec(_skew_spec)
_skew_spec.loader.exec_module(_skew_module)
CARBON_INTENSITY_SCENARIOS_SKEW = _skew_module.CARBON_INTENSITY_SCENARIOS_SKEW


def append_client_metrics(results_dir: str, client_id: int, row: dict) -> None:
    client_log_dir = Path(results_dir) / "client_logs"
    client_log_dir.mkdir(parents=True, exist_ok=True)

    output_path = client_log_dir / f"client_{client_id}.csv"
    file_exists = output_path.exists()

    fieldnames = [
        "round",
        "client_id",
        "sampling_method",
        "warmup_active",
        "sampling_ratio",
        "full_dataset_size",
        "sampled_dataset_size",
        "sampling_time_s",
        "training_time_s",
        "client_total_time_s",
        "samples_per_second",
        "train_loss",
        "carbon_intensity",
        "power_watts",
        "energy_kwh",
        "carbon_g",
    ]

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
 
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    power_watts = float(context.run_config.get("power-watts", 50.0))
    partition_method = str(context.run_config.get("partition-method", "dirichlet"))
    classes_per_partition = int(context.run_config.get("classes-per-partition", 8))
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 0.5))
    equal_share_count = int(context.run_config.get("equal-share-count", 0))
    excluded_clients = str(context.run_config.get("excluded-clients", ""))
    excluded_set = {int(x) for x in excluded_clients.split(",") if x.strip() != ""}
 
    lr = float(msg.content["config"]["lr"])
    server_round = int(msg.content["config"].get("server-round", 1))
    warmup_rounds = int(msg.content["config"].get("warmup_rounds", 0))
    sampling_ratio = float(msg.content["config"].get("sampling_ratio", 1.0))
    sampling_method = str(msg.content["config"].get("sampling_method", "full"))
    alpha = float(msg.content["config"].get("alpha", 0.9))
    m_min = int(msg.content["config"].get("m_min", 300))
    results_dir = str(msg.content["config"].get("results_dir", "results"))
 
    # ---- NEW: phase-switch config ----
    early_method = str(msg.content["config"].get("early_method", ""))
    phase_switch_threshold = float(msg.content["config"].get("phase_switch_threshold", 0.0))
    phase_active = phase_switch_threshold > 0.0
 
    # read the server's phase signal (written in global_evaluate the round before)
    phase_switched = False
    if phase_active:
        phase_path = Path(results_dir) / "phase.json"
        if phase_path.exists():
            try:
                phase_switched = json.load(open(phase_path)).get("switched", False)
            except Exception:
                phase_switched = False
        # choose the effective method for THIS round
        if phase_switched:
            sampling_method = "full"        # switched: everyone trains on full data
        else:
            sampling_method = early_method  # early phase: use the early method
    # ---- END NEW ----
 
    flagged_classes = []
    flag_path = Path(results_dir) / "flagged_classes.json"
    if flag_path.exists():
        with open(flag_path) as f:
            flagged_classes = json.load(f)
 
    carbon_scenario = str(context.run_config.get("carbon-scenario", "high"))
    if carbon_scenario in ("pathological_skew", "skew_dirichlet"):
        CLIENT_CARBON_INTENSITY = CARBON_INTENSITY_SCENARIOS_SKEW[carbon_scenario]
    else:
        CLIENT_CARBON_INTENSITY = CARBON_INTENSITY_SCENARIOS[carbon_scenario]
 
    carbon_intensity = CLIENT_CARBON_INTENSITY.get(partition_id, 300.0)
 
    # ---- NEW: exclusion applies only during the EARLY phase ----
    apply_exclusion = partition_id in excluded_set
    if phase_active and phase_switched:
        apply_exclusion = False   # after the switch, no client is excluded
    # ---- END NEW ----
 
    if apply_exclusion:
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": 0.0, "num-examples": 0,
            "client_id": int(partition_id), "server_round": int(server_round),
            "full_dataset_size": 0, "sampled_dataset_size": 0,
            "warmup_active": 0, "sampling_ratio": 0.0,
            "sampling_time_s": 0.0, "training_time_s": 0.0,
            "client_total_time_s": 0.0, "samples_per_second": 0.0,
            "carbon_intensity": float(carbon_intensity), "power_watts": float(power_watts),
            "energy_kwh": 0.0, "carbon_g": 0.0,
        }
        content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)
 
    warmup_active = server_round <= warmup_rounds
 
    start_total = time.perf_counter()
    trainset, _ = load_partition_datasets(
        partition_id, num_partitions,
        partition_method=partition_method,
        alpha=dirichlet_alpha,
        classes_per_partition=classes_per_partition,
    )
 
    start_sampling = time.perf_counter()
 
    if warmup_active or sampling_method == "full":
        selected_trainset = trainset
        effective_ratio = 1.0
 
    elif sampling_method == "random":
        selected_indices = random_sample_indices(
            dataset_size=len(trainset),
            ratio=sampling_ratio,
            seed=server_round + partition_id,
        )
        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = sampling_ratio
 
    elif sampling_method == "carbon_proportional_random":
        selected_indices = carbon_proportional_random_indices(
            dataset_size=len(trainset),
            carbon_intensity=carbon_intensity,
            alpha=alpha,
            m_min=m_min,
            max_intensity=1000.0,
            seed=server_round + partition_id,
        )
        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = len(selected_indices) / len(trainset)
 
    elif sampling_method == "carbon_proportional_scheduled":
        transition_round = int(msg.content["config"].get("transition_round", 10))
        selected_indices = carbon_proportional_scheduled_indices(
            dataset_size=len(trainset),
            carbon_intensity=carbon_intensity,
            alpha=alpha,
            m_min=m_min,
            server_round=server_round,
            transition_round=transition_round,
            max_intensity=1000.0,
            seed=server_round + partition_id,
        )
        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = len(selected_indices) / len(trainset)
 
    elif sampling_method == "carbon_proportional_protected":
        selected_indices = carbon_proportional_protected_indices(
            trainset=trainset,
            dataset_size=len(trainset),
            carbon_intensity=carbon_intensity,
            alpha=alpha,
            m_min=m_min,
            flagged_classes=flagged_classes,
            protection_cap=0.5,
            label_key="character",
            seed=server_round + partition_id,
        )
        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = len(selected_indices) / len(trainset)

    elif sampling_method in ("carbon_proportional_rarity", "carbon_proportional_rarity_only"):
        # Only HIGH-carbon clients use rarity selection; low/medium carbon use
        # plain carbon-proportional random.
        HIGH_CARBON_THRESHOLD = 400.0

        if carbon_intensity <= HIGH_CARBON_THRESHOLD:
            selected_indices = carbon_proportional_random_indices(
                dataset_size=len(trainset),
                carbon_intensity=carbon_intensity,
                alpha=alpha,
                m_min=m_min,
                max_intensity=1000.0,
                seed=server_round + partition_id,
            )
        else:
            gamma = float(msg.content["config"].get("rarity_gamma", 0.6))
            use_diff = (sampling_method == "carbon_proportional_rarity")
            scoring_loader = make_dataloader(trainset, batch_size=batch_size, shuffle=False)
            selected_indices = carbon_proportional_rarity_indices(
                model=model,
                scoring_loader=scoring_loader,
                dataset_size=len(trainset),
                carbon_intensity=carbon_intensity,
                alpha=alpha,
                m_min=m_min,
                device=device,
                gamma=gamma,
                max_intensity=1000.0,
                use_difficulty=use_diff,
                label_key="character",
                seed=server_round + partition_id,
            )

        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = len(selected_indices) / len(trainset)

    elif sampling_method == "carbon_proportional_rarity_burst":
        # read whether the rarity burst is active this round
        burst_active = False
        phase_path = Path(results_dir) / "phase.json"
        if phase_path.exists():
            try:
                burst_active = json.load(open(phase_path)).get("burst_active", False)
            except Exception:
                burst_active = False

        if burst_active:
            # rarity selection during the burst window
            gamma = float(msg.content["config"].get("rarity_gamma", 1.0))
            scoring_loader = make_dataloader(trainset, batch_size=batch_size, shuffle=False)
            selected_indices = carbon_proportional_rarity_indices(
                model=model,
                scoring_loader=scoring_loader,
                dataset_size=len(trainset),
                carbon_intensity=carbon_intensity,
                alpha=alpha,
                m_min=m_min,
                device=device,
                gamma=gamma,
                max_intensity=1000.0,
                use_difficulty=True,
                label_key="character",
                seed=server_round + partition_id,
            )
        else:
            # outside the burst: plain carbon-proportional random (cheap)
            selected_indices = carbon_proportional_random_indices(
                dataset_size=len(trainset),
                carbon_intensity=carbon_intensity,
                alpha=alpha,
                m_min=m_min,
                max_intensity=1000.0,
                seed=server_round + partition_id,
            )

        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = len(selected_indices) / len(trainset)
        
        
    elif sampling_method in ("carbon_proportional_entropy", "carbon_proportional_gradient", "carbon_proportional_entropy_low"):
        # only HIGH-carbon clients use the pool; low-carbon clients keep their
        # data via plain carbon-proportional random.
        HIGH_CARBON_THRESHOLD = 400.0
 
        if carbon_intensity <= HIGH_CARBON_THRESHOLD:
            selected_indices = carbon_proportional_random_indices(
                dataset_size=len(trainset),
                carbon_intensity=carbon_intensity,
                alpha=alpha,
                m_min=m_min,
                max_intensity=1000.0,
                seed=server_round + partition_id,
            )
        else:
            cache_path = _cached_indices_path(results_dir, partition_id, sampling_method)
            pool = load_or_none(cache_path)
            if pool is None:
                scoring_loader = make_dataloader(trainset, batch_size=batch_size, shuffle=False)
                if sampling_method == "carbon_proportional_entropy":
                    pool = carbon_proportional_entropy_pool(
                        model=model, scoring_loader=scoring_loader,
                        dataset_size=len(trainset), carbon_intensity=carbon_intensity,
                        alpha=alpha, m_min=m_min, device=device, max_intensity=1000.0,
                    )
                elif sampling_method == "carbon_proportional_entropy_low":
                    pool = carbon_proportional_entropy_low_pool(
                        model=model, scoring_loader=scoring_loader,
                        dataset_size=len(trainset), carbon_intensity=carbon_intensity,
                        alpha=alpha, m_min=m_min, device=device, max_intensity=1000.0,
                    )
                else:
                    pool = carbon_proportional_gradient_pool(
                        model=model, scoring_loader=scoring_loader,
                        dataset_size=len(trainset), carbon_intensity=carbon_intensity,
                        alpha=alpha, m_min=m_min, device=device, max_intensity=1000.0,
                    )
                save_indices(cache_path, pool)
 
            import random
            k = _carbon_proportional_count(len(trainset), carbon_intensity, alpha, m_min, 1000.0)
            rng = random.Random(server_round + partition_id)
            selected_indices = list(pool) if k >= len(pool) else rng.sample(pool, k)
 
        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = len(selected_indices) / len(trainset)
    
    elif sampling_method == "equal_share":
        selected_indices = fixed_count_indices(
            dataset_size=len(trainset),
            count=equal_share_count,
            seed=server_round + partition_id,
        )
        selected_trainset = Subset(trainset, selected_indices)
        effective_ratio = len(selected_indices) / len(trainset)
    
    else:
        # fallback: unknown method -> full data (keeps the run alive)
        selected_trainset = trainset
        effective_ratio = 1.0
 
    sampling_time_s = time.perf_counter() - start_sampling
 
    trainloader = make_dataloader(selected_trainset, batch_size=batch_size, shuffle=True)
 
    start_training = time.perf_counter()
    train_loss = train_fn(model, trainloader, local_epochs, lr, device)
    training_time_s = time.perf_counter() - start_training
 
    client_total_time_s = time.perf_counter() - start_total
    samples_per_second = len(selected_trainset) / training_time_s if training_time_s > 0 else 0.0
    energy_kwh = (power_watts * client_total_time_s) / 3_600_000
    carbon_g = energy_kwh * carbon_intensity
 
    model_record = ArrayRecord(model.state_dict())
 
    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
        "client_id": int(partition_id),
        "server_round": int(server_round),
        "full_dataset_size": len(trainset),
        "sampled_dataset_size": len(selected_trainset),
        "warmup_active": int(warmup_active),
        "sampling_ratio": float(effective_ratio),
        "sampling_time_s": float(sampling_time_s),
        "training_time_s": float(training_time_s),
        "client_total_time_s": float(client_total_time_s),
        "samples_per_second": float(samples_per_second),
        "carbon_intensity": float(carbon_intensity),
        "power_watts": float(power_watts),
        "energy_kwh": float(energy_kwh),
        "carbon_g": float(carbon_g),
    }
 
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
 
    append_client_metrics(
        results_dir=results_dir,
        client_id=partition_id,
        row={
            "round": int(server_round),
            "client_id": int(partition_id),
            "sampling_method": sampling_method,
            "warmup_active": int(warmup_active),
            "sampling_ratio": float(effective_ratio),
            "full_dataset_size": len(trainset),
            "sampled_dataset_size": len(selected_trainset),
            "sampling_time_s": float(sampling_time_s),
            "training_time_s": float(training_time_s),
            "client_total_time_s": float(client_total_time_s),
            "samples_per_second": float(samples_per_second),
            "train_loss": float(train_loss),
            "carbon_intensity": float(carbon_intensity),
            "power_watts": float(power_watts),
            "energy_kwh": float(energy_kwh),
            "carbon_g": float(carbon_g),
        },
    )
 
    return Message(content=content, reply_to=msg)


def _cached_indices_path(results_dir: str, client_id: int, method: str) -> Path:
    cache_dir = Path(results_dir) / "selected_indices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"client_{client_id}_{method}.json"


def load_or_none(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_indices(path: Path, indices: list) -> None:
    with open(path, "w") as f:
        json.dump(indices, f)

@app.evaluate()
def evaluate(msg: Message, context: Context):

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    partition_method = str(context.run_config.get("partition-method", "dirichlet"))
    classes_per_partition = int(context.run_config.get("classes-per-partition", 8))
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 0.5))

    _, testset = load_partition_datasets(
        partition_id, num_partitions,
    partition_method=partition_method,
    alpha=dirichlet_alpha,
    classes_per_partition=classes_per_partition,
    )
    valloader = make_dataloader(testset, batch_size=batch_size, shuffle=False)

    eval_loss, eval_acc = test(model, valloader, device)

    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
    }
    

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)