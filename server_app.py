import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, QFedAvg

from task import Net, load_centralized_testloader, test, test_per_class

_RUN_DIR = None
_SAMPLING_METHOD = None
_FLAGGED_CLASSES = []
_CLASS_ACC_HISTORY = {}
_SMOOTH_WINDOW = 3
_FLAG_START_ROUND = 15
_FLAG_MARGIN = 0.20
_UNFLAG_MARGIN = 0.10
_PHASE_SWITCH_THRESHOLD = None
_PHASE_SWITCHED = False
_BURST_THRESHOLD = None      
_BURST_LENGTH = 5            
_BURST_START = None          

app = ServerApp()


def create_run_dir() -> Path:
    run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: dict) -> None:
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def log_round_results(run_dir: Path, row: dict) -> None:
    output_path = run_dir / "round_metrics.csv"
    file_exists = output_path.exists()

    fieldnames = [
        "round", "sampling_method", "sampling_ratio", "warmup_rounds",
        "learning_rate", "local_epochs", "batch_size", "train_loss",
        "sampling_time_s", "training_time_s", "client_total_time_s",
        "samples_per_second", "sampled_dataset_size", "full_dataset_size",
        "client_eval_loss", "client_eval_acc", "server_eval_loss", "server_eval_acc",
    ]

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_testloader()
    test_loss, test_acc, class_correct, class_total = test_per_class(model, test_dataloader, device)

    print(f"Round {server_round} — server eval loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

    if _RUN_DIR is not None:
        per_class_path = Path(_RUN_DIR) / "per_class_accuracy.csv"
        file_exists = per_class_path.exists()
        with open(per_class_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["round"] + [f"class_{c}" for c in range(62)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            row = {"round": server_round}
            for c in range(62):
                if class_total[c] > 0:
                    row[f"class_{c}"] = class_correct[c] / class_total[c]
                else:
                    row[f"class_{c}"] = ""
            writer.writerow(row)
    if _BURST_THRESHOLD is not None and _RUN_DIR is not None:
        global _BURST_START
        # start the burst the first time we cross the threshold
        if _BURST_START is None and test_acc >= _BURST_THRESHOLD:
            _BURST_START = server_round
            print(f"Round {server_round}: crossed {_BURST_THRESHOLD:.2f}, rarity burst for {_BURST_LENGTH} rounds")
        # burst is active for rounds [start, start+length)
        burst_active = (
            _BURST_START is not None
            and _BURST_START <= server_round < _BURST_START + _BURST_LENGTH
        )
        with open(Path(_RUN_DIR) / "phase.json", "w") as f:
            json.dump({"burst_active": burst_active}, f)

    # --- flagged-class protection signal (only when the protected method uses it) ---
    if _SAMPLING_METHOD == "carbon_proportional_protected":
        global _FLAGGED_CLASSES, _CLASS_ACC_HISTORY

        for c in range(62):
            if class_total[c] > 0:
                acc_c = class_correct[c] / class_total[c]
                hist = _CLASS_ACC_HISTORY.setdefault(c, [])
                hist.append(acc_c)
                if len(hist) > _SMOOTH_WINDOW:
                    hist.pop(0)

        if server_round >= _FLAG_START_ROUND:
            smoothed = {c: sum(h) / len(h) for c, h in _CLASS_ACC_HISTORY.items()}
            vals = sorted(smoothed.values())
            median = vals[len(vals) // 2] if vals else 0.0

            new_flagged = set(_FLAGGED_CLASSES)
            for c, s in smoothed.items():
                if c in new_flagged:
                    if s > median - _UNFLAG_MARGIN:
                        new_flagged.discard(c)
                else:
                    if s < median - _FLAG_MARGIN:
                        new_flagged.add(c)
            _FLAGGED_CLASSES = sorted(new_flagged)

            if _RUN_DIR is not None:
                with open(Path(_RUN_DIR) / "flagged_classes.json", "w") as f:
                    json.dump(_FLAGGED_CLASSES, f)

            print(f"Round {server_round} flagged classes: {_FLAGGED_CLASSES}")
    # --- end flagging signal ---
        

    # --- phase switch signal ---
    if _PHASE_SWITCH_THRESHOLD is not None and _RUN_DIR is not None:
        global _PHASE_SWITCHED
        if not _PHASE_SWITCHED and test_acc >= _PHASE_SWITCH_THRESHOLD:
            _PHASE_SWITCHED = True
            print(f"Round {server_round}: crossed {_PHASE_SWITCH_THRESHOLD:.2f}, switching to full")
        with open(Path(_RUN_DIR) / "phase.json", "w") as f:
            json.dump({"switched": _PHASE_SWITCHED}, f)
    # --- end phase switch ---

    return MetricRecord({"loss": float(test_loss), "accuracy": float(test_acc)})


@app.main()
def main(grid: Grid, context: Context) -> None:

    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    lr = float(context.run_config["learning-rate"])
    warmup_rounds = int(context.run_config["warmup-rounds"])
    sampling_ratio = float(context.run_config["sampling-ratio"])
    sampling_method = str(context.run_config["sampling-method"])
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    alpha = float(context.run_config.get("alpha", 0.9))
    m_min = int(context.run_config.get("m-min", 300))
    aggregation = str(context.run_config.get("aggregation", "fedavg"))
    q_param = float(context.run_config.get("q-param", 0.1))
    phase_switch_threshold = float(context.run_config.get("phase-switch-threshold", 0.0))
    early_method = str(context.run_config.get("early-method", ""))
    burst_threshold = float(context.run_config.get("burst-threshold", 0.0))
    burst_length = int(context.run_config.get("burst-length", 5))
    global _BURST_THRESHOLD, _BURST_LENGTH
    _BURST_THRESHOLD = burst_threshold if burst_threshold > 0 else None
    _BURST_LENGTH = burst_length
    global _PHASE_SWITCH_THRESHOLD
    _PHASE_SWITCH_THRESHOLD = phase_switch_threshold if phase_switch_threshold > 0 else None

    experiment_config = {
        "num_server_rounds": num_rounds,
        "fraction_evaluate": fraction_evaluate,
        "learning_rate": lr,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "warmup_rounds": warmup_rounds,
        "sampling_ratio": sampling_ratio,
        "sampling_method": sampling_method,
        "alpha": alpha,
        "m_min": m_min,
        "aggregation": aggregation,
        "q_param": q_param,
        "experiment_description": context.run_config.get("experiment-description", ""),
    }

    run_dir = create_run_dir()
    global _RUN_DIR, _SAMPLING_METHOD
    _RUN_DIR = str(run_dir)
    _SAMPLING_METHOD = sampling_method
    save_config(run_dir, experiment_config)

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    if aggregation == "qfedavg":
        strategy = QFedAvg(
            client_learning_rate=lr,
            q=q_param,
            fraction_train=1.0,
            fraction_evaluate=1.0,
        )
    else:
        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=1.0,
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "warmup_rounds": warmup_rounds,
                "sampling_ratio": sampling_ratio,
                "sampling_method": sampling_method,
                "alpha": alpha,
                "m_min": m_min,
                "results_dir": str(run_dir),
                "early_method": early_method,
                "phase_switch_threshold": phase_switch_threshold,
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    all_rounds = sorted(
        set(result.train_metrics_clientapp.keys())
        | set(result.evaluate_metrics_clientapp.keys())
        | set(result.evaluate_metrics_serverapp.keys())
    )

    for rnd in all_rounds:
        train_m = result.train_metrics_clientapp.get(rnd, {})
        client_eval_m = result.evaluate_metrics_clientapp.get(rnd, {})
        server_eval_m = result.evaluate_metrics_serverapp.get(rnd, {})

        log_round_results(
            run_dir=run_dir,
            row={
                "round": rnd,
                "sampling_method": sampling_method,
                "sampling_ratio": sampling_ratio,
                "warmup_rounds": warmup_rounds,
                "learning_rate": lr,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "train_loss": train_m.get("train_loss"),
                "sampling_time_s": train_m.get("sampling_time_s"),
                "training_time_s": train_m.get("training_time_s"),
                "client_total_time_s": train_m.get("client_total_time_s"),
                "samples_per_second": train_m.get("samples_per_second"),
                "sampled_dataset_size": train_m.get("sampled_dataset_size"),
                "full_dataset_size": train_m.get("full_dataset_size"),
                "client_eval_loss": client_eval_m.get("eval_loss"),
                "client_eval_acc": client_eval_m.get("eval_acc"),
                "server_eval_loss": server_eval_m.get("loss"),
                "server_eval_acc": server_eval_m.get("accuracy"),
            }
        )

    print(f"\nSimulation complete. Results saved to: {run_dir}")

    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, run_dir / "final_model.pt")