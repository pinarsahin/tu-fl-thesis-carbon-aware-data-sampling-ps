import random
import torch
import torch.nn.functional as F


def random_sample_indices(dataset_size: int, ratio: float, seed: int | None = None) -> list:
    if dataset_size <= 0:
        return []
    k = max(1, int(dataset_size * ratio))
    k = min(k, dataset_size)
    rng = random.Random(seed)
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return indices[:k]



def carbon_proportional_random_indices(
    dataset_size: int,
    carbon_intensity: float,
    alpha: float,
    m_min: int,
    max_intensity: float = 1000.0,
    seed: int | None = None,
) -> list:
    # fixed reference scale: ratio = 1 - alpha * (carbon / max_intensity)
    # carbon intensity treated as absolute value, capped at max_intensity
    # per-client floor keeps at least m_min samples, then random selection
    if dataset_size <= 0:
        return []
    ci = min(carbon_intensity, max_intensity)
    ratio = 1.0 - alpha * (ci / max_intensity)
    p_min = m_min / dataset_size
    ratio = max(p_min, ratio)
    ratio = min(1.0, ratio)
    k = max(1, int(dataset_size * ratio))
    rng = random.Random(seed)
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return indices[:k]



def _carbon_proportional_count(

    dataset_size: int,

    carbon_intensity: float,

    alpha: float,

    m_min: int,

    max_intensity: float = 1000.0,

) -> int:

    # identical ratio-and-floor logic to carbon_proportional_random_indices.

    # kept here so smart methods select the SAME per-client amount, only the

    # choice of which samples differs. does not alter the existing function.

    if dataset_size <= 0:

        return 0

    ci = min(carbon_intensity, max_intensity)

    ratio = 1.0 - alpha * (ci / max_intensity)

    p_min = m_min / dataset_size

    ratio = max(p_min, ratio)

    ratio = min(1.0, ratio)

    return max(1, int(dataset_size * ratio))

import random
import torch
import torch.nn.functional as F


def random_sample_indices(dataset_size: int, ratio: float, seed: int | None = None) -> list:
    if dataset_size <= 0:
        return []
    k = max(1, int(dataset_size * ratio))
    k = min(k, dataset_size)
    rng = random.Random(seed)
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return indices[:k]
    
def compute_class_rarity_weights(model, gamma=1.0, device="cpu"):
    """Class rarity multipliers from final-layer L2 weight norms.
    Smaller norm -> assumed under-represented -> higher weight."""
    # your Net's final classifier layer is fc2
    if hasattr(model, "fc2"):
        final_layer = model.fc2
    elif hasattr(model, "fc"):
        final_layer = model.fc
    elif hasattr(model, "classifier"):
        final_layer = (model.classifier if isinstance(model.classifier, torch.nn.Linear)
                       else model.classifier[-1])
    else:
        return None
 
    weights = final_layer.weight.data                 # [num_classes, feat]
    class_norms = torch.norm(weights, p=2, dim=1)
    inv_norms = 1.0 / (class_norms + 1e-6)
    rarity = inv_norms / torch.mean(inv_norms)         # normalise around 1.0
    return torch.pow(rarity, gamma).to(device)
 
 
def carbon_proportional_rarity_indices(
    model, scoring_loader, dataset_size, carbon_intensity,
    alpha, m_min, device, gamma=1.0, max_intensity=1000.0,
    use_difficulty=True, label_key="character", seed=None,
):
    """Draw k indices by multinomial sampling on (difficulty x rarity) or rarity only.
    No candidate pool, no truncation, so rare classes are never fully removed."""
    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)
    k = min(k, dataset_size)
 
    rarity_weights = compute_class_rarity_weights(model, gamma=gamma, device=device)
 
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in scoring_loader:
            images = batch["image"].to(device)
            labels = batch[label_key].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
 
            if use_difficulty:
                p_true = probs[torch.arange(labels.size(0)), labels]
                base = 1.0 - p_true                    # sample difficulty
            else:
                base = torch.ones(labels.size(0), device=device)  # rarity only
 
            if rarity_weights is not None:
                score = base * rarity_weights[labels]
            else:
                score = base
            scores.append(score)
 
    scores = torch.cat(scores)
    p = (scores + 1e-6) / torch.sum(scores + 1e-6)
 
    if seed is not None:
        g = torch.Generator(device=p.device)
        g.manual_seed(seed)
        selected = torch.multinomial(p, num_samples=k, replacement=False, generator=g)
    else:
        selected = torch.multinomial(p, num_samples=k, replacement=False)
    return selected.tolist()

def carbon_proportional_protected_indices(
    trainset,
    dataset_size: int,
    carbon_intensity: float,
    alpha: float,
    m_min: int,
    flagged_classes: list,
    protection_cap: float = 0.5,
    label_key: str = "character",
    max_intensity: float = 1000.0,
    seed: int = None,
) -> list:
    # carbon still sets HOW MUCH: same count as the other carbon methods.
    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)
    if k == 0:
        return []

    rng = random.Random(seed)
    all_indices = list(range(dataset_size))

    # no flagged classes -> plain random selection of k (same as carbon_proportional_random)
    if not flagged_classes:
        rng.shuffle(all_indices)
        return all_indices[:k]

    # read this client's own labels (local data, never transmitted)
    labels = [int(trainset[i][label_key]) for i in range(dataset_size)]
    flagged_set = set(flagged_classes)
    flagged_idx = [i for i in all_indices if labels[i] in flagged_set]
    other_idx = [i for i in all_indices if labels[i] not in flagged_set]

    # reserve up to protection_cap * k for flagged-class samples the client holds
    max_protected = int(protection_cap * k)
    rng.shuffle(flagged_idx)
    protected = flagged_idx[:max_protected]

    # fill the rest of the budget with random samples from other classes
    remaining = k - len(protected)
    rng.shuffle(other_idx)
    fill = other_idx[:remaining]

    # if not enough "other" samples to fill, top up from leftover flagged samples
    if len(fill) < remaining:
        leftover = flagged_idx[max_protected:]
        rng.shuffle(leftover)
        fill += leftover[: remaining - len(fill)]

    selected = protected + fill
    return selected[:k]    

def _schedule_multiplier(server_round: int, transition_round: int) -> float:
    # Fixed linear schedule. Returns a value in [0, 1] that says how far
    # along the ramp from start ratio to target ratio we are.
    # 0.0 = still at start (full data), 1.0 = fully at carbon-proportional target.
    # For the adaptive version later, replace this function with one that reads
    # a training-progress signal instead of the round number.
    if transition_round <= 0:
        return 1.0
    progress = server_round / transition_round
    return min(1.0, progress)

def fixed_count_indices(dataset_size: int, count: int, seed: int | None = None) -> list:
    """Select exactly `count` random indices (capped at dataset_size).
    Used for the equal-absolute-share baseline: every client keeps the same
    absolute number of samples regardless of its own dataset size."""
    import random as _random
    rng = _random.Random(seed)
    n = min(int(count), dataset_size)
    return rng.sample(range(dataset_size), n)


def carbon_proportional_scheduled_indices(
    dataset_size: int,
    carbon_intensity: float,
    alpha: float,
    m_min: int,
    server_round: int,
    transition_round: int = 10,
    start_ratio: float = 1.0,
    max_intensity: float = 1000.0,
    seed: int | None = None,
) -> list:
    # Ratio ramps from start_ratio (round 1) down to the client's carbon-
    # proportional target ratio (by transition_round), then stays at target.
    # Random selection throughout: no scoring, no warmup.
    if dataset_size <= 0:
        return []

    # target ratio = the existing carbon-proportional ratio (with floor)
    ci = min(carbon_intensity, max_intensity)
    target_ratio = 1.0 - alpha * (ci / max_intensity)
    p_min = m_min / dataset_size
    target_ratio = max(p_min, target_ratio)
    target_ratio = min(1.0, target_ratio)

    # interpolate from start_ratio down to target_ratio by transition_round
    m = _schedule_multiplier(server_round, transition_round)
    ratio = start_ratio + m * (target_ratio - start_ratio)
    # safety clamp: never below the floor, never above start
    ratio = max(p_min, min(start_ratio, ratio))

    k = max(1, int(dataset_size * ratio))
    k = min(k, dataset_size)
    rng = random.Random(seed)
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return indices[:k]

def carbon_proportional_entropy_indices(

    model,

    scoring_loader,

    dataset_size: int,

    carbon_intensity: float,

    alpha: float,

    m_min: int,

    device,

    max_intensity: float = 1000.0,

) -> list:

    # same count as carbon_proportional_random, selected by prediction entropy.

    # higher entropy = model less certain on that sample = kept first.

    # one forward pass, scored with the current (warmed-up) global model.

    # scoring_loader MUST be unshuffled so returned positions map to dataset indices.

    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)

    if k == 0:

        return []

    model.eval()

    scores = []

    with torch.no_grad():
        for batch in scoring_loader:
            x = batch["image"].to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            batch_entropy = -(probs * log_probs).sum(dim=1)
            scores.append(batch_entropy.cpu())


    scores = torch.cat(scores)

    k = min(k, scores.numel())

    return torch.topk(scores, k).indices.tolist()

def carbon_proportional_entropy_low_indices(

    model,

    scoring_loader,

    dataset_size: int,

    carbon_intensity: float,

    alpha: float,

    m_min: int,

    device,

    max_intensity: float = 1000.0,

) -> list:

    # same count as carbon_proportional_random, selected by prediction entropy.

    # higher entropy = model less certain on that sample = kept first.

    # one forward pass, scored with the current (warmed-up) global model.

    # scoring_loader MUST be unshuffled so returned positions map to dataset indices.

    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)

    if k == 0:

        return []

    model.eval()

    scores = []

    with torch.no_grad():
        for batch in scoring_loader:
            x = batch["image"].to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            batch_entropy = -(probs * log_probs).sum(dim=1)
            scores.append(batch_entropy.cpu())


    scores = torch.cat(scores)

    k = min(k, scores.numel())

    return torch.topk(scores, k, largest=False).indices.tolist()

def carbon_proportional_gradient_indices(

    model,

    scoring_loader,

    dataset_size: int,

    carbon_intensity: float,

    alpha: float,

    m_min: int,

    device,

    max_intensity: float = 1000.0,

) -> list:

    # same count as carbon_proportional_random, selected by data importance

    # (He et al., Definition 1): squared estimated gradient norm from the output

    # layer. for softmax + cross-entropy the output-layer pre-activation gradient

    # is (probs - one_hot); its squared L2 norm estimates the full gradient norm

    # from a forward pass alone, avoiding the backward pass. the model-dependent

    # scaling constant (rho*gamma) is dropped since it does not change ranking.

    # scored once with the current (warmed-up) global model.

    # scoring_loader MUST be unshuffled so returned positions map to dataset indices.

    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)

    if k == 0:

        return []

    model.eval()

    scores = []

    with torch.no_grad():
        for batch in scoring_loader:
            x = batch["image"].to(device)
            y = batch["character"].to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            one_hot = F.one_hot(y, num_classes=probs.size(1)).float()
            grad_output = probs - one_hot
            importance = (grad_output ** 2).sum(dim=1)
            scores.append(importance.cpu())

    scores = torch.cat(scores)

    k = min(k, scores.numel())

    return torch.topk(scores, k).indices.tolist()


 

def carbon_proportional_entropy_pool(
    model, scoring_loader, dataset_size, carbon_intensity, alpha, m_min,
    device, max_intensity=1000.0, pool_fraction=0.70,):
    # Score every sample by predictive entropy (high = uncertain = informative).
    # Return a POOL of the most-informative indices (larger than k), to be
    # resampled from each round. High-entropy pool.
    import torch
    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)
    pool_size = max(k, int(pool_fraction * dataset_size))

    entropies = []
    model.eval()
    with torch.no_grad():
        for batch in scoring_loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            entropies.append(ent.cpu())
    entropies = torch.cat(entropies)
    # top pool_size by entropy (most uncertain)
    pool = torch.topk(entropies, min(pool_size, dataset_size), largest=True).indices.tolist()
    return pool


def carbon_proportional_entropy_low_pool(
    model, scoring_loader, dataset_size, carbon_intensity, alpha, m_min,
    device, max_intensity=1000.0, pool_fraction=0.70,):
    # Same but low-entropy (most confident) pool.
    import torch
    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)
    pool_size = max(k, int(pool_fraction * dataset_size))

    entropies = []
    model.eval()
    with torch.no_grad():
        for batch in scoring_loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            entropies.append(ent.cpu())
    entropies = torch.cat(entropies)
    pool = torch.topk(entropies, min(pool_size, dataset_size), largest=False).indices.tolist()
    return pool


def carbon_proportional_gradient_pool(
    model, scoring_loader, dataset_size, carbon_intensity, alpha, m_min,
    device, max_intensity=1000.0, pool_fraction=0.70,
):
    # Score by output-layer gradient magnitude (He et al. proxy: |probs - onehot|).
    # Return the high-gradient pool.
    import torch
    k = _carbon_proportional_count(dataset_size, carbon_intensity, alpha, m_min, max_intensity)
    pool_size = max(k, int(pool_fraction * dataset_size))

    scores = []
    model.eval()
    with torch.no_grad():
        for batch in scoring_loader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            onehot = torch.zeros_like(probs)
            onehot[torch.arange(labels.size(0)), labels] = 1.0
            grad_proxy = (probs - onehot).norm(dim=1)
            scores.append(grad_proxy.cpu())
    scores = torch.cat(scores)
    pool = torch.topk(scores, min(pool_size, dataset_size), largest=True).indices.tolist()
    return pool