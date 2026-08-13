import json
import random
from typing import List
from DRL.constraints_code.classes import Variable

random.seed(0)


def set_random_ordering(ordering: List[Variable]):
    random.shuffle(ordering) # in-place shuffling
    return ordering


def set_random_orderings(ordering: List[Variable], k: int, seed: int = 0) -> List[List[Variable]]:
    """Return ``k`` independently-shuffled copies of ``ordering``.

    Uses its own ``random.Random`` instance so callers can reproduce the bank
    from a single seed without disturbing the module-level RNG state used by
    ``set_random_ordering``.
    """
    rng = random.Random(seed)
    base = list(ordering)
    out: List[List[Variable]] = []
    for _ in range(k):
        shuffled = list(base)
        rng.shuffle(shuffled)
        out.append(shuffled)
    return out


def set_ordering(use_case, ordering: List[Variable] | List[str], label_ordering_choice: str, model_type: str, data_partition='test'):
    if label_ordering_choice == 'random':
        ordering = set_random_ordering(ordering)
    elif label_ordering_choice == 'predefined':
        if isinstance(ordering, str):
            ordering = list(map(lambda x: Variable(x), ordering.split()))
        # Otherwise the ordering was already parsed into List[Variable] by
        # ``parse_constraints_file`` (the typical entry point for the
        # textual constraints format), so we keep it as-is.
    else:
        json_filename = f'feature_ordering/feature_orderings.json'
        with open(json_filename, "r") as f:
            data = json.load(f)

        if label_ordering_choice == 'causal':
            ordering = data["feature_orderings"][use_case]['general'][label_ordering_choice]['train']
            ordering = list(map(lambda x: Variable(x), ordering.split()))
        else:
            model_type = model_type.lower()
            ordering = data["feature_orderings"][use_case][model_type][label_ordering_choice][data_partition]
            ordering = list(map(lambda x: Variable(x), ordering.split()))

    readable_ordering = [e.readable() for e in ordering]
    print(f'Using *{label_ordering_choice}* feature ordering:\n', readable_ordering, 'len:', len(readable_ordering))
    return ordering