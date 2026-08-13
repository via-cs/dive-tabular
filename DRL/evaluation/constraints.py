"""Stub for ``evaluation.constraints``.

The original DRL repo shipped a use-case-specific ``constraint_satisfaction``
helper that is only exercised by the WGAN training loop on a hand-coded URL
dataset. The synthesizer modules import the symbol unconditionally, so we
ship a minimal stub that raises if the helper is actually invoked.
"""


def constraint_satisfaction(*args, **kwargs):
    raise NotImplementedError(
        "constraint_satisfaction was never ported from the original DRL "
        "URL/botnet use cases. Remove the call site or provide a "
        "dataset-specific implementation."
    )
