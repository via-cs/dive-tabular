# TAB-DDPM source notice

`modules.py`, `gaussian_multinomial_diffusion.py`, and `utils.py` are adapted
from the official [yandex-research/tab-ddpm](https://github.com/yandex-research/tab-ddpm)
implementation at commit
`b476257dd460b778ba09eb97f7a51d6490fa17f8`.

The upstream implementation is distributed under the MIT license included in
this directory as `LICENSE.md`. The local `tabddpm.py` file provides modern
preprocessing, training, checkpoint, and sampling orchestration for this
repository's existing experiment format.

Please cite Kotelnikov et al., *TabDDPM: Modelling Tabular Data with Diffusion
Models* (ICML 2023). Full provenance and BibTeX are provided in the artifact's
root `THIRD_PARTY.md`.
