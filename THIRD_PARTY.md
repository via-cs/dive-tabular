# Third-party implementation provenance

This artifact adapts or vendors implementation code from the following public
research repositories. These references credit the upstream implementations;
they do not identify the authors of this artifact.

## CTGAN and TVAE

The implementations in:

- `synthetizers/CTGAN/base_ctgan.py`
- `synthetizers/CTGAN/ctgan.py`
- `synthetizers/TVAE/tvae.py`

were adapted from the CTGAN and TVAE implementations distributed in
[DRL_DGM](https://github.com/mihaela-stoian/DRL_DGM), the code release for
*Beyond the Convexity Assumption: Realistic Tabular Data Generation under
Quantifier-Free Real Linear Constraints*. DRL_DGM is distributed under the
Apache License 2.0; its license is preserved as
`licenses/DRL_DGM_LICENSE`. The runner scripts and common experiment workflow
in this artifact are local orchestration and are not part of DRL_DGM.

Recommended implementation and DRL paper citation:

```bibtex
@inproceedings{stoian2025drl,
  title     = {Beyond the Convexity Assumption: Realistic Tabular Data
               Generation under Quantifier-Free Real Linear Constraints},
  author    = {Mihaela C. Stoian and Eleonora Giunchiglia},
  booktitle = {Proceedings of the Thirteenth International Conference on
               Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=rx0TCew0Lj}
}
```

For the CTGAN/TVAE model family itself, also cite the original model paper used
by DRL_DGM:

```bibtex
@inproceedings{xu2019ctgan,
  title     = {Modeling Tabular Data using Conditional GAN},
  author    = {Lei Xu and Maria Skoularidou and Alfredo Cuesta-Infante and
               Kalyan Veeramachaneni},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}
```

## TabDDPM

The diffusion core in:

- `synthetizers/TabDDPM/modules.py`
- `synthetizers/TabDDPM/gaussian_multinomial_diffusion.py`
- `synthetizers/TabDDPM/utils.py`

was adapted from the official
[yandex-research/tab-ddpm](https://github.com/yandex-research/tab-ddpm)
implementation at commit `b476257dd460b778ba09eb97f7a51d6490fa17f8`. The
upstream implementation is distributed under the MIT License; its license is
preserved as `synthetizers/TabDDPM/LICENSE.md`. The local `tabddpm.py` and
`run_TabDDPM.py` files provide artifact-specific preprocessing, training,
checkpoint, and sampling orchestration.

Recommended paper citation:

```bibtex
@inproceedings{kotelnikov2023tabddpm,
  title     = {TabDDPM: Modelling Tabular Data with Diffusion Models},
  author    = {Akim Kotelnikov and Dmitry Baranchuk and Ivan Rubachev and
               Artem Babenko},
  booktitle = {Proceedings of the 40th International Conference on Machine
               Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {17564--17579},
  year      = {2023},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v202/kotelnikov23a.html}
}
```
