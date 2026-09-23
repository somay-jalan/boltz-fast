# Boltz Fast

Native and packed batched inference for Boltz-2, with an optional EDM Heun diffusion solver. Euler remains the default. Based on the open-source Boltz code and weights; original license and scientific attribution are retained below.

See the [batching and branch guide](docs/native_batching.md) for usage, solver restrictions and validation.

<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz2_title.png" width="300"/>
  <img src="https://model-gateway.boltz.bio/a.png?x-pxid=bce1627f-f326-4bff-8a97-45c6c3bc929d" />

[Boltz-1](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) |
[Slack](https://boltz.bio/join-slack) <br> <br>
</div>



![](docs/boltz1_pred_figure.png)


## Introduction

Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy. Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and binding affinities, a critical component towards accurate molecular design. Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods, while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) technical reports. To discuss updates, tools and applications join our [Slack channel](https://boltz.bio/join-slack).

## Experimental native batching

This branch contains native and packed batching plus the opt-in EDM Heun solver. Read the [branch, usage and validation guide](docs/native_batching.md) and [Heun solver notes](docs/edm_heun.md) before comparing it with pristine upstream. Euler remains the default solver.

### Optimized fixed packed version

The `feature/packed-triangle-kernels` branch is the optimized packed version.
It fixes the per-record triangle dispatch bottleneck by running heterogeneous
records through grouped Triton triangle multiplication and attention kernels.
It also fixes contraction rounding to match the reference BF16 precision boundary.
Other per-record operations remain; this is a performance fix, with the known
confidence-ranking limitation documented in the [full report](docs/full_structure116.md).

Enable the optimized backend explicitly after installing this branch:

```sh
boltz predict inputs --batch_layout packed --batch_size 4 \
  --packed_pair_backend triton --seed 42
```

The backend requires CUDA, supports inference only, and remains opt-in; the
sequential backend is still the default. See [implementation and CUDA tests](docs/packed_triangles.md).

### Packed MSA follow-up

The packed Triton backend also avoids repeated MSA input conversions and large
outer-product mask temporaries. Against the previous packed version (`e331765`),
the 116-target trunk benchmark fell from **742.99 to 727.42 seconds** including
process overhead (**2.10% less time**); MSA module time decreased **9.28%**.
All 62 CUDA tests and 58 sampled representation comparisons matched exactly.
The per-input MSA loops remain. See the [MSA profiling and validation report](docs/packed_msa_profiling.md).
This is a trunk-only comparison; the full-prediction results below predate these changes.

### Structure116 full-prediction baseline (`e331765`)

Full prediction completed on all 116 targets, producing 580 structures with the
packed B4 Triton backend at `e331765`. Compared with the saved original Boltz-2 B1 result:

| Measurement | Original B1 | Packed B4 + Triton |
|---|---:|---:|
| Total inference process time | 78.72 min | **56.22 min** |
| Protein lDDT | 0.8541 | 0.8534 |
| Backbone lDDT | 0.9086 | 0.9075 |
| Protein-interface DockQ (56 targets) | 0.6139 | 0.6125 |
| Peak observed device memory | 68.50 GiB | 94.20 GiB |

This is **1.40× throughput, with 28.6% less process time** on an RTX PRO 6000
Blackwell. Mean accuracy scores are slightly lower; paired target bootstrap
intervals include zero for all three differences. This one-seed experiment does
not establish equivalence, and the timing baseline is historical. Some individual
targets show substantial losses despite the close averages; the detailed report
includes the largest changes and checks of all five samples on two outliers.

Both configurations use Euler 200 steps, step scale 1.5, five recycles and five
samples per target; the latter two are benchmark overrides, not CLI defaults.
No targets were skipped or processes restarted. Nine allocation warnings were
recovered internally and remain included in the timing. See the
[full results, confidence-ranking audit and reproduction](docs/full_structure116.md).

## Installation

Install this repository in a fresh Python environment to use its batching changes:

```sh
git clone --branch feature/packed-triangle-kernels https://github.com/somay-jalan/boltz-fast.git
cd boltz-fast
pip install -e '.[cuda]'
```

The command above installs the optimized fixed packed version. The default `main` branch contains the earlier packed implementation and optional Heun solver; select this feature branch to obtain the grouped Triton backend. The Python package and CLI remain named `boltz`. Installing the upstream PyPI `boltz` package alone does not install these branch changes. For CPU-only installation, omit `[cuda]`; the batched experiments were validated on a GPU.

## Inference

You can run inference using Boltz with:

```
boltz predict input_path --use_msa_server
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md). By default, the `boltz` command will run the latest version of the model.


### Binding Affinity Prediction
There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. Its value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder. The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule. This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log10(IC50)`, derived from an `IC50` measured in `μM`. More details on how to run affinity predictions and parse the output can be found in our [prediction instructions](docs/prediction.md).

## Authentication to MSA Server

When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways. More information is available in our [prediction instructions](docs/prediction.md).
 
## Evaluation

⚠️ **Coming soon: updated evaluation code for Boltz-2!**

To encourage reproducibility and facilitate comparison with other models, on top of the existing Boltz-1 evaluation pipeline, we will soon provide the evaluation scripts and structural predictions for Boltz-2, Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset, and our affinity predictions on the FEP+ benchmark, CASP16 and our MF-PCBA test set.

![Affinity test sets evaluations](docs/pearson_plot.png)
![Test set evaluations](docs/plot_test_boltz2.png)


## Training

⚠️ **Coming soon: updated training code for Boltz-2!**

If you're interested in retraining the model, currently for Boltz-1 but soon for Boltz-2, see our [training instructions](docs/training.md).


## Contributing

We welcome external contributions and are eager to engage with the community. Connect with us on our [Slack channel](https://boltz.bio/join-slack) to discuss advancements, share insights, and foster collaboration around Boltz-2.

On recent NVIDIA GPUs, Boltz leverages the acceleration provided by [NVIDIA  cuEquivariance](https://developer.nvidia.com/cuequivariance) kernels. Boltz also runs on Tenstorrent hardware thanks to a [fork](https://github.com/moritztng/tt-boltz) by Moritz Thüning.

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
