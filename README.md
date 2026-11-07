# Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations

**Maximilian von Klinski · Sebastian Lapuschkin · Wojciech Samek · Lennart Bürger**

[Paper](https://arxiv.org/abs/2609.39807) · [Dataset](https://huggingface.co/datasets/maxvonk/anti-factual-personas) · [Citation](#citation)

Code and datasets accompanying our paper on the reliability of LLM lie detectors under role-play and spurious correlations.

[![Figure 1: Anti-factual role-play, failures of existing lie detectors, confounder tests, and our proposed probe.](Figure1.png)](Figure1.png)

*Figure 1. Overview of the benchmark, failure analysis, and proposed probe. From the [paper](https://arxiv.org/abs/2609.39807), CC BY 4.0.*

## Overview

Do lie detectors identify falsehoods when a language model adopts a persona with beliefs that contradict reality? We study this question using anti-factual personas and targeted tests of concepts that can be confused with truthfulness.

- **Anti-factual personas:** 8,916 human-reviewed responses across 15 personas and three language models, evaluated under original and shared-persona prompts.
- **Confounder tests:** datasets that separate factual correctness from response likelihood, persona belief, and instruction compliance.
- **Lie detection probes:** implementations of prior approaches and our linear probe, trained with instruction-conflicting answers to improve robustness.

## Getting started

Authenticate with Hugging Face using an account with access to the gated [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and [Gemma 3 27B Instruct](https://huggingface.co/google/gemma-3-27b-it) models. Accept each model's access conditions on its Hugging Face page. An existing login or the `HF_TOKEN` environment variable can be used.

```bash
git clone https://github.com/max-vkl/stress-testing-llm-lie-detectors.git
cd stress-testing-llm-lie-detectors
./setup.sh
```

Setup installs the dependencies and prepares the model and training data. Run the main Llama experiments with:

```bash
./reproduce_results.sh
```

Results are saved in `results/paper_main_llama/`, including aggregate tables, per-dataset metrics, trained probes, and plots.

## Code and data

| Location | Contents |
| --- | --- |
| [`data/`](data/) | Training data and persona and confounder datasets for Llama 3.3, Gemma 3, and Gemma 4 |
| [`deception_detection/`](deception_detection/) | Probe implementations, activation extraction, training, and evaluation |
| [`deception_detection/scripts/configs/paper_main_llama/`](deception_detection/scripts/configs/paper_main_llama/) | Main-experiment configurations |
| [`tests/`](tests/) | Tests for the data, probes, and experiment pipeline |

## Citation

```bibtex
@misc{vonklinski2026stresstesting,
  title         = {Stress-Testing {LLM} Lie Detectors: Role-Play Failures and Spurious Correlations},
  author        = {von Klinski, Maximilian and Lapuschkin, Sebastian and Samek, Wojciech and B{\"u}rger, Lennart},
  year          = {2026},
  eprint        = {2609.39807},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.39807}
}
```

## License

Our code is MIT-licensed and our datasets are CC BY 4.0. Model generations are also subject to the Llama 3.3 and Gemma licenses.

Our pipeline extends Apollo Research's [deception-detection](https://github.com/ApolloResearch/deception-detection) framework (no stated license), which also provides the roleplaying and descriptive QA data. Further training data come from [RepE](https://github.com/andyzoujm/representation-engineering) (MIT), [Truth is Universal](https://github.com/sciai-lab/Truth_is_Universal) (MIT; includes [Geometry of Truth](https://github.com/saprmarks/geometry-of-truth) data, no stated license), [DolusChat](https://huggingface.co/datasets/AlignmentResearch/DolusChat) (CC BY 4.0) and [UK AISI lie-detection rollouts](https://huggingface.co/datasets/ai-safety-institute/lie-detection-rollouts) (license "other"). These third-party parts keep their original terms.
