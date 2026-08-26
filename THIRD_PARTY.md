# Third-party software and method references

DiCRA is released under the MIT License. This notice describes the boundary
between DiCRA's source release, third-party Python dependencies, and adapted
baseline implementations.

## Python dependencies

PyTorch, Hugging Face Transformers, PEFT, Datasets, NumPy, SciPy,
scikit-learn, pandas, Matplotlib, ModelScope, and the remaining packages listed
in `requirements.txt` are external dependencies. They are not redistributed
by this repository and remain governed by their respective licenses.

## Adapted baseline implementations

The following files implement controlled adaptations to DiCRA's shared
sentence-level class-incremental interface. They are not claimed to be official
evaluations under the original papers' protocols.

| Baseline | Local implementation | Upstream reference |
|---|---|---|
| O-LoRA | `baselines/run_olora.py` | `cmnfriend/O-LoRA`, Orthogonal Subspace Learning for Language Model Continual Learning |
| CLoRA | `baselines/run_clora.py`, `src/clora_regularizer.py` | `sutakori/CLoRA`, source commit `802cda88cd21e839326701ba5c2ba48cbd317be0` |
| SLoRA-Pre | `baselines/run_slora.py`, `src/slora.py` | `alina1031/SLoRA`; audited source archive SHA256 `91140958447f931ac941be96b616044f9e860bfc105b6d3f44457e2c35529005` |
| SLCA/LCA-style controls | documented-dev LCA and alignment scripts | GengDavid/SLCA and the corresponding classifier-alignment method descriptions |

This repository does not redistribute the upstream repositories, source
archives, pretrained model weights, or datasets listed above. The upstream
projects and artifacts retain their own copyright and licensing status. Where
an upstream archive did not contain a license file, that archive is excluded
from this release; only the local adapted implementation and its provenance
metadata are provided.

## Models and datasets

Pretrained BERT weights and tokenizers are downloaded separately from their
original distributors. CLINC150, Banking77, and FewRel data are also excluded
from this repository. Users are responsible for obtaining those artifacts
under their original terms. DiCRA provides preparation scripts, split
manifests, content hashes, and class-order metadata but does not relicense the
underlying models or datasets.
