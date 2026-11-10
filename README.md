# CS781 Capstone — In-House Legal Intelligence System

## About this project

This CS781 capstone covers seven tasks involving Indian legal documents. Phase 1 trains a legal-domain language model. The remaining phases use its pretrained representations and compare different architectures and fine-tuning methods for each task.

The tasks cover text generation, named entity recognition, rhetorical-role classification, judgment prediction, statute identification, summarization, and prior-case retrieval. The experiments compare full fine-tuning, LoRA-based adaptation, hierarchical BiGRU models, and other phase-specific variants.

The notebooks were mainly developed in **Python and PyTorch on Kaggle GPUs**. Each phase folder has its own `README.md` explaining the dataset, model, code, experiments, and how to run that phase.

## Phases and best official results

The scores below are from the **recorded best Codabench submissions**. They are not the same as the validation metrics printed inside individual notebooks.

| Phase | Task | Best official model / approach | Primary metric | Best score | Recorded rank | Submission ID |
|---|---|---|---|---:|---:|---|
| [1](phase1/) | Legal language modeling | Decoder-only MHA + RoPE | BERTScore F1 | **0.827954** | 99 | `965195` |
| [2](phase2/) | Legal named entity recognition | Hybrid LoRA (Q/V + FFN) with partial fine-tuning | Strict entity-level Macro-F1 | **0.415849** | 98 | `976447` |
| [3](phase3/) | Rhetorical-role classification | Hybrid encoder; CL+IT, then CL training | Macro-F1 | **0.478979** | 79 | `991284` |
| [4](phase4/) | Court judgment prediction | Hybrid encoder + LoRA (Q/K/V/O) + hierarchical BiGRU | Macro-F1 | **0.767635** | 32 | `1008086` |
| [5](phase5/) | Legal statute identification | Encoder LoRA + BiGRU + attention | Macro-F1 | **0.351363** | 87 | `1014922` |
| [6](phase6/) | Legal judgment summarization | Encoder–decoder full fine-tuning | BERTScore F1 | **0.823633** | 113 | `1043776` |
| [7](phase7/) | Prior case retrieval | Hybrid LoRA+ | Micro-F1@7 | **0.255759** | 112 | `1037946` |

**Submission count:** **42 valid Codabench submissions** were made across the seven phases (4, 7, 6, 6, 6, 6, and 7 respectively).

## Repository structure

The intended final repository layout is:

```text
.
├── README.md
├── 
│   ├── phase1/
│   │   ├── README.md
│   │   └── ...
│   ├── phase2/
│   │   ├── README.md
│   │   └── ...
│   ├── phase3/
│   │   ├── README.md
│   │   └── ...
│   ├── phase4/
│   │   ├── README.md
│   │   └── ...
│   ├── phase5/
│   │   ├── README.md
│   │   └── ...
│   ├── phase6/
│   │   ├── README.md
│   │   └── ...
│   └── phase7/
│       ├── README.md
│       └── ...
└── report/
    └── CS781-CP-Sai-Kuruguti-261110059-report.pdf
```

Each phase folder should contain the **best-performing model's source code** and the non-weight configuration/support files needed to understand the implementation. The phase READMEs document the experimental notebooks as well. **Model weights and checkpoints are not part of this repository or final submission ZIP.**

## Implementation by phase

### [Phase 1 — Legal language model pretraining](phase1/README_PHASE1.md)

Phase 1 trains a decoder-only Transformer with multi-head attention and rotary position embeddings (RoPE) on Indian legal texts. A custom BPE tokenizer is used, and different model architectures are compared before selecting the MHA + RoPE decoder. The recorded validation perplexity improved from **4.4559** to **3.8183** after continued training.

**Best model code:** `phase1/mha-rope-decoder-exp1 (2).ipynb`  
**Preprocessing:** `phase1/preprocessing (2).ipynb`

### [Phase 2 — Legal named entity recognition](phase2/README.md)

Phase 2 predicts entity labels and **exact character spans** in legal judgments. Annotated spans are aligned with BPE tokens, token-level classifiers are trained, and final span offsets are reconstructed from predicted BIO tags. The best official result was recorded for a Hybrid LoRA and partial-fine-tuning approach.

**Code note:** `hybrid-model-2 (2).ipynb` documents an encoder + NER head experiment, but its active training code has **not** been confirmed as the exact model behind official submission `976447`. The winning code revision needs to be matched before claiming that file reproduces the best official score.

### [Phase 3 — Rhetorical-role classification](phase3/README.md)

Phase 3 classifies individual judgment sentences using a context-aware encoder model. Nearby sentences provide context, and the best Hybrid model is trained in two stages: first on **CL + IT** data, followed by fine-tuning on **CL**.

**Main model code:** `phase3/phase3hybrid (2).ipynb`

### [Phase 4 — Court judgment prediction](phase4/README.md)

Phase 4 treats judgment prediction as a binary classification task. Because judgments exceed a single Transformer input, they are split into overlapping chunks. Each chunk is encoded, and the chunk representations are combined with a **BiGRU**. The best approach used LoRA adapters on the encoder's Q, K, V, and O projections.

**Best model code:** `phase4/phase4-hybrid-lorma (2).ipynb`  
The filename mentions *lorma*, but the active best-model implementation uses **LoRA + hierarchical BiGRU**.

### [Phase 5 — Legal statute identification](phase5/README.md)

Phase 5 predicts statutes applicable to a judgment as a **multi-label classification** problem. The best model used a LoRA-adapted encoder, a BiGRU for combining chunk information, and attention before the 100-label output layer. The official best submission used a prediction threshold of **0.35**.

**Main model code:** `phase5/encoder-decoder-p5.ipynb`

### [Phase 6 — Legal judgment summarization](phase6/README.md)

Phase 6 trains sequence-to-sequence models to generate summaries from legal judgments. LoRA and other variants are compared, but the best official result comes from **full fine-tuning of an encoder–decoder Transformer**. In the saved training run, the best validation loss was **3.103317 at epoch 8** out of 10 epochs.

**Best model code:** `phase6/phase6-encoderdecoder-fullfinetuning.ipynb`

### [Phase 7 — Prior case retrieval](phase7/README.md)

Phase 7 ranks candidate prior cases for each query and returns the **top seven** results. The experiments use chunked text representations, an encoder-based retrieval setup, and random-negative sampling. The compared approaches include LoRA, LoRMA+, full fine-tuning, and other retrieval variants.

**Code note:** The best official submission used **Hybrid LoRA+**. The strongest saved local-validation notebook in the supplied archive uses **LoRMA+**, which is a different adapter method; it should not be labelled as the exact winning submission without confirming the corresponding code revision. See the [Phase 7 README](phase7/README.md) for the notebook-by-notebook breakdown.

## Running the code

The phase notebooks were developed on **Kaggle** and use phase-specific input paths and datasets. There is no single command that runs all seven tasks from start to finish. To rerun a phase:

1. Open its folder and read that phase's `README.md` for the correct notebook and setup.
2. Create a Kaggle GPU notebook or another PyTorch environment with a suitable GPU.
3. Make the relevant data from [`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) available to the runtime. Configure access through secrets if the dataset requires authentication.
4. Provide the **tokenizer, pretrained initialization, and any intermediate data files** expected by that notebook. These are external runtime inputs, and their paths may need updating.
5. Install the packages imported by the notebook, then run its preprocessing, model definition, training/validation, and prediction cells in order.
6. Check the final `prediction.json` and submission ZIP against the evaluation format required for that phase.
