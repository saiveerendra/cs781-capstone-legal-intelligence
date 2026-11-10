Phase 1: Legal Language Model Pretraining
In Phase 1, I trained language models on Indian legal documents and used them to generate text for the given test prompts. I tried a few Transformer architectures and submitted the decoder-only MHA model with RoPE as my best model.
1. Dataset and preprocessing
I used the Phase 1 data from the CS781 Capstone dataset under `Phase_1_Pretraining/`. The training documents came from four files:
`train_acts.json`
`train_high_courts.json`
`train_supreme_court.json`
`train_tribunals.json`
I used `preprocessing (2).ipynb` to examine the documents, remove duplicate text records, train a BPE tokenizer, and convert the text into token IDs. The tokenizer was configured for 8,000 tokens; the notebook output shows 7,999 vocabulary entries.
The model notebooks use about 200 million training tokens and 5 million validation tokens. For continued RoPE training, the notebook also loads a separate `train_tokens2.npy` file. The preprocessing notebook has more than one data-preparation experiment, so its cells should be checked before running them in order.
2. Models I tried
I compared these model setups:
Notebook	Model / experiment
`decoder-model-exp1 (2).ipynb`	Decoder-only Transformer with MHA and sinusoidal positions
`gqa-decoder-exp1 (2).ipynb`	Decoder-only Transformer with grouped-query attention
`mha-rope-decoder-exp1 (2).ipynb`	Decoder-only Transformer with MHA and RoPE; best submitted model
`mha-kernel (1).ipynb`	Decoder experiment with kernel-based attention
`hybridencoder (4).ipynb`	Encoder-based model experiment
`encoder-decoder (3).ipynb`	Encoder–decoder Transformer experiment
`preprocessing (2).ipynb`	Data preparation and tokenizer training
These were separate experiments, and not every training setting was kept identical across models.
3. Best model architecture
My best Phase 1 submission used a causal Transformer decoder with multi-head self-attention and rotary positional embeddings (RoPE). The model predicts the next token using the tokens before it. I used RoPE to handle position information inside attention instead of adding a fixed sinusoidal position vector to the input.
Setting	Value
Architecture	Decoder-only Transformer with causal MHA + RoPE
Transformer layers	12
Hidden size	512
Attention heads / KV heads	8 / 8
Feed-forward size	4,096
Maximum context length	1,024 tokens
Tokenizer	BPE (7,999 entries in notebook output)
Total parameters	67,091,456
Dropout	0.1
The code for this model is in `mha-rope-decoder-exp1 (2).ipynb`.
4. Training approach and implementation
I first prepared the text and token arrays, then trained the Transformer for next-token prediction. The training code samples fixed-length contexts from the token arrays and uses PyTorch to run the model and update its parameters.
For the best model, the notebook continues training from an earlier MHA–RoPE checkpoint. It restores the model and optimizer states, so it is a continuation of the previous run, not a fresh training run.
Training setting	Value
Optimizer	AdamW
Learning rate	`3e-4`
Weight decay	`0.1`
Batch size	16
Additional training steps in continuation	12,500
GPUs shown in saved output	2 × NVIDIA Tesla T4
After training, the notebook evaluates validation perplexity. For test prompts, it encodes the prompt with BPE, generates a continuation, decodes the tokens, and writes predictions to `prediction.json`. The saved inference setup uses `top_k=50`, `temperature=0.8`, and targets 128 generated words per prompt. The notebook also creates `submission.zip` for Codabench.
5. Results
The earlier checkpoint had a validation perplexity of 4.4559. After the continuation run, the best recorded validation perplexity was 3.8183.
Official Phase 1 result	Value
Best model	Decoder MHA + RoPE
Codabench Macro BERTScore F1	0.827954
Submission ID	`965195`
Recorded rank	99
The perplexity values are from saved notebook outputs. The Codabench score and rank are the recorded results for the project; they are not calculated by the notebook itself.
6. How to run
The notebooks were developed in Kaggle, so several cells use `/kaggle/input/...` paths. They also expect data files created in preprocessing or attached as Kaggle input datasets.
Open a Kaggle notebook with GPU enabled and install any missing Python packages.
Make the `Exploration-Lab/CS-781-Capstone` dataset available. The preprocessing notebook reads a Hugging Face token from Kaggle Secrets using the key `HF` where access is needed.
Run the relevant cells in `preprocessing (2).ipynb` to prepare the BPE tokenizer and training/validation token arrays. Some cells belong to alternative preprocessing runs, so select the branch you intend to use.
Update the hard-coded data paths in `mha-rope-decoder-exp1 (2).ipynb` for your Kaggle environment.
For the saved continuation setup, attach the earlier MHA–RoPE checkpoint expected by the notebook. Then run the model, optimizer loading, and training cells in order.
Run the validation and generation cells to produce `prediction.json` and `submission.zip`.
For a fresh training run, the continuation setup would need to be changed so the model starts with newly initialized weights. The original run cannot be reproduced exactly from this folder alone without its external token arrays, tokenizer, and earlier checkpoint.
Dependencies
The notebooks use Python, PyTorch, NumPy, and standard Python libraries. Data-preparation cells also use Hugging Face Hub, Kaggle Secrets, ijson, pandas, and matplotlib. Package versions are not pinned in this folder.
7. Files needed outside this folder
This folder contains experiment notebooks, but does not include the original corpus, serialized tokenizer, `.npy` token arrays, or saved model weights. These files are referenced through external datasets and Kaggle paths. I have not included weights or checkpoints in the code archive, as required by the submission instructions.
The `.ipynb_checkpoints/` folder in the original ZIP contains Jupyter autosaves, not separately trained models.
Best model code: The code for my best-performing Phase 1 model (MHA + RoPE decoder) is present in `mha-rope-decoder-exp1 (2).ipynb`. The preprocessing code is in `preprocessing (2).ipynb`.
