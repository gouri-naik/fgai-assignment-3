# English to French translation with four model architectures

**Gouri Naik · 230953086 · ICT4403 Foundations of Generative AI Assignment 3**

## Objective

Develop and compare four English-to-French machine translation systems on the same held-out parallel-text test set: a basic Transformer trained from scratch, mT5-small with LoRA fine-tuning, and zero-shot prompted LLaMA 3.2 1B Instruct and Gemma 3 270M IT. Compare quality and compute, inspect individual translations, and discuss responsible use.

## Dataset and preprocessing

The public [OPUS Books English–French parallel-text dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books) (`Helsinki-NLP/opus_books`, `en-fr`) supplies sentence pairs. The notebook normalizes Unicode with NFKC, collapses whitespace, filters empty or very long sentences (more than 180 characters), removes exact duplicate bilingual pairs, and shuffles with seed 42. It reported 99,264 cleaned pairs. The fixed subset contains 2,000 training, 100 validation, and 100 test pairs. The 100 test sources and references are in `results/translations.csv`. Exact duplicate pairs were removed before the split, though near duplicates and the same source paired with different targets may remain.

## Models and execution

| Model | Architecture | Task-specific adaptation |
| --- | --- | --- |
| Basic Transformer | Encoder–decoder | Three encoder and three decoder layers, shared SentencePiece vocabulary fitted on the training pairs, three training epochs |
| `google/mt5-small` | Encoder–decoder | One epoch of LoRA fine-tuning, 344,064 trainable adapter parameters |
| `meta-llama/Llama-3.2-1B-Instruct` | Decoder-only | Zero-shot instruction prompting, no task-specific training |
| `google/gemma-3-270m-it` | Decoder-only | Zero-shot instruction prompting, no task-specific training |

Run `FGAI_Assignment3_Colab.ipynb` in Google Colab with a GPU. Its first code cell installs `transformers`, `datasets`, `accelerate`, `peft`, `sentencepiece`, `sacrebleu`, `huggingface_hub`, `pandas`, and `matplotlib`; Colab supplies PyTorch. It removes an optional outdated `torchao` that conflicts with PEFT. The two gated models require accepting their respective [LLaMA](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [Gemma](https://huggingface.co/google/gemma-3-270m-it) terms. Save a Hugging Face read token in Colab Secrets as `HF_TOKEN`; do not put it in the notebook or repository. Select a T4 GPU and run cells in order. This run used a Tesla T4.

The notebook's `SAVE_TO_DRIVE` setting is `False`; set it to `True` **before a new run** to retain results in Google Drive. The provided `results/` directory contains the actual outputs of the completed run. No new run is needed to read those results.

## Measured results

All models translated the same 100 test sentences with greedy generation and a 64-token output cap for the main comparison. SacreBLEU corpus BLEU and corpus chrF use one reference per example. Training time excludes dataset and model downloads; inference time includes tokenization and generation for all 100 test examples but excludes model loading. Prompt-only models have no task-specific training time (N/A).

| Model | BLEU | chrF | Task training (s) | Test inference (s) | Total parameters | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Basic Transformer | 0.213 | 7.613 | 14.627 | 29.836 | 6,039,456 | 6,039,456 |
| mT5-small LoRA | 0.124 | 4.092 | 41.201 | 114.947 | 300,176,768 | 344,064 |
| LLaMA 3.2 1B prompt | **12.138** | **33.671** | N/A | 55.096 | 1,235,814,400 | 0 |
| Gemma 3 270M prompt | 7.677 | 29.977 | N/A | 79.594 | 268,098,176 | 0 |

The full precision metrics are in `results/results.csv`; the comparison chart is in `results/metric_comparison.png`. LLaMA performed best on both quality measures. The small Transformer was fastest at test inference but produced largely repetitive fragments. mT5 generated sentinel tokens and repetitions after only one fine-tuning epoch. Prompted models benefited from much larger-scale pretraining, and they differ in tokenizers, size, precision, and adaptation method; these results do **not** isolate an architecture effect. Gemma ran in float32 with eager attention after an earlier float16 generation failure, which also affects its inference-time comparison.

## Translation review and responsible use

`results/translations.csv` contains all 100 sources, references, and outputs from the four models. `QUALITATIVE_REVIEW.md` analyzes concrete errors. For example, Gemma changes *my uncle* to *mon grand-père* in test row 3 and sometimes leaves English untranslated. LLaMA changes *I am stifling* to *je suis en colère* in row 5. These errors can alter relationships, meaning, or sensitive statements despite a higher corpus score. Human review is needed for consequential uses. The literary English–French subset and a single seed cannot establish performance on other domains, dialects, or languages.

## Sampling temperature analysis

The main comparison is greedy, without sampling. Separately, LLaMA translated five held-out items once each at temperatures 0.3, 0.7, and 1.0 with top-p 0.9. The 15 outputs are in `results/temperature_samples.csv`. At 0.7, one Sunday sentence becomes an unrelated statement about Monday; at 1.0, a *Maria Stella* output adds an explanatory English wrapper. Lower temperature did not eliminate instruction prefixes such as `French:` or `English:`. These examples show variation, but five sentences and one sample per setting are insufficient to establish a reliable quality trend. The optional Gemma temperature cell was not enabled in this run.

## Conclusion and repository files

LLaMA 3.2 1B Instruct was the best-performing system by BLEU and chrF in this measured setting. It used zero-shot prompting and the largest model, so its advantage cannot be attributed to decoder-only design alone. Real deployment should preserve the source text for review, verify entities, negation, numbers, and sensitive content, and evaluate the intended language and domain.

`results/` contains the full outputs, histories, split metadata, and plot. The executed notebook retains the code and printed outputs. `QUALITATIVE_REVIEW.md` records the error analysis and mitigation measures. Model weights, dataset caches, and access tokens are intentionally excluded.
