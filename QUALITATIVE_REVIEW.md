# Translation examples and responsible use

Rows below are zero-based indices in `results/translations.csv`. The observations are based on actual outputs from the 100-sentence test run.

| Row | Source and observed output | Concern |
| ---: | --- | --- |
| 3 | Source: “my uncle”; Gemma: “mon grand-père” | The model invents a different family relationship. |
| 5 | Source: “I am stifling”; LLaMA: “je suis en colère” | Physical distress becomes anger, changing the meaning of a potentially sensitive statement. |
| 4–5 | Gemma repeats the English source instead of translating it. | The instruction is not followed and the target language is missing. |
| 8 | Source: “the smith”; LLaMA: “le foudre” rather than “le forgeron”. | An occupation is mistranslated. |
| 9 | Reference: “Je me sentis perdu”; Gemma: “Je me suis senti terminé.” | Literal wording does not preserve the sense of being in trouble. |
| 0–9 | The scratch Transformer repeats short fragments; mT5 emits `<extra_id_0>` and repeated fragments. | These outputs are incomplete and unreliable despite producing numerical scores. |

The errors above include invented details, semantic changes, untranslated text, and malformed outputs. The two low-scoring encoder–decoder systems had far less task training than their pretrained prompted counterparts. The examples alone do not measure a failure rate, and the dataset does not contain a designed bias or cultural-sensitivity audit. Gendered language, names, dialects, cultural references, and sensitive content need targeted test cases before any such claim can be made.

For responsible use, keep the original text beside the translation; require a human reviewer for medical, legal, or safety-critical passages; check names, relationships, numbers, negation, and quoted speech; test diverse identities and dialects; document provenance and domain limits; and provide a way to correct harmful or inaccurate translations. BLEU and chrF measure overlap with references, not factual fidelity or safety.

## Temperature sample review

`results/temperature_samples.csv` contains five LLaMA examples at each temperature 0.3, 0.7, and 1.0. One sample per sentence per setting does not support a general statistical conclusion. In this particular sample, the Sunday sentence at 0.7 becomes “On ne retourne plus le lundi,” changing the day and omitting most source content. At 1.0, the *Maria Stella* sample includes English explanatory text and alters the name to “Marie-Stella.” At 0.3, several outputs still include `French:` or `English:` prefixes. Sampling changes the wording and can introduce errors; a practical deployment should prefer controlled decoding and still review the output. The main reported BLEU and chrF scores use greedy generation, not these sampled outputs.
