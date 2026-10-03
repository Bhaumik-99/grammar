# Grammar scoring for spoken English (SHL Hiring Assessment 2026)

Predicts a continuous grammar score (0 to 5, MOS rubric) for 45-60 s spoken-English answers.

* Kaggle competition: `shl-hiring-assessment-2026`
* Kaggle username: `simpra26`
* Final notebook: [`notebook/shl_grammar_scoring_engine.ipynb`](notebook/shl_grammar_scoring_engine.ipynb), as executed on
  Kaggle, with all outputs.

## Results

| Metric | Value |
|---|---|
| Cross-validated RMSE (speaker-grouped; 732 scorable training clips) | 0.510 |
| Cross-validated Pearson r | 0.865 |
| Cross-validated composite (RMSE + 1 - r) / 2, a proxy for the leaderboard scale | 0.3225 |
| Cross-validated RMSE re-weighted to the test set's mix of recording batches | 0.519 |
| Training RMSE (in-sample, all 769 training clips) | 0.285 |
| Public leaderboard score (competition metric combining RMSE and Pearson r; lower is better) | 0.3316 (3rd of 63 on 3 Oct 2026) |

## Approach

1. **Zero-score gate.** The 37 training clips graded 0 are noise-masked recordings with a distinctive signature
   (dynamic range below 5 dB, peak amplitude above 0.9). A rule separates them perfectly on the training set (37 of 37,
   no false positives) and sets them to 0; no test clip triggers it. All regressors are trained on the 732 scorable
   clips.
2. **Speaker-grouped validation.** Speakers recur in the training set (same-speaker grades vary by only about 0.2)
   while test speakers are new, so random folds overstate performance. Pseudo-speaker groups are built from low-layer
   WavLM statistics. Every fold split, inner hyper-parameter search and label-using feature (judge anchors, DeBERTa
   folds) is speaker-grouped.
3. **Complementary views of each clip.**
   * *How the speaker sounds:* frozen Whisper-large-v3 encoder, WavLM-large, w2v-BERT 2.0 and HuBERT-large states,
     and the audio-token states of two audio LLMs (Voxtral-Mini-3B, Qwen2-Audio-7B), using fixed layer bands pooled
     over time.
   * *What the speaker said:* three ASR transcripts that differ in how faithfully they keep learner errors
     (Whisper-large-v3; Whisper with a disfluent prompt; Parakeet-CTC without a language model). From these:
     fluency, ASR-confidence and cross-ASR disagreement features, spaCy syntactic complexity, minimal-edit grammatical
     error correction rates, LLM surprisal, an anchored rubric judge (Qwen3-8B), Qwen3-8B hidden states,
     Qwen3-Embedding vectors, and a DeBERTa-v3-large regressor.
4. **Models.** Small regularised heads per view (ridge with speaker-grouped penalty selection, PCA + RBF-SVR,
   LightGBM) produce out-of-fold predictions. A non-negative least-squares stack combines them and is evaluated with
   a second-level speaker-grouped CV on different folds. Predictions are clipped to [1, 5].
5. **Selection on cross-validation only.** No leaderboard feedback was used for tuning. A component was kept only if
   it improved the stack's cross-validated score. Three components were tried and dropped: a Voxtral-Small-24B view,
   pooling of the Voxtral states over real-audio tokens only, and a LoRA-tuned Voxtral-Mini regressor. See notebook
   section 11 and `experiments/`.

The notebook also covers calibration, residuals by grade and by recording batch, the stack weights, SHAP values of the
tabular model, layer-wise probes of each encoder, and limitations.

## Repository layout

```
notebook/          final notebook (CPU, ~7 min on Kaggle) and its kernel-metadata.json; writes submission.csv
pipeline/01-08_*/  Kaggle GPU notebooks whose outputs the final notebook reads
experiments/       components tried and not kept; ECAPA speaker-embedding cross-check
analysis/          local development scripts: feature builders, layer probes, stacking experiments, ablation tests
```

| Pipeline notebook | Produces | Needs |
|---|---|---|
| `01_eda_asr` | audio statistics; Whisper-large-v3 transcripts with word timings | competition data |
| `02_asr_verbatim_views` | error-preserving transcripts: Whisper with a disfluent prompt, Parakeet-CTC-1.1B | competition data |
| `03_audio_embeddings` | Whisper-large-v3 encoder, WavLM-large, w2v-BERT 2.0 layer-wise states | competition data |
| `04_audio_llm_embeddings` | Voxtral-Mini-3B audio-token states, HuBERT-large states | competition data |
| `05_qwen2_audio_embeddings` | Qwen2-Audio-7B-Instruct audio-token states | competition data |
| `06_text_llm_features` | for the clean Whisper transcript: Qwen3-8B judge (zero-shot and anchored), surprisal, hidden states, GEC corrections; Qwen3-Embedding | 01, 03 |
| `07_text_llm_features_views` | for the two error-preserving transcripts: Qwen3-8B anchored judge, surprisal, hidden states; Qwen3-Embedding (no GEC) | 01, 02, 03 |
| `08_text_deberta` | DeBERTa-v3-large regressor, speaker-grouped out-of-fold predictions | 01, 02, 03 |

The scripts in `pipeline/` and `experiments/` are the code that produced the attached outputs, with two changes made
afterwards: docstrings were corrected for accuracy, and the competition's grammar rubric was removed from the judge
prompt in 06 and 07 (it is left as a marked placeholder).

## Reproducing

The competition data is not included, as the competition rules require. With a Kaggle account that has joined the
competition:

1. In every `kernel-metadata.json` (including `notebook/`), replace the username in `id` and `kernel_sources` with
   yours.
2. In `pipeline/06_text_llm_features/llm_judge.py` and `pipeline/07_text_llm_features_views/llm_views.py`, replace
   the `RUBRIC` placeholder with the 1-5 grammar rubric from the competition's data description page.
3. Run the pipeline notebooks on GPU, for example `kaggle kernels push -p pipeline/01_eda_asr`. Notebooks 01-05 are
   independent; 06 needs 01 and 03; 07 and 08 need 01-03.
4. Run the final notebook on CPU with `kaggle kernels push -p notebook`. Its metadata attaches the competition data
   and the eight pipeline notebooks. It prints the cross-validated metrics and the training RMSE, and writes
   `submission.csv`.

Fold assignments depend on the scikit-learn version: `StratifiedGroupKFold(shuffle=True)` in the Kaggle image shuffles
differently from recent releases. A re-run elsewhere therefore moves the CV figures by about +/-0.002.

The scripts in `analysis/` run locally. They expect the competition CSVs in `data/` and the pipeline outputs in
`outputs/`: 01 in `outputs/eda`, 02 in `outputs/asr_views`, 03-05 in `outputs/audio_emb`, 06-07 in
`outputs/llm_feats`, 08 in `outputs/text_deberta`. `analysis/speaker_groups.py` writes the pseudo-speaker groups to
`outputs/speakers`. Experiment outputs go to `outputs/voxtral_lora` (LoRA regressor), `outputs/speakers` (ECAPA
clusters) and `outputs/audio_emb` (Voxtral-Small and real-audio-pooled Voxtral-Mini states). Both folders are
git-ignored.

## Models and licences

Whisper-large-v3 (Apache-2.0 weights, MIT code), WavLM-large (UniSpeech licence), w2v-BERT 2.0 (MIT), HuBERT-large
(Apache-2.0), Voxtral-Mini-3B (Apache-2.0), Qwen2-Audio-7B-Instruct (Apache-2.0), Qwen3-8B and Qwen3-Embedding-4B
(Apache-2.0), DeBERTa-v3-large (MIT), Parakeet-CTC-1.1B (CC-BY-4.0), spaCy (MIT), scikit-learn (BSD-3-Clause),
LightGBM (MIT). Used only in `experiments/`: Voxtral-Small-24B (Apache-2.0) and SpeechBrain ECAPA-TDNN
`spkrec-ecapa-voxceleb` (Apache-2.0).

The code in this repository is released under the MIT licence (see `LICENSE`).
