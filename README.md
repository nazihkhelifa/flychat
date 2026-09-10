# FLY/READ

A trained English text-emotion classifier using the **real 499-neuron MaleCNS recurrent circuit published in Oruk's speech experiment**. The browser now loads a trained model. The former keyword rules and decorative network are gone.

This is a computational model of a recorded fly subgraph, not a living brain or a complete physiological simulation. The graph contains 15,865 directed connections. Its recorded topology and published signed/scaled recurrent weights are preserved exactly. Original MaleCNS body IDs are absent from Oruk's public export; all cell numbers in this app are explicitly circuit-local indices.

## Run the app

Requires Node.js. From this directory:

```powershell
node scripts/serve.mjs
```

Open http://127.0.0.1:4173/. The checked-in `dist/model/` artifacts let the app work without Python, an API key, or a remote inference service. Initial model assets are approximately 15.6 MB; all assets are served locally. An HTTPS static host also works. The server binds to loopback only.

The interface supports text classification, recorded anatomy rotation, actual activity replay, a heatmap of all 499 cells (rows averaged for display), and silencing 0–200 neurons ranked by trained readout weight. Silencing reruns the same sentence without retraining or rescaling. Negative class scores are valid raw ridge scores, not probabilities.

## Reproduce training

Python 3.12 and the project-local environment were used. The existing environment is `.venv`. On a new checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe scripts/prepare_inputs.py
.\.venv\Scripts\python.exe scripts/train_fly_text_reservoir.py
node scripts/verify_model.mjs
```

`prepare_inputs.py` downloads the public circuit/anatomy and the 20,000-row split configuration of `dair-ai/emotion` at pinned revision `cab853a1dbdf4c42c2b3ef2173804746df8825fe`. It records hashes and checks the source model against Oruk's published SHA-256. No full EM volume or 1.1 GB graph download is needed because the exact experimental subgraph is available.

Raw inputs and local Python packages stay in ignored `data/` and `.venv/` directories. Training writes browser artifacts to `dist/model/` and a readable report to `research/results.json`. It does not upload or publish anything. The training command replaces the generated model artifacts.

## Method

1. Preserve the official train/validation/test boundaries, removing duplicate canonical token sequences in train-first order. Remaining sizes: **15,969 / 1,993 / 1,986**. Author metadata is unavailable, so this is not an author-independent evaluation.
2. Fit TF-IDF and a 128-component LSA representation using training text only, without labels. Normalize each word embedding. A seed-42 fixed random projection maps embeddings into the 499 cells. Unknown words have zero input. All text is limited to 96 tokens; inputs shorter than four steps repeat the last token for pooling. The tokenizer handles English alphabetic words and apostrophes; it does not model punctuation or prosody.
3. Run the fixed circuit: `x[t] = 0.65*x[t-1] + 0.35*tanh(W*x[t-1] + input[token])`. Each sentence starts from zero. `W` uses Oruk's published spectral scaling and transmitter-sign assumptions unchanged. No labels train `W` or the input projection.
4. Concatenate the mean neuron states from four successive temporal bins. Fit a class-balanced ridge classifier, selecting its penalty on validation macro-F1. The exported classifier receives **only the 1,996 circuit features**, without a direct text shortcut. Standardization is folded into exported float64 readout coefficients.
5. Fit independent readouts for scrambled connectivity (outgoing counts, signs, and weights retained before global spectral rescaling), no recurrence, the same text encoder alone, and a stronger TF-IDF text-only baseline. All comparisons use the same splits and vocabulary. Test labels are used after validation-based selection.

## Measured results

These are single-seed results on 1,986 deduplicated held-out sentences, not the audio experiment's metrics.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Recorded fly wiring | 55.4% | 50.6% |
| Scrambled wiring | 48.2% | 43.1% |
| No recurrence | 56.4% | 51.9% |
| Same text encoder only | 40.0% | 35.3% |
| TF-IDF text-only baseline | 78.4% | 74.1% |

The majority-label baseline is 34.6% accuracy. A 500-resample paired sentence bootstrap gives a conditional 95% interval of +5.5 to +9.4 percentage points for fly-minus-scrambled macro-F1. This does **not** establish a biological-wiring advantage: the no-recurrence control and TF-IDF baseline both do better, and the interval is conditional on one seed, one circuit and one split.

This modest classifier makes obvious mistakes, including classifying the fixture “I am furious about this.” as love. That fixture is retained, not hidden or special-cased. Short text, sarcasm, domains beyond the training corpus, unknown words, and neutral text are difficult. The six training labels are sadness, joy, love, anger, fear, surprise; there is no neutral label. Scores do not reveal a person's actual emotional state.

See `research/results.json` / `dist/model/report.json` for full-precision results, source hashes, validation trials, per-class precision/recall/F1, and confusion matrices.

## Verification

- Source CSR offsets, indices, and weights match the released circuit exactly; the graph is one strongly connected component.
- Sampled geometry retains all 499 cell owners and consistent paired segment endpoints.
- Browser-engine/Python parity was checked on nine inputs, intact and with 50 silenced neurons: maximum raw-score error < 0.000003. Final states and activity traces also match; silenced cells remain zero.
- Invalid counts, empty text, excessive input length, unknown words, truncation and determinism are checked by `node scripts/verify_model.mjs`.
- Local HTML, modules, model assets, report and credits respond with HTTP 200 and appropriate MIME types.

Full browser UI testing was not requested and was not performed. The optional feature-detected `classify_fly_text` WebMCP registration uses the same inference action; no supported WebMCP validation context was available, so that optional integration is unverified.

## Sources and terms

- [MaleCNS v1.0](https://male-cns.janelia.org/download/): FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research; CC BY 4.0.
- [Oruk / Nathan Roll](https://oruk.ai/research/we-taught-a-fruit-fly-to-read-human-emotion): 499-cell circuit selection and public anatomy export. The speech model's trained classifier, audio input projection, private dataset and recordings are not used here.
- [DAIR.AI emotion dataset](https://huggingface.co/datasets/dair-ai/emotion), [Saravia et al., CARER (2018)](https://aclanthology.org/D18-1404/): dataset card restricts use to educational and research purposes. Raw training sentences are not included in the web app.
- [Xenova fruit-fly simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation): reference for local simulation interaction, not a code or asset dependency.

See `dist/model/NOTICE.txt` for attribution and adaptation details. The text input, readout, and browser renderer here are new; anatomical lines are derived from measured centerlines, not generated artwork.
