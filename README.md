# TikHarm Moderation Triage

**A multimodal (video + sound + text) first-pass triage assistant for short-video moderation. It recommends, cites a policy rule, and leaves the decision to a human.**

Upload a short clip. The system looks at the pictures, listens to the sound, reads the speech and the on-screen text, estimates one of four classes
(**Safe, Harmful, Adult, Suicide/self-harm**), finds the matching rule in a policy, recommends an action and shows everything on a one-page "case card".
It never removes or punishes anything on its own.

> The policy, the retrieval test questions and the
> action table were **written for this project**. They are not any platform's real rules.
<img width="838" height="536" alt="image" src="https://github.com/user-attachments/assets/aa43febf-1971-4d5e-a656-2d356906de03" />


## The research question

> Does fusing video, audio and text outperform any single modality for human-state recognition, and when?

**Answer: yes, modestly.** Video carries most of the signal. Adding sound and text gives a small but consistent gain, and the combined model can be misled by mismatched sound or text.

## Key results

All scores are macro-F1 (an average over the four classes; 1.0 is perfect, about 0.25 is guessing) on **870 held-out clips from creators who never appear in training**.

| Approach | Macro-F1 |
|---|---|
| File information only (length, size, video quality; no pictures or sound): **the bar to beat** | 0.530 |
| Text only (speech transcript + on-screen text) | 0.570 |
| Sound only | 0.673 |
| Video only | 0.863 |
| Video + sound + text, simple join | 0.867 |
| Video + sound + text, gated | 0.871 |
| **Video + sound + text, cross-attention** | **0.887** |
| Final saved model (5-seed ensemble, trained on train+val) | **0.882** (95% range 0.860 to 0.902) |

- **Is the gain real?** Repeated on 5 different creator-grouped splits with 5 training runs each, cross-attention beat video-only by about **+0.025** (all clips) and **+0.030** (length-matched test), positive in 5 of 5 splits.
- **Does it use sound and text?** Removing both brings the score back to about the video-only level (0.865). Swapping in sound or text from a clip of a different class lowers it below video-only (0.827 and 0.844).
- **Per-class recall (final model):** Safe 0.96, Suicide 0.96, Adult 0.83, Harmful 0.77.
- **Policy layer (self-written policy and 48 test questions):** a correct rule is first for 77% of questions and in the top three for 100%. Ordering the review queue by risk times rule severity places 69% of Suicide-class clips in the top 20% (random order: 20%).

## How it works

```
clip (first 10 s) ─► video frames ─► ViT ───────────────┐
                 ├─► audio ────────► AST (+ music/speech scores)
                 ├─► speech (only if detected) ► Whisper ─┐
                 └─► 5 frames ─────► EasyOCR ─────────────┴► multilingual sentence encoder
                                       │
                 cross-attention fusion head (5-seed ensemble; missing inputs are masked)
                                       │
          class probabilities ─► policy retrieval (limited to rules that fit the class) ─► action table ─► case card ─► human
```

- **Frozen pretrained models** (ViT-B/16, AST, Whisper-small, EasyOCR, multilingual MiniLM) turn each input into numbers. Only a small fusion head is trained, so everything runs on a free Kaggle GPU.
- **Router:** mostly simple rules (no sound: video only; no speech: skip Whisper; no text: ignore it). A learned gate was tested and did not learn to separate music from speech.
- **Policy layer:** 31 prototype rules in 6 groups. Retrieval is semantic search over the rules (not full RAG: no text generation). The action table is rule-based and includes safety rules: Suicide-related cases are escalated to a wellbeing reviewer and **never auto-removed**, Adult predictions are at least age-restricted, and anything uncertain goes to a human.

## Data

Uses the public **TikHarm** dataset (Kaggle: `anhoangvo/tikharm-dataset`), short TikTok-style videos in four folders.
**The data is not included in this repository.** Download it yourself and check its licence and TikTok's terms of use.
After cleaning: 3,891 clips (3,948 as downloaded; 34 duplicate copies, 19 files labelled as two different classes and 4 near-duplicates removed).
Evaluation uses a **creator-grouped split** (no creator appears in more than one of train, validation and test).

The audit also found and measured "shortcuts" (clip length, video quality and amount of movement differ by class), which is why the **file-information-only score (0.530)** is reported as the bar a model must beat.

## Limitations (read before using)

- The policy, the retrieval test questions and the action table were written by us. Scores show the design works, not that decisions match real moderators.
- Real clips give the rule search only speech, on-screen text and sound tags, but the test questions describe what is shown. Rule matching on real clips is **not measured**.
- Only the first 10 seconds of a clip are analysed. One dataset, mostly Vietnamese and English. OCR reads Latin-script text only.
- The dataset is about 73% non-Safe, far above a real platform, so precision and queue numbers are optimistic.
- Harmful content is the hardest class (77% found). The Suicide class partly depends on how a video looks (plain background, captions, little movement), so harmless text slides may be flagged.
- Fused models can be fooled by misleading sound or text.
- Speech recognition runs when the sound model hears speech, so songs with vocals can be transcribed as lyrics.

## Responsible use and sensitive content

- Intended use: decision support for human reviewers, research and education. **Not** for automatic enforcement, and **not** for surveillance or identifying people.
- The Adult and Suicide folders were never viewed during development. Work was done through numbers, and case cards hide transcripts and on-screen text for these two classes.
- **Do not commit transcripts, on-screen text tables, case-card files or raw clips.** Keep them private (see `.gitignore`).
- Child-safety content is out of scope. Real systems use trained teams and hash matching against restricted databases.
- Suicide-related cases should always show local support resources. Helpline details are deliberately not hard-coded; configure them for your region.

## Models used (check each licence before reuse)

`google/vit-base-patch16-224-in21k`, `MIT/ast-finetuned-audioset-10-10-0.4593`, `openai/whisper-small`, EasyOCR,
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; `facebook/wav2vec2-base` was compared and not used.





