# Labeling Guide

The model is only as good as the labels. These four classes are chosen to align
with public meme datasets and to keep `Dark Humor` separate from targeted abuse.

## Why four classes

- **Promotional was removed.** It is a purpose (advertising), not a tone, it
  overlaps every other class, and no public dataset labels it. Keeping it would
  force you to hand-collect a class that the model cannot learn reliably.
- **Dark was renamed Dark Humor.** This makes it clear that the class is edgy or
  dark comedy, not dark imagery.
- **Offensive is reserved for targeted abuse.** This keeps it separate from
  non-targeted edgy jokes.

If you later collect enough promotional memes, add a `Promotional/` folder and
add the label to `CATEGORIES` in `utils/inference.py`. The training pipeline is
folder-driven and will pick up the extra class automatically.

## Precedence rule

When a meme fits more than one class, apply this order:

1. Offensive
2. Political
3. Dark Humor
4. Wholesome

## Class definitions

| Class | Use for | Do not use for |
|---|---|---|
| Wholesome | Uplifting, cute, supportive, wholesome humor | Anything offensive or hateful |
| Dark Humor | Dark comedy, death, tragedy, edgy jokes without a direct attack | Slurs or targeted abuse (use Offensive) |
| Political | Elections, parties, politicians, policy, propaganda | A political meme that is also abusive (use Offensive) |
| Offensive | Slurs, hate, sexual harassment, targeted abuse, extreme vulgarity | Mild edgy humor without a target (use Dark Humor) |

If a meme is genuinely ambiguous, skip it. Noisy labels cost more accuracy than
a smaller clean dataset.

## Data layout

```
data/
├── train/
│   ├── Wholesome/
│   ├── Dark Humor/
│   ├── Political/
│   └── Offensive/
└── test/
    ├── Wholesome/
    ├── Dark Humor/
    ├── Political/
    └── Offensive/
```

- Aim for **300–500 images per class** to start, balanced across classes.
- Keep **20%** of each class in `data/test`.
- Split by meme template, not randomly. The same template with different text
  leaks between train and test and inflates accuracy.
- Deduplicate with a perceptual hash before splitting.

## Suggested sources

- **Memotion (SemEval-2020 Task 8)** — sentiment/emotion; maps to Wholesome, Dark Humor, Offensive.
- **Hateful Memes (Meta)** — hateful/not; maps to Offensive, with hard negatives.
- **HarMeme / HarMeme-V1** — harmful COVID and US-politics memes; good for Political and Offensive.
- **MAMI (SemEval-2022 Task 5)** — misogynous memes; maps to Offensive.
- **MMHS150K** — large Twitter image set; text-heavy, useful for Offensive and Dark Humor.
- **MultiOFF** — small 2016 US-election offensive set; Political + Offensive.

Remap every source to the precedence rule above and manually review a sample of
each mapped class.

## Measuring progress

Always compare models on the same `data/test` folder:

```powershell
python training/evaluate.py --data-dir data/test --mode zero-shot
python training/evaluate.py --data-dir data/test --mode fusion
```

Report **accuracy, macro-F1, per-class F1, and the confusion matrix**. Accuracy
alone is misleading when classes are imbalanced.
