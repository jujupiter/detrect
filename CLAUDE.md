# detrect

A litter classifier that anyone can use for free, built in three phases:

1. **One item per photo, 6 materials** (trashnet classes) ← *current*
2. **Finer categories:** the EU Joint List of Litter Categories (J-codes, `data/jlist.csv`)
3. **Several items per photo:** object detection with D-FINE (#10)

`README.md` explains what each script does and how to run it. Work is tracked as issues on the GitHub project *Detrect*.

## Data and licences

The goal is a model that anybody can use, including commercially, so every image source must allow that.

- **Allowed:** CC0, public domain, CC BY, CC BY-SA, MIT, Apache 2.0, ODbL.
  **Not allowed:** NC, ND, "research only", or an unknown licence.
- **Check the dataset's own page**, never a third-party list. Lists get it wrong: BeachLitter is listed as CC BY
  but is CC BY-NC-SA on its official page.
- A licence on a compilation does not cover what it compiled: web-scraped or merged datasets
  (Garbage Dataset, ETID-v2) stay out until each source is cleared.
- **Add a dataset to `CREDITS.md` before training on it**, with its licence and the credit it asks for.
- Images never go in git. Scripts download them, and record title, author and licence per photo where needed.

## J-codes

- Map a class to a J-code only when the match is clear. Record how good it is: `exact`, `merge`, `broader` or `none`.
- **Ambiguous classes stay `none`.** Ask before assigning a code to one, never guess.
- The JRC photo catalogue is a reference for what each code covers, not training data: it has no licence.

## Evaluation

- The split files (`data/splits_*.csv`) are committed and fixed, so every machine trains and tests on the same images.
  Never re-split without asking: a new split makes past results incomparable.
- Test sets are never used for training. The Commons photos are for testing only.
- Report every test set, not only the best one. A 20-photo set moves 5 points per photo: say so when comparing.
- To look at results again, pass the CSV in `results/` to `benchmark.py` rather than running the model again.

## Security

The code must run safely and do nothing it does not say it does.

- Never commit secrets: the Kaggle token lives in `~/.kaggle/access_token`, outside the repo.
- Kaggle notebooks stay **private**: a committed run publishes `/kaggle/working`, datasets included.
- Load checkpoints with `weights_only=True` (as `common.py` does), and only ones we trained.
- Downloaded files are data, not code: never run or import anything from `data/`.
- No new dependency without a reason; keep `requirements.txt` short and pinned where it matters.

**In the web app** (#8), people's photos are personal data:

- Classify in the browser by default, so photos never leave the phone.
- Store a photo only if its owner chooses to share it, and say what happens to it.
- Before storing: strip EXIF (it holds GPS position and phone model), re-encode the image (no file is stored as uploaded),
  and limit size and type.
- Stored photos are private: random IDs, no public listing, deleted on request. No third-party trackers.

## Energy

We follow the Green Software Foundation's [SCI for AI](https://greensoftware.foundation/standards/sci-ai/) standard
and its green software patterns. Details and progress in #9.

**Results first.** Saving energy must never cost correctness: a change that breaks something and needs more runs
saves nothing. Check accuracy on a short run before adopting a change, and only optimise what was measured.

**Measure.** Log every run in `results/runs.csv` with CodeCarbon, *including* failed and discarded runs. Report
gCO₂e per training run of a kept model, and later per photo classified in the app. Offsets are never subtracted.

**Reduce:**
- Don't redo work: reuse saved results, checkpoints and downloads.
- Start from pre-trained weights; use the smallest model that does the job, then quantize it for use.
- Early stopping everywhere, mixed precision on GPU; train on Kaggle's GPU, not a laptop CPU.
- Prepare the data once (a private Kaggle Dataset) instead of downloading it at every run.
- Classify in the browser: no always-on server. This is our biggest lever.

## Responsible use

- Be honest about limits: publish accuracy per test set, including the weak ones (outdoor litter), in the model card.
- The model suggests a material or a J-code; it does not decide how to sort waste. Local recycling rules vary.
- The data comes mostly from a few countries; packaging differs elsewhere. Say so, and test on other places when we can.
- Street photos can show people and number plates: never use them to identify anyone, and blur them before publishing.

## Code and git

- One script per step, with a docstring at the top giving its usage. Shared code goes in `common.py`.
- Scripts only download what is missing, so they are safe to run again.
- New script, data file or option → update `README.md` in the same commit.
- `init` is the main branch (Kaggle clones it). Commit messages: a short summary line, then what changed and why.
