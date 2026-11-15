# RGB pilot coverage plan — 14 October 2026

**The next data plan contains 60 fitting videos, 51 distinct subject/take groups
and 40,193 retained windows.** It keeps all original pilot rows, supplies positive
examples of every supervised activity in every new held-subject fold, and includes
all **28 pair-positive recording groups** available in the fitting inventory.
There are **638 TV/drinking co-labeled windows**, versus 562 in the original pilot.

This is an audited data plan, not a new accuracy result. No model was trained,
no GPU job was submitted, and no test arrays were loaded. The existing launcher
still requires four fitting videos per subject; the 60-video plan needs runtime
integration and a separately frozen training protocol before it can run.

## Coverage and retained data

| Quantity | Original pilot | Expanded plan |
|---|---:|---:|
| Fitting subjects | 9 | 9 |
| Whole fitting videos | 36 | 60 |
| Distinct subject/take groups | 30 | 51 |
| Fitting windows | 31,859 | 40,193 |
| Pair-positive recording groups | 24 | 28 |
| Pair-positive windows | 562 | 638 |
| Cleaning: subjects with positives | 1 | 4 |
| Hygiene: subjects with positives | 2 | 3 |
| Eating: subjects with positives | 3 | 5 |

The original 36 fitting videos already have verified RGB caches. The plan adds
**24 videos with 8,334 retained windows** requiring RGB extraction. All 77
validation videos were also extracted previously; their caches must be retrieved
and verified before reuse. Validation remains the complete 63,220-window p04/p15
split. Its examples did not influence video selection or fold assignment.

Every original positive example is retained. Multiple camera views in the original
caches remain available for fitting but count as the same recording for diversity.
All views of one subject stay in the same fold.

## What the full fitting audit changed

The inventory reconstructs labels for all **274 fitting videos / 217,090 windows**
from original annotations and the exact retained extraction positions. It reproduces
the full manifest aggregates and all 31,859 cached pilot rows, including starts,
labels, masks and sampled overlap. Annotation presence alone is not used as evidence
that an activity survived extraction.

Hygiene has only **96 positive windows**, confined to p03, p13 and p19. The old
fold A contains none of these subjects, so no selection of additional videos could
repair that fold's missing hygiene support. A separate next-experiment fold layout
was chosen using fitting labels and counts only:

| New fold | Held subjects |
|---|---|
| A | p03, p07, p25 |
| B | p06, p12, p19 |
| C | p09, p13, p17 |

The selector enumerates all 280 distinct partitions into three groups of three
subjects. Ninety supply every supervised class and every pair state in every
fold. Among those, it minimizes full-inventory held-row imbalance, then pair-window
imbalance, then lexicographic membership. These folds belong to the next plan;
the completed pair-head experiment and its rejected result are unchanged.

The full inventory contains only **711 co-labeled pair windows**. p17 has just
one pair-positive subject/take group. An initial rule requiring two such groups
from every subject was proven infeasible and saved with its source and recipe.
Increasing pilot size cannot satisfy that rule.

The revised recipe explicitly requires up to two available pair-positive takes
per subject, at least one for every subject, and at least six distinct positive
takes per fold. It records p17's availability shortfall. All original accuracy
and compute gates remain unchanged.

## How the plan is constructed

One bounded integer optimization selects a **36-video coverage core**: four whole
videos per subject, at most one camera per take, each supervised class positive
in at least three subjects and every fold/complement, all four pair states in
each subject, and three eligible probe subjects. It minimizes retained-window
cost after meeting those coverage constraints. Sorted inputs and the recorded
solver make replay inspectable; rank-sum ties mean frozen IDs are authoritative
across solver versions.

That core has only 17,117 windows and 322 pair positives. To preserve the larger
existing training sample, the plan takes its union with all original pilot videos.
It then adds one view of each of the four remaining pair-positive recordings:
P06T04, P12T07, P19T03 and P19T04. The view rule prefers more fitting pair labels,
then fewer retained windows, then video ID. No model scores enter this rule and
no additional model candidate is evaluated.

The final plan covers all available pair-positive takes, but does not include
every camera view or every one of the 711 full-inventory positive windows.
Coverage is still limited evidence: hygiene has just 22 positives in the plan,
including only two in fold A. Meeting nonzero-support rules does not establish
adequate statistical power or guarantee the required 30% pair scores.

## Compute planning

The original caps remain **900 seconds for setup/probing, 4,500 seconds for the
pilot, and 40,500 seconds for the session**, with the existing 600-second pilot-head
and 3,600-second full-head reserves. Probe IDs are frozen from the coverage core:
P25T01C05, P09T05C05 and P07T04C05. Each has 100–1,000 retained windows and a different
fitting subject. Fresh complete probes are required for a new run.

The old conservative window-rate estimator projects **72.59 minutes** for the
expanded pilot without cache reuse. This is close to the 75-minute cap and is
insufficient as a scheduling decision: new videos can have as much as 698.28 source
seconds per retained window, so sparse clips expose costs hidden by window counts.

With all existing fitting and validation caches verified and reused, a separate
planning sensitivity charges both historical elapsed/window and elapsed/decoded-frame
rates for the new videos, plus the full 900-second setup/probe allowance and
600-second head reserve. It estimates **35.14 minutes**. This is empirical planning
evidence, not a measured run or a guaranteed upper bound. The three core probes
are previously extracted videos; timing them alone will not verify sparse new-clip
costs. Runtime integration must explicitly profile sparse new videos and verify
cached features before a new bounded experiment.

## Implementation and verification

- [Inventory builder](../scripts/build_rgb_coverage_inventory.py) replays frozen
  retained-window supervision and refuses alternate roots that could attest code
  different from the imported helpers.
- [Coverage selector](../scripts/select_rgb_pilot.py) binds inventory facts to the
  completed builder audit, source and input hashes; records the recipe before
  optimization; independently checks selected counts; and distinguishes solver
  limits from proven infeasibility. Successful and rejected recipes are preserved.
- **48 isolated CPU tests pass**: 17 inventory tests and 31 selector tests, including
  malformed provenance/masks, immutable outputs, infeasible coverage, availability
  shortfalls, probe selection, solver limits and forged solutions.

Rebuild the same audited fitting inventory with:

```powershell
python scripts/build_rgb_coverage_inventory.py --output-dir artifacts/rgb-coverage-20261013/verified-20261014
python run_tests.py --suite test_rgb_coverage_inventory --suite test_rgb_pilot_selection
```

To produce another metadata selection from this same recipe, use a fresh output
directory with `scripts/select_rgb_pilot.py --inventory <inventory.json>
--output-dir <new-directory>`. Existing recipe/output files cannot be replaced.
This command performs no training or submission.

Private evidence is under `artifacts/rgb-coverage-20261013/`: `verified-20261014/`
contains the inventory/audit; `selection-20261014/` preserves the rejected strict
rule; `availability-20261014/` contains the accepted core, source/recipe, forecasts,
and final **`expanded_pilot.json`**. The 56-video intermediate union is retained
as `augmented_pilot.json`; it is superseded by the 60-video plan for future work.
