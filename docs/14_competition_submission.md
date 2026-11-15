# PromptWars competition repository

Use [PARTHG0106/BehaviorSense-PromptWars](https://github.com/PARTHG0106/BehaviorSense-PromptWars) for the competition repository link and public project posts.

The competition requires a public repository, exactly one branch, and a repository smaller than 10 MB. This source snapshot uses a single `main` branch. The packaging check requires fewer than **10,000,000 bytes** for both tracked source files and a fresh clone including Git metadata. Node dependencies, private databases, model assets and generated audit output are excluded from Git.

## Included project files

The repository includes the complete `src/`, `web/`, `scripts/`, `configs/`, `tests/` and `notebooks/` trees, dependency files, Dockerfile, license, documentation and research reports. The supplied [Kaggle backend notebook](../notebooks/notebookef04f4f9bb%20%281%29.ipynb) uses the saved RGB hybrid on T4 ×2.

Trained model weights and detector weights are obtained from the two public Kaggle datasets in the [README](../README.md#run-the-project). Development conversation archives and operator settings are excluded from this package. Generated artifacts and model files are ignored by Git.

The activity/model baseline comes from original commit `778785d2829233cdabe0e0a75da65df7d8d59951`. The [validation record](../results/submission_validation_20261115.md) covers that baseline. The competition snapshot also includes [shared care management](15_care_management.md) and the [accessibility improvements](16_accessibility.md), with separate API and browser tests. The dependency lists explicitly include the multipart upload parser and coordinator HTTP library. The test runner resolves temporary-directory aliases for clean Windows environments.

The final care update makes the care platform the homepage and gives activity reports their own page. It adds a demo walkthrough, daily confirmation counts, immutable caregiver notes and printing. Its [targeted validation](../results/care_submission_validation_20261115.md) passed 61/61 checks and all six browser accessibility views.

Earlier development history remains available in the [original project](https://github.com/PARTHG0106/BehaviorSense). This snapshot starts its own history to keep the submission package small.

## Check before submitting

Verify the public status and branch list:

```powershell
gh repo view PARTHG0106/BehaviorSense-PromptWars --json visibility,diskUsage,defaultBranchRef
git ls-remote --heads origin
```

The branch list should contain only `refs/heads/main`. GitHub's `diskUsage` value is in KB; the source-package measurement also checks the uncompressed files. A ZIP of tracked source can be created from the repository root:

```powershell
git archive --format=zip --output=../BehaviorSense-PromptWars.zip HEAD
```

The competition permits a maximum of two submission attempts. Use the verified repository URL when completing the competition form. The README distinguishes caregiver-reported intake, scheduled reminders and optional video observations.
