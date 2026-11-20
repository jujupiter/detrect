---
name: new-task
description: Start and finish a new detrect task the project's way, with a ticket in the Detrect GitHub Project, its own branch, and a pull request merged into init. Use whenever the user asks for a new piece of work (feature, fix, data or training job) that has no ticket or branch yet.
---

# New task: ticket, branch, pull request

`init` is the main branch and Kaggle clones it, so only merged work gets trained. Never commit a new job on
`init` or on another task's branch, and never rename a branch to reuse it: start a new one.

## 1. Confirm first

Before creating anything, show the user in a few lines:
- the ticket title and what it covers (a short checklist)
- the branch name: `<issue number>-<short-slug>`, e.g. `19-codecarbon`
- whether you will merge at the end, or stop at the pull request

Wait for their OK, unless they already said to go ahead with this task.

## 2. Ticket

```bash
gh issue list --state all --search "<keywords>"   # reuse an existing ticket if there is one
url=$(gh issue create --title "<title>" --body "<goal, then a - [ ] checklist>")
gh project item-add 1 --owner jujupiter --url "$url"   # project 1 = Detrect
```

## 3. Branch

```bash
git checkout init && git pull --ff-only
git checkout -b <number>-<slug>
```

## 4. Work

Follow CLAUDE.md: README in the same commit as a new script, data file or option; commit messages are a
short summary line, then what changed and why.

## 5. Pull request and merge

```bash
git push -u origin <branch>
gh pr create --base init --title "<title>" --body "<what changed and why>

Closes #<number>"
gh pr merge <pr> --merge --delete-branch
git checkout init && git pull --ff-only && git fetch --prune
```

Merge only once the change is checked:
- Code: it runs.
- Training or results: accuracy checked on a short run first ("results first" in CLAUDE.md). Until then,
  stop at the pull request and say what is left to check.

Finish by telling the user the ticket, the pull request and whether it was merged.
