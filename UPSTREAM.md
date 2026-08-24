# Upstream provenance and synchronization

`LocalStrayRuntime` is an independent, owner-controlled mirror seeded from
[`adriancmurray/qwen-prime-runtime`](https://github.com/adriancmurray/qwen-prime-runtime)
at commit `4217dd9a071cf094a46a5d31f3c08f68acf2d8c3` on 2026-08-24.

The upstream source is public under the MIT License. Keep its license, notices,
and attribution intact when synchronizing. Local Stray releases use this mirror
as their stable runtime source; upstream remains the source to monitor for
runtime improvements and security fixes.

## Synchronizing a new upstream revision

Clone this repository and configure the original repository as a read-only
upstream remote:

```bash
git clone https://github.com/titofebus/LocalStrayRuntime.git
cd LocalStrayRuntime
git remote add upstream https://github.com/adriancmurray/qwen-prime-runtime.git
git fetch upstream --tags
git log --oneline origin/main..upstream/main
```

Review the incoming commits, dependencies, license notices, and focused runtime
tests before merging them. Then fast-forward `main`, push the reviewed result,
and mirror any new upstream tags:

```bash
git checkout main
git merge --ff-only upstream/main
uv sync --extra dev
uv run pytest
git push origin main
git push origin --follow-tags
```

Do not force-push the mirror or silently replace reviewed Local Stray runtime
revisions. If upstream rewrites history, investigate and create a deliberate
reviewed commit rather than mirroring the rewrite automatically.
