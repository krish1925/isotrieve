# GitHub Action: Isotrieve Gate

Run the Isotrieve quality gate on embedding config changes directly in CI.

## Usage

```yaml
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: krish1925/isotrieve/.github/actions/gate-action@main
        with:
          mapping-path: mapping.isotrieve
          source-vectors: source.npy
          target-vectors: target.npy
          format: json
          junit: "true"
```

A copy of this lives in this repo at `.github/workflows/gate-template.yml` —
copy it into your own repository to get started.

## Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `mapping-path` | yes | — | Path to the `.isotrieve` mapping file |
| `source-vectors` | yes | — | NPY of source embeddings |
| `target-vectors` | yes | — | NPY of target embeddings |
| `format` | no | `md` | Report format: `json`, `md`, `html` |
| `junit` | no | `true` | Emit a JUnit XML summary (`gate-results.xml`) |
| `fail-on-warn` | no | `false` | Exit 1 on WARN (not just FAIL) |

## Outputs

- `gate-report.<format>` — the gate report artifact.
- `gate-results.xml` — a JUnit summary when `junit` is enabled, for use with
  GitHub's test result reporting.

## Example repositories

The gate action is used end-to-end in the main `krish1925/isotrieve`
repository CI. The minimal standalone starter is the
`.github/workflows/gate-template.yml` workflow in this repository — copy it
into your own repo, point the three input paths at your mapping and NPYs, and
merge it as your migration-CI gate.
