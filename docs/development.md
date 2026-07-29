# Development

## Prerequisites

- Python 3.12
- Node.js 20 or newer (for JavaScript syntax checking)
- Docker
- AWS SAM CLI (for building and deploying the stack)

## Detector tests

From `foundry-autowaller`:

```powershell
docker build --tag foundry-autowaller-worker:test aws\functions\worker
docker run --rm `
  --volume "${PWD}:/workspace" `
  --workdir /workspace `
  --env PYTHONPATH=/var/task `
  --entrypoint python `
  foundry-autowaller-worker:test `
  -m unittest discover -s tests -v
```

The worker container can be built through SAM:

```powershell
sam build --template-file aws\template.yaml
```

## Result contract

`schemas/wall-plan.schema.json` is the boundary between cloud analysis and the
Foundry module. Detection implementations may change, but existing fields must
retain their meaning. Increment `schemaVersion` for breaking changes.

Coordinates in a wall plan are local to Foundry's unpadded scene rectangle.
The module adds the current canvas scene offset when creating Wall documents.

## Detector limitations

The current detector uses Claude Haiku 4.5 through Amazon Bedrock, followed by
deterministic validation and geometry simplification. It can still misclassify
ambiguous artwork and does not yet create Foundry door or window document
types. High-confidence entrances are left as gaps.
