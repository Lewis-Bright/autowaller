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

The v1 detector finds long straight boundaries using Canny edges and a
probabilistic Hough transform. It does not yet:

- distinguish doors or windows;
- understand curved walls;
- infer openings hidden by map art;
- merge all parallel edges of a thick wall into a centre line.

Those improvements belong in the worker and do not require an API/module
redesign.
