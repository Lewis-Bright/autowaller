# Foundry Autowaller

Foundry Autowaller is a GM-controlled Foundry VTT module backed by an
asynchronous AWS wall-detection service.

## Status

This first implementation provides:

- an AWS SAM stack with an HTTP job API, private S3 storage, SQS worker queue,
  and DynamoDB job state;
- direct browser-to-S3 uploads using presigned URLs;
- a container-based Python/OpenCV worker which detects and simplifies long
  straight wall segments;
- a one-click Foundry module which confirms, analyses, and applies walls;
- a versioned, Foundry-independent wall-plan schema.

The detector is deliberately deterministic. A vision-model classifier can be
added later without changing the module/API contract.

## Layout

```text
aws/       SAM infrastructure, API Lambda, and worker
module/    installable Foundry VTT module
schemas/   service result contracts
tests/     local unit tests
```

The module has restricted world settings for its service URL and private API
key. See [docs/development.md](docs/development.md)
for local tests and [docs/deployment.md](docs/deployment.md) for AWS and
Foundry setup.
