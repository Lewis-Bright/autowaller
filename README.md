# Autowaller

Autowaller is a GM-controlled Foundry VTT module backed by an
asynchronous AWS wall-detection service.

## Status

This first implementation provides:

- an AWS SAM stack with an HTTP job API, private S3 storage, SQS worker queue,
  and DynamoDB job state;
- direct browser-to-S3 uploads using presigned URLs;
- a container-based Bedrock vision worker which identifies structural
  boundaries and then validates and simplifies its wall geometry;
- a one-click Foundry module which confirms, analyses, and applies walls;
- a versioned, Foundry-independent wall-plan schema.

The worker currently uses Claude Haiku 4.5 through an EU Bedrock inference
profile. Model output is constrained, bounds-checked, capped, cleared around
high-confidence entrances, and simplified before it reaches Foundry.

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
