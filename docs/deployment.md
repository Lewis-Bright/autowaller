# Deployment

## 1. Deploy the AWS stack

Install Docker and the AWS SAM CLI, authenticate the AWS CLI, then run from the
repository root:

```powershell
sam build --template-file autowaller\aws\template.yaml
sam deploy --guided --region eu-west-1
```

Choose a long random value (at least 24 characters) for `ApiKey`. Do not commit
it. SAM builds the vision worker as a Lambda container image.

`DetectorMode` defaults to `bedrock`, where the vision model supplies wall
coordinates. For an isolated OpenCV/Bedrock comparison stack, deploy the same
template under a different stack name with `DetectorMode=hybrid`. The hybrid
worker detects and labels whole-map contours with OpenCV, then asks Bedrock to
select the genuine floor-facing structural candidates. Each stack has its own
API, queues, table, bucket, and worker, so switching the module's service URL
does not mix their jobs.

The worker invokes Claude Haiku 4.5 through the EU Amazon Bedrock inference
profile. Before the first deployment, submit the one-time Anthropic use-case
details and activate its usage-based model agreement in Amazon Bedrock. No
provisioned throughput is required. The worker role receives only
`bedrock:InvokeModel` in addition to its existing job and artifact permissions.

The deployment outputs an `ApiUrl`. The S3 bucket remains private; map uploads
and wall-plan downloads use short-lived presigned URLs.

The initial API key is suitable for a private installation. Before public
release, replace it with per-user authentication and quotas.

## 2. Install the Foundry module

Copy `autowaller/module` to the Foundry user-data modules directory as:

```text
Data/modules/autowaller/
```

Restart Foundry and enable **Autowaller** in the world. Under
**Configure Settings → Module Settings**, enter the stack's `ApiUrl` output and
the private Autowaller API key.

The module declares Foundry v14 as its minimum and verified version.

## 3. Use it

1. Open a Scene with a PNG, JPEG, or WebP background.
2. Open Wall Controls and select **Auto Wall**.
3. Confirm that the current Scene should be processed.
4. Wait for analysis to finish; qualifying walls are applied immediately.

Applied walls are immediately active and are tagged with the analysis job ID.

## Operational defaults

- Worker timeout: 15 minutes
- Worker memory: 3,008 MB
- Worker concurrency: governed by the account Lambda concurrency quota
- Artifact retention: 7 days
- Upload URL validity: 15 minutes
- Result URL validity: 5 minutes
- SQS retry count before dead-lettering: 3

For a public launch, add user authentication, per-user rate limits, billing
records, malware/file validation, usage alarms, and a privacy policy before
raising worker concurrency.
