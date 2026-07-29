import json
import os
import time

import boto3
from botocore.exceptions import ClientError

from bedrock_detector import detect_walls_with_bedrock


TABLE_NAME = os.environ["JOB_TABLE"]
BUCKET_NAME = os.environ["ARTIFACT_BUCKET"]

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
s3 = boto3.client("s3")


def _process(job_id):
    item = table.get_item(Key={"jobId": job_id}, ConsistentRead=True).get("Item")
    if not item:
        raise ValueError(f"Unknown job {job_id}")
    if item["status"] == "complete":
        return

    try:
        table.update_item(
            Key={"jobId": job_id},
            UpdateExpression="SET #status = :processing, startedAt = :now",
            ConditionExpression="#status IN (:queued, :failed)",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":processing": "processing",
                ":queued": "queued",
                ":failed": "failed",
                ":now": int(time.time()),
            },
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return
        raise
    source = s3.get_object(Bucket=BUCKET_NAME, Key=item["inputKey"])["Body"].read()
    result = detect_walls_with_bedrock(
        source, int(item["width"]), int(item["height"])
    )
    result_key = f"jobs/{job_id}/wall-plan.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=result_key,
        Body=json.dumps(result, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
    )
    table.update_item(
        Key={"jobId": job_id},
        UpdateExpression=(
            "SET #status = :complete, resultKey = :result, "
            "completedAt = :now, wallCount = :count"
        ),
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={
            ":complete": "complete",
            ":result": result_key,
            ":now": int(time.time()),
            ":count": len(result["walls"]),
        },
    )


def lambda_handler(event, _context):
    failures = []
    for record in event.get("Records", []):
        job_id = None
        try:
            job_id = json.loads(record["body"])["jobId"]
            _process(job_id)
        except Exception as exc:
            if job_id:
                table.update_item(
                    Key={"jobId": job_id},
                    UpdateExpression="SET #status = :failed, #error = :error, failedAt = :now",
                    ExpressionAttributeNames={"#status": "status", "#error": "error"},
                    ExpressionAttributeValues={
                        ":failed": "failed",
                        ":error": str(exc)[:500],
                        ":now": int(time.time()),
                    },
                )
            failures.append({"itemIdentifier": record["messageId"]})
    return {"batchItemFailures": failures}
