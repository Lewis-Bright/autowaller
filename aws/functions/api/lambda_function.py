import base64
import json
import os
import time
import uuid

import boto3
from botocore.exceptions import ClientError


TABLE_NAME = os.environ["JOB_TABLE"]
BUCKET_NAME = os.environ["ARTIFACT_BUCKET"]
QUEUE_URL = os.environ["JOB_QUEUE_URL"]
API_KEY = os.environ["AUTOWALL_API_KEY"]

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
s3 = boto3.client("s3")
sqs = boto3.client("sqs")


def _response(status, body):
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "cache-control": "no-store",
        },
        "body": json.dumps(body),
    }


def _body(event):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    return json.loads(raw)


def _authorized(event):
    headers = {key.lower(): value for key, value in (event.get("headers") or {}).items()}
    return headers.get("authorization") == f"Bearer {API_KEY}"


def _create_job(event):
    payload = _body(event)
    width = int(payload.get("width", 0))
    height = int(payload.get("height", 0))
    content_type = payload.get("contentType", "")

    if width < 1 or height < 1 or width > 20000 or height > 20000:
        return _response(400, {"error": "Scene dimensions must be between 1 and 20000 pixels."})
    if content_type not in {"image/png", "image/jpeg", "image/webp"}:
        return _response(400, {"error": "Only PNG, JPEG, and WebP maps are supported."})

    job_id = str(uuid.uuid4())
    now = int(time.time())
    input_key = f"jobs/{job_id}/input"
    table.put_item(
        Item={
            "jobId": job_id,
            "status": "awaiting_upload",
            "inputKey": input_key,
            "contentType": content_type,
            "width": width,
            "height": height,
            "createdAt": now,
            "expiresAt": now + 7 * 86400,
        },
        ConditionExpression="attribute_not_exists(jobId)",
    )
    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": BUCKET_NAME,
            "Key": input_key,
            "ContentType": content_type,
        },
        ExpiresIn=900,
    )
    return _response(201, {"jobId": job_id, "uploadUrl": upload_url, "expiresIn": 900})


def _start_job(job_id):
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=f"jobs/{job_id}/input")
        table.update_item(
            Key={"jobId": job_id},
            UpdateExpression="SET #status = :queued, queuedAt = :now",
            ConditionExpression="#status = :awaiting",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":queued": "queued",
                ":awaiting": "awaiting_upload",
                ":now": int(time.time()),
            },
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
        return _response(409, {"error": "The map has not been uploaded."})
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        item = table.get_item(Key={"jobId": job_id}).get("Item")
        if not item:
            return _response(404, {"error": "Job not found."})
        return _response(409, {"error": f"Job is already {item['status']}."})

    try:
        sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps({"jobId": job_id}))
    except Exception:
        table.update_item(
            Key={"jobId": job_id},
            UpdateExpression="SET #status = :awaiting REMOVE queuedAt",
            ConditionExpression="#status = :queued",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":awaiting": "awaiting_upload", ":queued": "queued"},
        )
        raise
    return _response(202, {"jobId": job_id, "status": "queued"})


def _get_job(job_id):
    item = table.get_item(Key={"jobId": job_id}).get("Item")
    if not item:
        return _response(404, {"error": "Job not found."})

    result = {"jobId": job_id, "status": item["status"]}
    if item["status"] == "complete":
        result["resultUrl"] = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": item["resultKey"]},
            ExpiresIn=300,
        )
    if item["status"] == "failed":
        result["error"] = item.get("error", "Wall detection failed.")
    return _response(200, result)


def lambda_handler(event, _context):
    if not _authorized(event):
        return _response(401, {"error": "Unauthorized."})

    request = event["requestContext"]["http"]
    method = request["method"]
    path = request["path"].rstrip("/")
    path_parameters = event.get("pathParameters") or {}
    job_id = path_parameters.get("jobId")

    try:
        if method == "POST" and path.endswith("/jobs"):
            return _create_job(event)
        if method == "POST" and path.endswith("/start") and job_id:
            return _start_job(job_id)
        if method == "GET" and job_id:
            return _get_job(job_id)
        return _response(404, {"error": "Route not found."})
    except (ValueError, TypeError, json.JSONDecodeError):
        return _response(400, {"error": "Invalid request."})
