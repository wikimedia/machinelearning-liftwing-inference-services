"""S3 sink tests (moto-mocked; no network).

Mirrors the FileSink suite (store contract, key layout, ContentType,
overwrite idempotence) plus the startup-probe failure classes: every
misconfiguration must fail at SINK CONSTRUCTION (i.e. deploy time via
build_sink in the service lifespan), never at first write.

Against real infrastructure the same behavior was validated with MinIO
(docker run -p 9000:9000 minio/minio server /data); moto keeps CI
network-free.
"""

import base64

import boto3
import pytest
from moto import mock_aws

from src.models.tts_section_generator.tts_generator.sinks import (
    S3Sink,
    SinkWriteError,
    artifact_key,
)

# moto 5 intercepts by matching AWS URL patterns; a custom endpoint
# (thanos-swift...) would fall through to a real connection. Unit tests
# therefore use an AWS-shaped endpoint; the real-endpoint wire behavior
# (path-style against Swift/MinIO) is covered by the MinIO integration
# pass, not CI.
ENDPOINT = "https://s3.us-east-1.amazonaws.com"
BUCKET = "tts-artifacts-test"


@pytest.fixture()
def s3_bucket():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield


def test_store_returns_s3_uri_and_size(s3_bucket):
    sink = S3Sink(ENDPOINT, BUCKET)
    key = artifact_key("enwiki", 9228, 1362915217, "upper-atmosphere", "audio_opus")
    out = sink.store(key, b"OggS" + b"\x00" * 100, "audio/ogg; codecs=opus")
    assert out == {
        "blob_uri": f"s3://{BUCKET}/enwiki/9228/1362915217/upper-atmosphere.opus",
        "size_bytes": 104,
    }


def test_object_lands_with_content_type(s3_bucket):
    sink = S3Sink(ENDPOINT, BUCKET)
    sink.store("enwiki/1/2/lead.vtt", b"WEBVTT\n", "text/vtt")
    obj = boto3.client("s3", region_name="us-east-1").get_object(
        Bucket=BUCKET, Key="enwiki/1/2/lead.vtt"
    )
    assert obj["Body"].read() == b"WEBVTT\n"
    assert obj["ContentType"] == "text/vtt"


def test_overwrite_is_idempotent(s3_bucket):
    sink = S3Sink(ENDPOINT, BUCKET)
    sink.store("enwiki/1/2/lead.opus", b"one", "audio/ogg")
    out = sink.store("enwiki/1/2/lead.opus", b"one", "audio/ogg")
    assert out["size_bytes"] == 3
    body = (
        boto3.client("s3", region_name="us-east-1")
        .get_object(Bucket=BUCKET, Key="enwiki/1/2/lead.opus")["Body"]
        .read()
    )
    assert body == b"one"


def test_binary_roundtrip_is_exact(s3_bucket):
    sink = S3Sink(ENDPOINT, BUCKET)
    payload = base64.b64decode(base64.b64encode(bytes(range(256)) * 41))
    sink.store("enwiki/1/2/x.opus", payload, "audio/ogg")
    body = (
        boto3.client("s3", region_name="us-east-1")
        .get_object(Bucket=BUCKET, Key="enwiki/1/2/x.opus")["Body"]
        .read()
    )
    assert body == payload


# -- Startup-probe failure classes: fail the deploy, not request #1 ---------


def test_missing_config_fails_at_construction():
    with pytest.raises(RuntimeError, match="requires TTS_GEN_S3_ENDPOINT"):
        S3Sink("", "")


def test_absent_bucket_fails_at_construction():
    with mock_aws():  # endpoint answers, bucket does not exist
        with pytest.raises(RuntimeError, match="startup check failed"):
            S3Sink(ENDPOINT, "no-such-bucket")


def test_write_failure_raises_sink_write_error(s3_bucket, monkeypatch):
    sink = S3Sink(ENDPOINT, BUCKET)

    from botocore.exceptions import ClientError

    def boom(**kwargs):
        raise ClientError(
            {"Error": {"Code": "503", "Message": "slow down"}}, "PutObject"
        )

    monkeypatch.setattr(sink._client, "put_object", boom)
    with pytest.raises(SinkWriteError, match="put_object failed"):
        sink.store("enwiki/1/2/lead.opus", b"x", "audio/ogg")
