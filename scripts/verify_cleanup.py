#!/usr/bin/env python3
"""
Profit Sentinel - Cleanup Verification Script

Run this AFTER destroy_test_data.py to verify all test data has been removed.

Usage:
    python scripts/verify_cleanup.py

Prerequisites:
    pip install boto3
    AWS credentials configured (~/.aws/credentials or environment variables)

Output:
    Prints verification results and overall PASS/FAIL status
"""

import os
import sys
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")
LOG_GROUPS = ["/ecs/profitsentinel-dev"]
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


class VerificationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, check: str, details: str = ""):
        self.passed.append((check, details))
        print(f"  [PASS] {check}" + (f" - {details}" if details else ""))

    def add_fail(self, check: str, details: str):
        self.failed.append((check, details))
        print(f"  [FAIL] {check} - {details}")

    def add_warning(self, check: str, details: str):
        self.warnings.append((check, details))
        print(f"  [WARN] {check} - {details}")

    def is_clean(self) -> bool:
        return len(self.failed) == 0


def verify_s3_clean(result: VerificationResult):
    """Verify S3 bucket is completely clean."""
    print("\n" + "=" * 50)
    print("S3 BUCKET VERIFICATION")
    print("=" * 50)

    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        # Check for current objects
        response = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1)

        if "Contents" in response:
            result.add_fail(
                "Current objects",
                f"Found {response.get('KeyCount', 1)} objects still in bucket",
            )
        else:
            result.add_pass("Current objects", "Bucket is empty")

        # Check for versions
        versioning = s3.get_bucket_versioning(Bucket=S3_BUCKET)
        if versioning.get("Status") == "Enabled":
            versions = s3.list_object_versions(Bucket=S3_BUCKET, MaxKeys=1)

            version_count = len(versions.get("Versions", []))
            marker_count = len(versions.get("DeleteMarkers", []))

            if version_count > 0:
                result.add_fail(
                    "Object versions", f"Found {version_count} versions still exist"
                )
            else:
                result.add_pass("Object versions", "No versions found")

            if marker_count > 0:
                result.add_warning(
                    "Delete markers",
                    f"Found {marker_count} delete markers (harmless but can be removed)",
                )
            else:
                result.add_pass("Delete markers", "No delete markers")
        else:
            result.add_pass("Versioning", "Disabled (no versions to check)")

        # Check lifecycle rule
        try:
            lifecycle = s3.get_bucket_lifecycle_configuration(Bucket=S3_BUCKET)
            for rule in lifecycle.get("Rules", []):
                if "NoncurrentVersionExpiration" in rule:
                    days = rule["NoncurrentVersionExpiration"].get("NoncurrentDays")
                    if days and days > 7:
                        result.add_warning(
                            "Lifecycle rule",
                            f"Non-current versions retained for {days} days (consider reducing)",
                        )
                    else:
                        result.add_pass(
                            "Lifecycle rule", f"Versions expire after {days} days"
                        )
        except ClientError:
            result.add_warning("Lifecycle rule", "No lifecycle configuration found")

    except NoCredentialsError:
        result.add_fail("AWS credentials", "Not configured")
    except ClientError as e:
        result.add_fail("S3 access", str(e))


def verify_cloudwatch_clean(result: VerificationResult):
    """Verify CloudWatch logs are clean."""
    print("\n" + "=" * 50)
    print("CLOUDWATCH LOGS VERIFICATION")
    print("=" * 50)

    logs = boto3.client("logs", region_name=AWS_REGION)

    for group_name in LOG_GROUPS:
        try:
            # Check if log group exists
            response = logs.describe_log_groups(logGroupNamePrefix=group_name)
            if not response.get("logGroups"):
                result.add_pass(f"Log group {group_name}", "Does not exist")
                continue

            log_group = response["logGroups"][0]

            # Check retention
            retention = log_group.get("retentionInDays")
            if retention is None:
                result.add_warning(
                    f"Log group {group_name}",
                    "No retention policy (logs never expire)",
                )
            elif retention > 7:
                result.add_warning(
                    f"Log group {group_name}",
                    f"Retention is {retention} days (consider reducing)",
                )
            else:
                result.add_pass(
                    f"Log group {group_name}", f"Retention set to {retention} days"
                )

            # Check for remaining log streams
            streams = logs.describe_log_streams(
                logGroupName=group_name, limit=5, orderBy="LastEventTime", descending=True
            )

            stream_count = len(streams.get("logStreams", []))
            if stream_count > 0:
                result.add_warning(
                    f"Log streams in {group_name}",
                    f"Found {stream_count} log streams (may contain old data)",
                )
            else:
                result.add_pass(f"Log streams in {group_name}", "No log streams")

        except ClientError as e:
            result.add_fail(f"Log group {group_name}", str(e))


def verify_rds_snapshots(result: VerificationResult):
    """Check for RDS snapshots (informational)."""
    print("\n" + "=" * 50)
    print("RDS SNAPSHOTS VERIFICATION")
    print("=" * 50)

    rds = boto3.client("rds", region_name=AWS_REGION)

    try:
        # Check for manual snapshots
        response = rds.describe_db_cluster_snapshots(SnapshotType="manual")

        manual_count = len(response.get("DBClusterSnapshots", []))
        if manual_count > 0:
            result.add_warning(
                "Manual RDS snapshots",
                f"Found {manual_count} manual snapshots (review manually)",
            )
            for snap in response["DBClusterSnapshots"]:
                print(f"    - {snap['DBClusterSnapshotIdentifier']}")
        else:
            result.add_pass("Manual RDS snapshots", "None found")

    except ClientError as e:
        if "DBClusterNotFoundFault" in str(e):
            result.add_pass("RDS cluster", "Not deployed")
        else:
            result.add_warning("RDS access", str(e))


def print_summary(result: VerificationResult):
    """Print verification summary."""
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    print(f"\n  Passed: {len(result.passed)}")
    print(f"  Failed: {len(result.failed)}")
    print(f"  Warnings: {len(result.warnings)}")

    if result.failed:
        print("\n" + "!" * 50)
        print("VERIFICATION FAILED - Issues found:")
        print("!" * 50)
        for check, details in result.failed:
            print(f"  - {check}: {details}")
        print("\nRun destroy_test_data.py again to address failures.")

    elif result.warnings:
        print("\n" + "-" * 50)
        print("VERIFICATION PASSED with warnings:")
        print("-" * 50)
        for check, details in result.warnings:
            print(f"  - {check}: {details}")
        print("\nReview warnings and address if necessary.")

    else:
        print("\n" + "=" * 50)
        print("VERIFICATION PASSED")
        print("All checks passed - production is clean!")
        print("=" * 50)

    return result.is_clean()


def main():
    """Main entry point."""
    print("\n" + "=" * 50)
    print("PROFIT SENTINEL - CLEANUP VERIFICATION")
    print("=" * 50)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"Region: {AWS_REGION}")

    result = VerificationResult()

    verify_s3_clean(result)
    verify_cloudwatch_clean(result)
    verify_rds_snapshots(result)

    is_clean = print_summary(result)

    print("\n" + "=" * 50)
    print("MANUAL CHECKS STILL REQUIRED:")
    print("=" * 50)
    print("1. [ ] Supabase: Check email_signups table for test emails")
    print("2. [ ] Supabase: Review analysis_synopses if concerned")
    print("3. [ ] Email provider: Check for sent reports containing test data")
    print("4. [ ] Git history: Ensure no secrets were committed")

    print("\n" + "=" * 50)

    sys.exit(0 if is_clean else 1)


if __name__ == "__main__":
    main()
