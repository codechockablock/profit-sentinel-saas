#!/usr/bin/env python3
"""
Profit Sentinel - Test Data Search Script

This script searches production AWS resources for traces of test data.
Run this at home with AWS credentials configured.

Usage:
    python scripts/find_test_data.py

Prerequisites:
    pip install boto3
    AWS credentials configured (~/.aws/credentials or environment variables)

Output:
    Prints findings to console and saves to find_test_data_report.txt
"""

import json
import os
import sys
from datetime import datetime, timedelta

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# =============================================================================
# CONFIGURATION - Update these with actual values from your AWS environment
# =============================================================================

# S3 bucket name (from Terraform: profitsentinel-dev-uploads)
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")

# CloudWatch log groups to search
LOG_GROUPS = [
    "/ecs/profitsentinel-dev",
    # Add any other log groups here
]

# RDS cluster identifier
RDS_CLUSTER_ID = "profitsentinel-dev-cluster"

# AWS region
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Report file
REPORT_FILE = f"find_test_data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================


def log_finding(message: str, level: str = "INFO"):
    """Log a finding to console and report file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    print(line)
    with open(REPORT_FILE, "a") as f:
        f.write(line + "\n")


def check_s3_bucket():
    """Check S3 bucket for any objects (current and versions)."""
    log_finding("=" * 60)
    log_finding("S3 BUCKET CHECK")
    log_finding("=" * 60)

    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        # Check bucket exists
        s3.head_bucket(Bucket=S3_BUCKET)
        log_finding(f"Bucket '{S3_BUCKET}' exists and is accessible")

        # List current objects
        log_finding("\n--- Current Objects ---")
        response = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=100)

        if "Contents" in response:
            count = len(response["Contents"])
            log_finding(f"WARNING: Found {count} objects in bucket", "WARNING")
            for obj in response["Contents"][:20]:  # Show first 20
                log_finding(
                    f"  - {obj['Key']} "
                    f"(Size: {obj['Size']} bytes, "
                    f"Modified: {obj['LastModified']})"
                )
            if count > 20:
                log_finding(f"  ... and {count - 20} more objects")
        else:
            log_finding("No current objects in bucket (good)", "PASS")

        # Check versioning status
        log_finding("\n--- Versioning Status ---")
        versioning = s3.get_bucket_versioning(Bucket=S3_BUCKET)
        status = versioning.get("Status", "Disabled")
        log_finding(f"Versioning status: {status}")

        if status == "Enabled":
            log_finding(
                "WARNING: Versioning is ENABLED - deleted files may still exist!",
                "WARNING",
            )

            # List all versions
            log_finding("\n--- Object Versions ---")
            versions_response = s3.list_object_versions(Bucket=S3_BUCKET, MaxKeys=100)

            version_count = len(versions_response.get("Versions", []))
            delete_marker_count = len(versions_response.get("DeleteMarkers", []))

            if version_count > 0:
                log_finding(
                    f"CRITICAL: Found {version_count} object versions!", "CRITICAL"
                )
                for ver in versions_response.get("Versions", [])[:20]:
                    is_latest = "CURRENT" if ver.get("IsLatest") else "NON-CURRENT"
                    log_finding(
                        f"  [{is_latest}] {ver['Key']} "
                        f"(Version: {ver['VersionId'][:8]}..., "
                        f"Modified: {ver['LastModified']})"
                    )
            else:
                log_finding("No object versions found (good)", "PASS")

            if delete_marker_count > 0:
                log_finding(f"Found {delete_marker_count} delete markers", "INFO")
                for marker in versions_response.get("DeleteMarkers", [])[:10]:
                    log_finding(f"  - {marker['Key']} (deleted at {marker['LastModified']})")

        # Check lifecycle rules
        log_finding("\n--- Lifecycle Rules ---")
        try:
            lifecycle = s3.get_bucket_lifecycle_configuration(Bucket=S3_BUCKET)
            for rule in lifecycle.get("Rules", []):
                log_finding(f"Rule: {rule.get('ID', 'unnamed')}")
                if "NoncurrentVersionExpiration" in rule:
                    days = rule["NoncurrentVersionExpiration"].get("NoncurrentDays")
                    log_finding(f"  Non-current versions expire after: {days} days")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
                log_finding("No lifecycle rules configured")
            else:
                raise

    except ClientError as e:
        log_finding(f"ERROR accessing S3: {e}", "ERROR")
    except NoCredentialsError:
        log_finding("ERROR: AWS credentials not configured", "ERROR")


def check_cloudwatch_logs():
    """Search CloudWatch logs for traces of test data."""
    log_finding("\n" + "=" * 60)
    log_finding("CLOUDWATCH LOGS CHECK")
    log_finding("=" * 60)

    logs = boto3.client("logs", region_name=AWS_REGION)

    # Search patterns that might indicate test data
    search_patterns = [
        "inventory",
        "test",
        "sample",
        ".csv",
        ".xlsx",
        "anonymous/",
    ]

    for group_name in LOG_GROUPS:
        log_finding(f"\n--- Log Group: {group_name} ---")

        try:
            # Check if log group exists
            response = logs.describe_log_groups(logGroupNamePrefix=group_name)
            if not response.get("logGroups"):
                log_finding(f"Log group not found: {group_name}")
                continue

            log_group = response["logGroups"][0]
            log_finding(f"Retention: {log_group.get('retentionInDays', 'NEVER EXPIRE')}")
            log_finding(f"Stored bytes: {log_group.get('storedBytes', 0):,}")

            # List recent log streams
            streams = logs.describe_log_streams(
                logGroupName=group_name,
                orderBy="LastEventTime",
                descending=True,
                limit=10,
            )

            log_finding(f"Recent log streams: {len(streams.get('logStreams', []))}")

            # Search for patterns in recent logs (last 7 days)
            start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)

            log_finding("\nSearching for test data patterns...")
            for pattern in search_patterns:
                try:
                    result = logs.filter_log_events(
                        logGroupName=group_name,
                        startTime=start_time,
                        endTime=end_time,
                        filterPattern=f'"{pattern}"',
                        limit=5,
                    )

                    if result.get("events"):
                        log_finding(
                            f"WARNING: Found {len(result['events'])}+ matches for '{pattern}'",
                            "WARNING",
                        )
                        for event in result["events"][:3]:
                            msg = event["message"][:100].replace("\n", " ")
                            log_finding(f"  - {msg}...")
                except ClientError:
                    pass  # Pattern not found or access denied

        except ClientError as e:
            log_finding(f"ERROR accessing log group: {e}", "ERROR")


def check_rds_snapshots():
    """Check for RDS snapshots that might contain test data."""
    log_finding("\n" + "=" * 60)
    log_finding("RDS SNAPSHOTS CHECK")
    log_finding("=" * 60)

    rds = boto3.client("rds", region_name=AWS_REGION)

    try:
        # List automated snapshots
        log_finding("\n--- Automated Snapshots ---")
        auto_snapshots = rds.describe_db_cluster_snapshots(
            DBClusterIdentifier=RDS_CLUSTER_ID,
            SnapshotType="automated",
        )

        if auto_snapshots.get("DBClusterSnapshots"):
            count = len(auto_snapshots["DBClusterSnapshots"])
            log_finding(f"Found {count} automated snapshots")
            for snap in auto_snapshots["DBClusterSnapshots"][:5]:
                log_finding(
                    f"  - {snap['DBClusterSnapshotIdentifier']} "
                    f"(Created: {snap['SnapshotCreateTime']})"
                )
        else:
            log_finding("No automated snapshots found")

        # List manual snapshots
        log_finding("\n--- Manual Snapshots ---")
        manual_snapshots = rds.describe_db_cluster_snapshots(
            DBClusterIdentifier=RDS_CLUSTER_ID,
            SnapshotType="manual",
        )

        if manual_snapshots.get("DBClusterSnapshots"):
            count = len(manual_snapshots["DBClusterSnapshots"])
            log_finding(f"WARNING: Found {count} manual snapshots", "WARNING")
            for snap in manual_snapshots["DBClusterSnapshots"]:
                log_finding(
                    f"  - {snap['DBClusterSnapshotIdentifier']} "
                    f"(Created: {snap['SnapshotCreateTime']})"
                )
            log_finding(
                "Review these snapshots to determine if they contain test data",
                "WARNING",
            )
        else:
            log_finding("No manual snapshots found")

    except ClientError as e:
        if "DBClusterNotFoundFault" in str(e):
            log_finding("RDS cluster not found (may not be deployed yet)")
        else:
            log_finding(f"ERROR accessing RDS: {e}", "ERROR")


def check_dynamodb_tables():
    """Check if any DynamoDB tables exist and have data."""
    log_finding("\n" + "=" * 60)
    log_finding("DYNAMODB CHECK")
    log_finding("=" * 60)

    dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)

    try:
        tables = dynamodb.list_tables()

        if tables.get("TableNames"):
            log_finding(f"Found {len(tables['TableNames'])} DynamoDB tables")
            for table_name in tables["TableNames"]:
                if "profitsentinel" in table_name.lower() or "sentinel" in table_name.lower():
                    log_finding(f"WARNING: Found related table: {table_name}", "WARNING")
                    desc = dynamodb.describe_table(TableName=table_name)
                    item_count = desc["Table"].get("ItemCount", 0)
                    log_finding(f"  Items: {item_count}")
        else:
            log_finding("No DynamoDB tables found")

    except ClientError as e:
        log_finding(f"ERROR accessing DynamoDB: {e}", "ERROR")


def generate_summary():
    """Generate a summary of findings."""
    log_finding("\n" + "=" * 60)
    log_finding("SUMMARY")
    log_finding("=" * 60)
    log_finding(f"Search completed at: {datetime.now().isoformat()}")
    log_finding(f"Report saved to: {REPORT_FILE}")
    log_finding("\nNext steps:")
    log_finding("1. Review findings above for any test data traces")
    log_finding("2. If data found, run destroy_test_data.py (DRY RUN first!)")
    log_finding("3. After cleanup, run verify_cleanup.py to confirm")


def main():
    """Main entry point."""
    # Initialize report file
    with open(REPORT_FILE, "w") as f:
        f.write(f"Profit Sentinel - Test Data Search Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"AWS Region: {AWS_REGION}\n")
        f.write("=" * 60 + "\n\n")

    print("\n" + "=" * 60)
    print("PROFIT SENTINEL - TEST DATA SEARCH")
    print("=" * 60)
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"Region: {AWS_REGION}")
    print(f"Report will be saved to: {REPORT_FILE}")
    print("=" * 60 + "\n")

    # Run checks
    check_s3_bucket()
    check_cloudwatch_logs()
    check_rds_snapshots()
    check_dynamodb_tables()
    generate_summary()

    print(f"\n\nFull report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
