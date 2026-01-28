#!/usr/bin/env python3
"""
Profit Sentinel - Test Data Destruction Script

DESTRUCTIVE SCRIPT - This permanently deletes data from production!
Always run with --dry-run first to review what will be deleted.

Usage:
    python scripts/destroy_test_data.py --dry-run    # Preview only
    python scripts/destroy_test_data.py              # Actual deletion (requires confirmation)

Prerequisites:
    pip install boto3
    AWS credentials configured (~/.aws/credentials or environment variables)

Output:
    Logs all actions to destruction_log_<timestamp>.txt
"""

import argparse
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
# CONFIGURATION - Update these with actual values from your AWS environment
# =============================================================================

# S3 bucket name
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")

# CloudWatch log groups to clear
LOG_GROUPS = [
    "/ecs/profitsentinel-dev",
]

# AWS region
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Log file
LOG_FILE = f"destruction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# =============================================================================
# LOGGING
# =============================================================================


def log_action(message: str, level: str = "INFO"):
    """Log an action to console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# =============================================================================
# DESTRUCTION FUNCTIONS
# =============================================================================


def destroy_s3_objects(dry_run: bool = True):
    """Delete all S3 objects including all versions and delete markers."""
    log_action("=" * 60)
    log_action("S3 BUCKET CLEANUP")
    log_action("=" * 60)

    if dry_run:
        log_action("DRY RUN MODE - No actual deletions", "DRY_RUN")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        # Get bucket versioning status
        versioning = s3.get_bucket_versioning(Bucket=S3_BUCKET)
        is_versioned = versioning.get("Status") == "Enabled"
        log_action(f"Bucket versioning: {'ENABLED' if is_versioned else 'DISABLED'}")

        objects_deleted = 0
        versions_deleted = 0
        markers_deleted = 0

        if is_versioned:
            # Delete all versions and delete markers
            log_action("\n--- Deleting All Versions ---")
            paginator = s3.get_paginator("list_object_versions")

            for page in paginator.paginate(Bucket=S3_BUCKET):
                # Delete versions
                for version in page.get("Versions", []):
                    key = version["Key"]
                    version_id = version["VersionId"]
                    is_latest = "CURRENT" if version.get("IsLatest") else "NON-CURRENT"

                    log_action(f"  [{is_latest}] {key} (Version: {version_id[:8]}...)")

                    if not dry_run:
                        s3.delete_object(
                            Bucket=S3_BUCKET, Key=key, VersionId=version_id
                        )
                        log_action(f"    DELETED", "DELETE")
                    else:
                        log_action(f"    Would delete", "DRY_RUN")

                    versions_deleted += 1

                # Delete markers
                for marker in page.get("DeleteMarkers", []):
                    key = marker["Key"]
                    version_id = marker["VersionId"]

                    log_action(f"  [DELETE_MARKER] {key} (Version: {version_id[:8]}...)")

                    if not dry_run:
                        s3.delete_object(
                            Bucket=S3_BUCKET, Key=key, VersionId=version_id
                        )
                        log_action(f"    DELETED", "DELETE")
                    else:
                        log_action(f"    Would delete", "DRY_RUN")

                    markers_deleted += 1

        else:
            # Non-versioned bucket - just delete objects
            log_action("\n--- Deleting Objects ---")
            paginator = s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=S3_BUCKET):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    log_action(f"  {key}")

                    if not dry_run:
                        s3.delete_object(Bucket=S3_BUCKET, Key=key)
                        log_action(f"    DELETED", "DELETE")
                    else:
                        log_action(f"    Would delete", "DRY_RUN")

                    objects_deleted += 1

        log_action("\n--- S3 Summary ---")
        log_action(f"Objects/Versions processed: {versions_deleted + objects_deleted}")
        log_action(f"Delete markers processed: {markers_deleted}")

        if dry_run:
            log_action("DRY RUN - No actual deletions performed", "DRY_RUN")

    except ClientError as e:
        log_action(f"ERROR: {e}", "ERROR")
    except NoCredentialsError:
        log_action("ERROR: AWS credentials not configured", "ERROR")


def destroy_cloudwatch_logs(dry_run: bool = True):
    """Delete CloudWatch log streams (keeps log group)."""
    log_action("\n" + "=" * 60)
    log_action("CLOUDWATCH LOGS CLEANUP")
    log_action("=" * 60)

    if dry_run:
        log_action("DRY RUN MODE - No actual deletions", "DRY_RUN")

    logs = boto3.client("logs", region_name=AWS_REGION)

    for group_name in LOG_GROUPS:
        log_action(f"\n--- Log Group: {group_name} ---")

        try:
            # List all log streams
            paginator = logs.get_paginator("describe_log_streams")
            streams_deleted = 0

            for page in paginator.paginate(logGroupName=group_name):
                for stream in page.get("logStreams", []):
                    stream_name = stream["logStreamName"]
                    last_event = stream.get("lastEventTimestamp")

                    if last_event:
                        last_event_time = datetime.fromtimestamp(last_event / 1000)
                        log_action(f"  {stream_name} (Last event: {last_event_time})")
                    else:
                        log_action(f"  {stream_name} (No events)")

                    if not dry_run:
                        logs.delete_log_stream(
                            logGroupName=group_name, logStreamName=stream_name
                        )
                        log_action(f"    DELETED", "DELETE")
                    else:
                        log_action(f"    Would delete", "DRY_RUN")

                    streams_deleted += 1

            log_action(f"Log streams processed: {streams_deleted}")

            # Set retention policy if not dry run
            if not dry_run:
                log_action("Setting retention policy to 7 days...")
                logs.put_retention_policy(
                    logGroupName=group_name, retentionInDays=7
                )
                log_action("Retention policy set", "UPDATE")

        except ClientError as e:
            if "ResourceNotFoundException" in str(e):
                log_action(f"Log group not found: {group_name}")
            else:
                log_action(f"ERROR: {e}", "ERROR")


def list_rds_snapshots():
    """List RDS snapshots for manual review (no auto-deletion)."""
    log_action("\n" + "=" * 60)
    log_action("RDS SNAPSHOTS (MANUAL REVIEW REQUIRED)")
    log_action("=" * 60)

    log_action(
        "WARNING: RDS snapshots require manual review before deletion.", "WARNING"
    )
    log_action("This script will NOT auto-delete snapshots.\n")

    rds = boto3.client("rds", region_name=AWS_REGION)

    try:
        # List all snapshots
        response = rds.describe_db_cluster_snapshots()

        manual_snapshots = []
        auto_snapshots = []

        for snap in response.get("DBClusterSnapshots", []):
            snap_info = {
                "id": snap["DBClusterSnapshotIdentifier"],
                "created": snap["SnapshotCreateTime"],
                "type": snap["SnapshotType"],
                "cluster": snap.get("DBClusterIdentifier", "unknown"),
            }

            if snap["SnapshotType"] == "manual":
                manual_snapshots.append(snap_info)
            else:
                auto_snapshots.append(snap_info)

        if manual_snapshots:
            log_action("--- Manual Snapshots (Review Required) ---")
            for snap in manual_snapshots:
                log_action(f"  {snap['id']}")
                log_action(f"    Cluster: {snap['cluster']}")
                log_action(f"    Created: {snap['created']}")
                log_action(f"    To delete: aws rds delete-db-cluster-snapshot \\")
                log_action(f"              --db-cluster-snapshot-identifier {snap['id']}")
        else:
            log_action("No manual snapshots found")

        if auto_snapshots:
            log_action(f"\n--- Automated Snapshots ({len(auto_snapshots)} found) ---")
            log_action("Automated snapshots are managed by RDS retention policy.")
            log_action("They will be deleted automatically based on backup retention.")

    except ClientError as e:
        log_action(f"ERROR: {e}", "ERROR")


def generate_summary(dry_run: bool):
    """Generate a summary of actions."""
    log_action("\n" + "=" * 60)
    log_action("DESTRUCTION SUMMARY")
    log_action("=" * 60)

    if dry_run:
        log_action("MODE: DRY RUN (no actual deletions)", "DRY_RUN")
        log_action("\nTo perform actual deletion, run:")
        log_action("  python scripts/destroy_test_data.py")
        log_action("\nYou will be prompted for confirmation.")
    else:
        log_action("MODE: ACTUAL DELETION", "DELETE")
        log_action("All specified data has been permanently deleted.")

    log_action(f"\nLog saved to: {LOG_FILE}")
    log_action("\nNext steps:")
    log_action("1. Run verify_cleanup.py to confirm all data is removed")
    log_action("2. Review any RDS snapshots listed above manually")
    log_action("3. Check Supabase dashboard for any data to purge")
    log_action("4. Check email provider (Resend/SendGrid) for sent reports")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Destroy test data from production AWS resources"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write(f"Profit Sentinel - Data Destruction Log\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Mode: {'DRY RUN' if args.dry_run else 'ACTUAL DELETION'}\n")
        f.write("=" * 60 + "\n\n")

    print("\n" + "=" * 60)
    print("PROFIT SENTINEL - DATA DESTRUCTION")
    print("=" * 60)
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"Region: {AWS_REGION}")
    print(f"Log file: {LOG_FILE}")

    if args.dry_run:
        print("\nMODE: DRY RUN (preview only)")
        print("=" * 60 + "\n")
    else:
        print("\n" + "!" * 60)
        print("WARNING: ACTUAL DELETION MODE")
        print("This will PERMANENTLY DELETE data from production!")
        print("!" * 60)

        confirmation = input("\nType 'DESTROY' to confirm deletion: ")
        if confirmation != "DESTROY":
            print("\nAborted. No data was deleted.")
            sys.exit(0)

        print("\nProceeding with deletion...")
        print("=" * 60 + "\n")

    # Run destruction
    destroy_s3_objects(dry_run=args.dry_run)
    destroy_cloudwatch_logs(dry_run=args.dry_run)
    list_rds_snapshots()  # Always list-only, never auto-delete
    generate_summary(dry_run=args.dry_run)

    print(f"\n\nLog saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
