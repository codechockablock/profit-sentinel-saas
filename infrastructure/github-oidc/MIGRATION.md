# GitHub OIDC Migration for AWS Authentication

## Overview
Replaces static `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` GitHub secrets
with short-lived OIDC credentials. No long-lived AWS keys stored in GitHub.

## Steps

### 1. Apply the Terraform

```bash
cd infrastructure/github-oidc
terraform init
terraform plan -var="github_org=YOUR_ORG" -var="github_repo=YOUR_REPO"
# Review the plan carefully
terraform apply -var="github_org=YOUR_ORG" -var="github_repo=YOUR_REPO"
```

### 2. Add GitHub Secret

Add `AWS_ACCOUNT_ID` as a GitHub Actions repository secret (your 12-digit AWS account ID).

### 3. Update Workflow Files

In **each workflow** (`deploy.yml`, `deploy-production.yml`, `deploy-staging.yml`),
replace every occurrence of the static credentials block:

```yaml
# REMOVE this:
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: us-east-1

# REPLACE with:
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::${{ secrets.AWS_ACCOUNT_ID }}:role/github-actions-deploy
    aws-region: us-east-1
```

Add OIDC permissions to each **job** that needs AWS access:

```yaml
jobs:
  deploy-backend:
    permissions:
      id-token: write
      contents: read
    # ... rest of job
```

### 4. Test

- Trigger a workflow_dispatch run on a non-main branch first
- Verify AWS operations succeed with OIDC credentials
- Check CloudTrail for `AssumeRoleWithWebIdentity` events

### 5. Clean Up

Once confirmed working, **remove** the old GitHub secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Then deactivate/delete the corresponding IAM user access keys in AWS.

## Cost Implications
None — OIDC is free. Actually saves money by reducing IAM user management overhead.
