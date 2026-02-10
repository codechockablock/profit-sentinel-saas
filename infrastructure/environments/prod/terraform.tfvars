# Production environment variables
# NOTE: AWS credentials come from env vars or ~/.aws/credentials, NOT here.

acm_certificate_arn = "arn:aws:acm:us-east-1:133608785306:certificate/fcc87c1f-030c-4633-9d74-0deb936386b9"

anthropic_api_key_secret_arn     = "arn:aws:secretsmanager:us-east-1:133608785306:secret:profitsentinel/api-key-JCMGGj"
supabase_url                     = "https://kbjiejqotrjsdeuxhtcx.supabase.co"
supabase_service_key_secret_arn  = "arn:aws:secretsmanager:us-east-1:133608785306:secret:profitsentinel/supabase-service-key-8l3rV4"
resend_api_key_secret_arn        = "arn:aws:secretsmanager:us-east-1:133608785306:secret:profitsentinel-dev/resend-api-key-Z5SULk"
