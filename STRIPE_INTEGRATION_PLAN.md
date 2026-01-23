# Stripe Billing Integration Plan

## Overview
Integrate Stripe for subscription billing with 14-day free trial for Profit Sentinel Pro ($99/month).

---

## 1. DATABASE MIGRATION

**File:** `supabase/migrations/005_stripe_billing.sql`

Add Stripe billing columns to `user_profiles` table:
- `stripe_customer_id` (TEXT) - Stripe customer ID
- `stripe_subscription_id` (TEXT) - Active subscription ID
- `subscription_status` (TEXT) - trialing, active, past_due, canceled, expired
- `current_period_end` (TIMESTAMPTZ) - End of current billing period
- `trial_starts_at` (TIMESTAMPTZ) - When trial started
- `trial_expires_at` (TIMESTAMPTZ) - When trial ends (trial_starts_at + 14 days)

Update trigger to set trial on signup:
- `trial_starts_at = NOW()`
- `trial_expires_at = NOW() + INTERVAL '14 days'`
- `subscription_status = 'trialing'`
- `subscription_tier = 'pro'` (full access during trial)

Add helper functions:
- `check_user_access(user_id)` - Returns boolean for access check
- `expire_trials()` - Cron job to expire old trials

---

## 2. BACKEND CONFIGURATION

**File:** `apps/api/src/config.py`

Add settings:
```python
stripe_secret_key: str = ""
stripe_webhook_secret: str = ""
stripe_price_id: str = ""  # Price ID for $99/mo Pro plan
stripe_success_url: str = ""
stripe_cancel_url: str = ""
```

---

## 3. BILLING ROUTES

**File:** `apps/api/src/routes/billing.py`

### POST /billing/create-checkout-session
- Requires authenticated user
- Creates Stripe customer if not exists
- Creates Checkout Session for subscription
- Returns { url: string } for redirect

### POST /billing/webhook
- Validates Stripe signature
- Handles events:
  - `checkout.session.completed` - Activate subscription
  - `customer.subscription.updated` - Update status/period
  - `customer.subscription.deleted` - Mark as canceled
  - `invoice.payment_failed` - Mark as past_due

### POST /billing/portal
- Requires authenticated user
- Creates Stripe Customer Portal session
- Returns { url: string } for redirect

### GET /billing/status
- Returns user's subscription status
- Days remaining in trial
- Current period end

---

## 4. BILLING SERVICE

**File:** `apps/api/src/services/billing.py`

```python
class BillingService:
    def __init__(self):
        self.stripe = stripe
        stripe.api_key = settings.stripe_secret_key

    def get_or_create_customer(user_id, email) -> str
    def create_checkout_session(customer_id, price_id) -> Session
    def create_portal_session(customer_id) -> Session
    def handle_checkout_completed(session) -> None
    def handle_subscription_updated(subscription) -> None
    def handle_subscription_deleted(subscription) -> None
```

---

## 5. ACCESS CONTROL UPDATE

**File:** `apps/api/src/dependencies.py`

Update `get_user_tier()` to use new logic:
```python
async def check_subscription_access(user_id: str) -> dict:
    """
    Returns:
        {
            "has_access": bool,
            "status": str,  # trialing, active, past_due, canceled, expired
            "trial_days_left": int | None,
            "current_period_end": datetime | None
        }
    """
```

New dependency:
```python
async def require_active_subscription(authorization: str) -> str:
    """Requires trialing or active subscription. Returns user_id."""
```

---

## 6. IMPLEMENTATION ORDER

### Phase 1: Database & Config
1. Create migration `005_stripe_billing.sql`
2. Add Stripe settings to `config.py`
3. Add `stripe` to requirements

### Phase 2: Core Billing
4. Create `services/billing.py`
5. Create `routes/billing.py`
6. Register routes in `__init__.py`

### Phase 3: Access Control
7. Update `dependencies.py` with new access check
8. Update analysis routes to use subscription check

### Phase 4: Webhook Security
9. Implement webhook signature verification
10. Add idempotency handling

---

## 7. ENV VARIABLES

```
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_ID=price_...
STRIPE_SUCCESS_URL=https://app.profitsentinel.com/dashboard?upgraded=true
STRIPE_CANCEL_URL=https://app.profitsentinel.com/pricing
```

---

## 8. STRIPE DASHBOARD SETUP

Create in Stripe Dashboard (or via API):

1. **Product:** "Profit Sentinel Pro"
   - Description: "AI-powered profit leak detection for retail"

2. **Price:** $99/month recurring
   - Billing period: Monthly
   - Trial period: 14 days (configured in checkout, not price)

3. **Customer Portal Settings:**
   - Allow cancellation
   - Allow payment method update
   - Show invoice history

4. **Webhook Endpoint:**
   - URL: `https://api.profitsentinel.com/billing/webhook`
   - Events: checkout.session.completed, customer.subscription.*

---

## 9. TRIAL FLOW DIAGRAM

```
User Signup (Supabase Auth)
    │
    ▼
Trigger: handle_new_user()
    │
    ├─► trial_starts_at = NOW()
    ├─► trial_expires_at = NOW() + 14 days
    ├─► subscription_status = 'trialing'
    └─► subscription_tier = 'pro'
    │
    ▼
User Has Full Access (14 days)
    │
    ├─► [Clicks Upgrade] ──► Stripe Checkout ──► subscription_status = 'active'
    │
    └─► [Trial Expires] ──► subscription_status = 'expired'
                              subscription_tier = 'free'
                              │
                              ▼
                        Read-only Access
                        "Upgrade to continue" prompt
```

---

## 10. FILES TO CREATE/MODIFY

| Action | File |
|--------|------|
| CREATE | `supabase/migrations/005_stripe_billing.sql` |
| MODIFY | `apps/api/src/config.py` |
| CREATE | `apps/api/src/services/billing.py` |
| CREATE | `apps/api/src/routes/billing.py` |
| MODIFY | `apps/api/src/routes/__init__.py` |
| MODIFY | `apps/api/src/dependencies.py` |
| MODIFY | `apps/api/requirements.txt` |

---

## 11. SECURITY CONSIDERATIONS

1. **Webhook Verification:** Always verify Stripe signature
2. **Idempotency:** Handle duplicate webhook events gracefully
3. **Rate Limiting:** Apply rate limits to billing endpoints
4. **Audit Logging:** Log all subscription changes
5. **PCI Compliance:** Never handle card data directly (use Checkout)
