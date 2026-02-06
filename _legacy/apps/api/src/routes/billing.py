"""
Billing endpoints - Stripe subscription management.

Endpoints:
- POST /billing/create-checkout-session - Create Stripe Checkout for upgrade
- POST /billing/webhook - Handle Stripe webhook events
- POST /billing/portal - Create Customer Portal session
- GET /billing/status - Get current subscription status
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from stripe.error import SignatureVerificationError

from ..config import get_settings
from ..dependencies import get_supabase_client, require_user
from ..services.billing import (
    get_billing_service,
    parse_checkout_session,
    parse_subscription_data,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class CheckoutSessionResponse(BaseModel):
    """Response for checkout session creation."""

    url: str
    session_id: str


class PortalSessionResponse(BaseModel):
    """Response for portal session creation."""

    url: str


class SubscriptionStatusResponse(BaseModel):
    """Response for subscription status."""

    has_access: bool
    status: str  # trialing, active, past_due, canceled, expired
    tier: str  # free, pro, enterprise
    trial_days_left: int | None = None
    current_period_end: datetime | None = None
    is_trial: bool = False
    stripe_customer_id: str | None = None


class WebhookResponse(BaseModel):
    """Response for webhook processing."""

    received: bool
    event_type: str | None = None


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/create-checkout-session", response_model=CheckoutSessionResponse)
@limiter.limit("10/minute")
async def create_checkout_session(
    request: Request,
    user_id: str = Depends(require_user),
    success_url: str | None = None,
    cancel_url: str | None = None,
) -> CheckoutSessionResponse:
    """
    Create a Stripe Checkout session for Pro subscription upgrade.

    Requires authenticated user. Creates or retrieves Stripe customer,
    then creates a checkout session for the $99/month Pro subscription.

    Args:
        success_url: Optional URL to redirect after successful payment
        cancel_url: Optional URL to redirect if payment canceled

    Returns:
        CheckoutSessionResponse with URL to redirect user to Stripe Checkout
    """
    billing = get_billing_service()

    if not billing.is_available:
        raise HTTPException(
            status_code=503,
            detail="Billing service not available. Please try again later.",
        )

    # Get user data from Supabase
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Fetch user profile
        result = (
            supabase.table("user_profiles")
            .select("email, full_name, stripe_customer_id")
            .eq("id", user_id)
            .single()
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="User profile not found")

        user_data = result.data
        email = user_data.get("email")
        name = user_data.get("full_name")
        existing_customer_id = user_data.get("stripe_customer_id")

        if not email:
            raise HTTPException(status_code=400, detail="User email not found")

        # Get or create Stripe customer
        customer_id = billing.get_or_create_customer(
            user_id=user_id,
            email=email,
            name=name,
            existing_customer_id=existing_customer_id,
        )

        # Store customer ID if new
        if customer_id != existing_customer_id:
            supabase.table("user_profiles").update(
                {"stripe_customer_id": customer_id, "updated_at": "now()"}
            ).eq("id", user_id).execute()

        # Create checkout session
        session = billing.create_checkout_session(
            customer_id=customer_id,
            user_id=user_id,
            success_url=success_url,
            cancel_url=cancel_url,
        )

        # Log without user_id (privacy - use session ID instead)
        logger.info(f"Created checkout session: {session.id}")

        return CheckoutSessionResponse(
            url=session.url,
            session_id=session.id,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Log error without stack trace in production (M3 fix)
        logger.error(f"Failed to create checkout session: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create checkout session. Please try again.",
        )


@router.post("/portal", response_model=PortalSessionResponse)
@limiter.limit("10/minute")
async def create_portal_session(
    request: Request,
    user_id: str = Depends(require_user),
    return_url: str | None = None,
) -> PortalSessionResponse:
    """
    Create a Stripe Customer Portal session.

    Allows users to:
    - Update payment method
    - View and download invoices
    - Cancel subscription

    Args:
        return_url: Optional URL to return to after portal

    Returns:
        PortalSessionResponse with URL to redirect user to Customer Portal
    """
    billing = get_billing_service()

    if not billing.is_available:
        raise HTTPException(
            status_code=503,
            detail="Billing service not available. Please try again later.",
        )

    # Get user's Stripe customer ID
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        result = (
            supabase.table("user_profiles")
            .select("stripe_customer_id")
            .eq("id", user_id)
            .single()
            .execute()
        )

        if not result.data or not result.data.get("stripe_customer_id"):
            raise HTTPException(
                status_code=400,
                detail="No billing account found. Please upgrade first.",
            )

        customer_id = result.data["stripe_customer_id"]

        # Create portal session
        session = billing.create_portal_session(
            customer_id=customer_id,
            return_url=return_url,
        )

        return PortalSessionResponse(url=session.url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create portal session: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create portal session. Please try again.",
        )


@router.get("/status", response_model=SubscriptionStatusResponse)
@limiter.limit("30/minute")
async def get_subscription_status(
    request: Request,
    user_id: str = Depends(require_user),
) -> SubscriptionStatusResponse:
    """
    Get current subscription status for authenticated user.

    Returns subscription status, tier, trial information,
    and whether the user has active access.
    """
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Use the database function to check access
        result = supabase.rpc("check_user_access", {"p_user_id": user_id}).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")

        access_data = result.data[0]

        # Also get current tier and customer ID
        profile_result = (
            supabase.table("user_profiles")
            .select("subscription_tier, stripe_customer_id")
            .eq("id", user_id)
            .single()
            .execute()
        )

        tier = "free"
        customer_id = None
        if profile_result.data:
            tier = profile_result.data.get("subscription_tier", "free")
            customer_id = profile_result.data.get("stripe_customer_id")

        status = access_data.get("subscription_status", "none")
        is_trial = status == "trialing"

        return SubscriptionStatusResponse(
            has_access=access_data.get("has_access", False),
            status=status,
            tier=tier,
            trial_days_left=access_data.get("trial_days_left"),
            current_period_end=access_data.get("current_period_end"),
            is_trial=is_trial,
            stripe_customer_id=customer_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subscription status: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get subscription status.",
        )


@router.post("/webhook", response_model=WebhookResponse)
async def handle_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
) -> WebhookResponse:
    """
    Handle Stripe webhook events.

    Processes:
    - checkout.session.completed - Activate subscription
    - customer.subscription.updated - Update status/period
    - customer.subscription.deleted - Mark as canceled
    - invoice.payment_failed - Mark as past_due

    Note: No rate limiting on webhooks (Stripe manages delivery)
    """
    billing = get_billing_service()
    settings = get_settings()

    if not billing.is_available:
        logger.warning("Webhook received but Stripe not configured")
        return WebhookResponse(received=True, event_type=None)

    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")

    if not settings.stripe_webhook_secret:
        logger.error("Stripe webhook secret not configured")
        raise HTTPException(status_code=500, detail="Webhook not configured")

    # Get raw body for signature verification
    body = await request.body()

    try:
        event = billing.verify_webhook_signature(body, stripe_signature)
    except SignatureVerificationError as e:
        logger.warning(f"Invalid webhook signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event.type
    logger.info(f"Received Stripe webhook: {event_type}")

    supabase = get_supabase_client()
    if not supabase:
        logger.error("Database unavailable for webhook processing")
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        # Handle different event types
        if event_type == "checkout.session.completed":
            await _handle_checkout_completed(event.data.object, supabase, billing)

        elif event_type == "customer.subscription.updated":
            await _handle_subscription_updated(event.data.object, supabase)

        elif event_type == "customer.subscription.deleted":
            await _handle_subscription_deleted(event.data.object, supabase)

        elif event_type == "invoice.payment_failed":
            await _handle_payment_failed(event.data.object, supabase)

        else:
            logger.debug(f"Unhandled webhook event type: {event_type}")

        return WebhookResponse(received=True, event_type=event_type)

    except Exception as e:
        logger.error(f"Error processing webhook {event_type}: {type(e).__name__}")
        # Return 200 to prevent Stripe retries for processing errors
        # The webhook was received, we just failed to process it
        return WebhookResponse(received=True, event_type=event_type)


# =============================================================================
# WEBHOOK EVENT HANDLERS
# =============================================================================


async def _handle_checkout_completed(session, supabase, billing) -> None:
    """
    Handle checkout.session.completed event.

    Updates user profile with:
    - stripe_customer_id
    - stripe_subscription_id
    - subscription_status = 'active'
    - subscription_tier = 'pro'
    """
    session_data = parse_checkout_session(session)
    user_id = session_data.get("user_id")
    customer_id = session_data.get("stripe_customer_id")
    subscription_id = session_data.get("stripe_subscription_id")

    if not user_id:
        logger.warning(f"Checkout completed without user_id: {session.id}")
        return

    # Log without user_id (privacy - use session ID instead)
    logger.info(f"Processing checkout completion for session {session.id}")

    # Get subscription details
    if subscription_id:
        subscription = billing.get_subscription(subscription_id)
        sub_data = parse_subscription_data(subscription)
    else:
        sub_data = {
            "subscription_status": "active",
            "current_period_end": None,
        }

    # Update user profile
    update_data = {
        "stripe_customer_id": customer_id,
        "stripe_subscription_id": subscription_id,
        "subscription_status": sub_data.get("subscription_status", "active"),
        "subscription_tier": "pro",
        "current_period_end": (
            sub_data["current_period_end"].isoformat()
            if sub_data.get("current_period_end")
            else None
        ),
        "updated_at": datetime.now(datetime.UTC).isoformat(),
    }

    supabase.table("user_profiles").update(update_data).eq("id", user_id).execute()

    logger.info(f"Activated subscription for session {session.id}")


async def _handle_subscription_updated(subscription, supabase) -> None:
    """
    Handle customer.subscription.updated event.

    Updates subscription status and period end.
    """
    sub_data = parse_subscription_data(subscription)
    customer_id = subscription.customer

    logger.info(
        f"Subscription updated: {subscription.id} -> {sub_data['subscription_status']}"
    )

    # Determine tier based on status
    tier = (
        "pro"
        if sub_data["subscription_status"] in ("active", "trialing", "past_due")
        else "free"
    )

    update_data = {
        "stripe_subscription_id": subscription.id,
        "subscription_status": sub_data["subscription_status"],
        "subscription_tier": tier,
        "current_period_end": (
            sub_data["current_period_end"].isoformat()
            if sub_data.get("current_period_end")
            else None
        ),
        "updated_at": datetime.now(datetime.UTC).isoformat(),
    }

    supabase.table("user_profiles").update(update_data).eq(
        "stripe_customer_id", customer_id
    ).execute()


async def _handle_subscription_deleted(subscription, supabase) -> None:
    """
    Handle customer.subscription.deleted event.

    Marks subscription as canceled and downgrades to free tier.
    """
    customer_id = subscription.customer

    logger.info(f"Subscription deleted: {subscription.id}")

    update_data = {
        "subscription_status": "canceled",
        "subscription_tier": "free",
        "stripe_subscription_id": None,  # Clear the subscription ID
        "updated_at": datetime.now(datetime.UTC).isoformat(),
    }

    supabase.table("user_profiles").update(update_data).eq(
        "stripe_customer_id", customer_id
    ).execute()


async def _handle_payment_failed(invoice, supabase) -> None:
    """
    Handle invoice.payment_failed event.

    Marks subscription as past_due (still allows access for grace period).
    """
    customer_id = invoice.customer
    subscription_id = invoice.subscription

    if not subscription_id:
        return

    logger.warning(f"Payment failed for subscription {subscription_id}")

    update_data = {
        "subscription_status": "past_due",
        "updated_at": datetime.now(datetime.UTC).isoformat(),
    }

    supabase.table("user_profiles").update(update_data).eq(
        "stripe_customer_id", customer_id
    ).execute()
