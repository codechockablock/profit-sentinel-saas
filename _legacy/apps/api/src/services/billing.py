"""
Billing Service - Stripe subscription management.

Handles:
- Stripe customer creation
- Checkout session creation for subscription
- Customer portal session creation
- Webhook event processing

Trial Flow:
1. User signs up -> 14-day trial starts automatically (via DB trigger)
2. User clicks upgrade -> Checkout session created
3. User completes payment -> Subscription activated
4. Trial users who don't convert -> Access expires
"""

import logging
from datetime import datetime
from typing import Any

import stripe
from stripe import Webhook

from ..config import get_settings

logger = logging.getLogger(__name__)


class BillingService:
    """Service for Stripe billing operations."""

    def __init__(self):
        """Initialize Stripe with API key from settings."""
        settings = get_settings()
        self.settings = settings

        if settings.stripe_secret_key:
            stripe.api_key = settings.stripe_secret_key
            self._available = True
            logger.info("Stripe billing service initialized")
        else:
            self._available = False
            logger.warning("Stripe not configured - billing features disabled")

    @property
    def is_available(self) -> bool:
        """Check if Stripe is properly configured."""
        return self._available and self.settings.has_stripe

    def get_or_create_customer(
        self,
        user_id: str,
        email: str,
        name: str | None = None,
        existing_customer_id: str | None = None,
    ) -> str:
        """
        Get existing Stripe customer or create a new one.

        Args:
            user_id: Internal user ID (stored in metadata)
            email: Customer email
            name: Customer name (optional)
            existing_customer_id: Existing Stripe customer ID if known

        Returns:
            Stripe customer ID (cus_xxx)
        """
        if not self.is_available:
            raise RuntimeError("Stripe billing not configured")

        # If we already have a customer ID, verify it exists
        if existing_customer_id:
            try:
                customer = stripe.Customer.retrieve(existing_customer_id)
                if not customer.deleted:
                    return customer.id
            except stripe.error.InvalidRequestError:
                logger.warning(f"Stored customer ID {existing_customer_id} not found")

        # Search for existing customer by email
        customers = stripe.Customer.search(query=f'email:"{email}"')
        if customers.data:
            # Return first matching customer
            return customers.data[0].id

        # Create new customer
        customer = stripe.Customer.create(
            email=email,
            name=name,
            metadata={
                "user_id": user_id,
                "source": "profit_sentinel",
            },
        )

        logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
        return customer.id

    def create_checkout_session(
        self,
        customer_id: str,
        user_id: str,
        success_url: str | None = None,
        cancel_url: str | None = None,
    ) -> stripe.checkout.Session:
        """
        Create a Stripe Checkout session for Pro subscription.

        Args:
            customer_id: Stripe customer ID
            user_id: Internal user ID (for metadata)
            success_url: URL to redirect after successful payment
            cancel_url: URL to redirect if payment canceled

        Returns:
            Stripe Checkout Session with URL for redirect
        """
        if not self.is_available:
            raise RuntimeError("Stripe billing not configured")

        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[
                {
                    "price": self.settings.stripe_price_id,
                    "quantity": 1,
                },
            ],
            success_url=success_url or self.settings.stripe_success_url,
            cancel_url=cancel_url or self.settings.stripe_cancel_url,
            subscription_data={
                "metadata": {
                    "user_id": user_id,
                },
                # No trial_period_days here - users already have 14-day trial
                # If they're upgrading, they get immediate access
            },
            metadata={
                "user_id": user_id,
            },
            # Allow promotion codes
            allow_promotion_codes=True,
            # Collect billing address for tax
            billing_address_collection="auto",
        )

        logger.info(f"Created checkout session {session.id} for customer {customer_id}")
        return session

    def create_portal_session(
        self,
        customer_id: str,
        return_url: str | None = None,
    ) -> stripe.billing_portal.Session:
        """
        Create a Stripe Customer Portal session.

        Allows customers to:
        - Update payment method
        - View invoices
        - Cancel subscription

        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after portal (default: dashboard)

        Returns:
            Portal session with URL for redirect
        """
        if not self.is_available:
            raise RuntimeError("Stripe billing not configured")

        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url
            or self.settings.stripe_success_url.replace("?upgraded=true", ""),
        )

        logger.info(f"Created portal session for customer {customer_id}")
        return session

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> stripe.Event:
        """
        Verify Stripe webhook signature and parse event.

        Args:
            payload: Raw request body bytes
            signature: Stripe-Signature header value

        Returns:
            Verified Stripe Event object

        Raises:
            SignatureVerificationError: If signature is invalid
        """
        if not self.settings.stripe_webhook_secret:
            raise RuntimeError("Stripe webhook secret not configured")

        event = Webhook.construct_event(
            payload,
            signature,
            self.settings.stripe_webhook_secret,
        )

        return event

    def get_subscription(self, subscription_id: str) -> stripe.Subscription:
        """
        Retrieve a subscription from Stripe.

        Args:
            subscription_id: Stripe subscription ID (sub_xxx)

        Returns:
            Stripe Subscription object
        """
        if not self.is_available:
            raise RuntimeError("Stripe billing not configured")

        return stripe.Subscription.retrieve(subscription_id)

    def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> stripe.Subscription:
        """
        Cancel a subscription.

        Args:
            subscription_id: Stripe subscription ID
            at_period_end: If True, cancel at end of billing period

        Returns:
            Updated subscription
        """
        if not self.is_available:
            raise RuntimeError("Stripe billing not configured")

        if at_period_end:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True,
            )
        else:
            subscription = stripe.Subscription.cancel(subscription_id)

        logger.info(f"Canceled subscription {subscription_id}")
        return subscription


def parse_subscription_data(subscription: stripe.Subscription) -> dict[str, Any]:
    """
    Extract relevant data from a Stripe Subscription object.

    Args:
        subscription: Stripe Subscription object

    Returns:
        Dict with normalized subscription data
    """
    # Map Stripe status to our status
    status_map = {
        "active": "active",
        "trialing": "trialing",
        "past_due": "past_due",
        "canceled": "canceled",
        "unpaid": "past_due",
        "incomplete": "incomplete",
        "incomplete_expired": "expired",
    }

    return {
        "stripe_subscription_id": subscription.id,
        "subscription_status": status_map.get(subscription.status, subscription.status),
        "current_period_end": datetime.fromtimestamp(
            subscription.current_period_end, tz=datetime.UTC
        ),
        "cancel_at_period_end": subscription.cancel_at_period_end,
        "user_id": subscription.metadata.get("user_id"),
    }


def parse_checkout_session(session: stripe.checkout.Session) -> dict[str, Any]:
    """
    Extract relevant data from a completed Checkout Session.

    Args:
        session: Stripe Checkout Session object

    Returns:
        Dict with session data
    """
    return {
        "stripe_customer_id": session.customer,
        "stripe_subscription_id": session.subscription,
        "user_id": session.metadata.get("user_id"),
        "customer_email": (
            session.customer_details.email if session.customer_details else None
        ),
    }


# Singleton instance
_billing_service: BillingService | None = None


def get_billing_service() -> BillingService:
    """Get the singleton billing service instance."""
    global _billing_service
    if _billing_service is None:
        _billing_service = BillingService()
    return _billing_service
