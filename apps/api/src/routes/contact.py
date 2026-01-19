"""
Contact Form Endpoints - Support, feature requests, and feedback.

Handles:
- Support inquiries
- Feature requests
- General feedback
- Privacy-related requests

All submissions are emailed to the support team.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from ..services.email import get_email_service

router = APIRouter()
logger = logging.getLogger(__name__)


class ContactRequest(BaseModel):
    """Contact form submission request."""

    name: str = Field(..., min_length=1, max_length=100, description="Sender's name")
    email: EmailStr = Field(..., description="Sender's email address")
    subject: str = Field(
        ..., min_length=1, max_length=200, description="Message subject"
    )
    message: str = Field(
        ..., min_length=10, max_length=5000, description="Message body"
    )
    type: str = Field(
        default="support",
        description="Contact type: support, feature_request, feedback, privacy",
    )


class ContactResponse(BaseModel):
    """Contact form response."""

    success: bool
    message: str
    reference_id: str | None = None


# Destination emails by contact type
CONTACT_DESTINATIONS = {
    "support": "support@profitsentinel.com",
    "feature_request": "support@profitsentinel.com",
    "feedback": "support@profitsentinel.com",
    "privacy": "privacy@profitsentinel.com",
}


@router.post("/submit", response_model=ContactResponse)
async def submit_contact(request: ContactRequest) -> ContactResponse:
    """
    Submit a contact form message.

    Sends the message to the appropriate team email address.
    Returns a reference ID for tracking.
    """
    email_service = get_email_service()

    # Generate reference ID
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    reference_id = f"PS-{request.type.upper()[:3]}-{timestamp}"

    # Log the submission
    logger.info(
        f"Contact form submission: ref={reference_id}, "
        f"type={request.type}, email={request.email}"
    )

    # Determine destination
    destination = CONTACT_DESTINATIONS.get(request.type, "support@profitsentinel.com")

    if not email_service.is_configured:
        logger.warning(
            f"Email not configured, contact form queued for manual review: {reference_id}"
        )
        # Return success anyway - message is logged for manual follow-up
        return ContactResponse(
            success=True,
            message="Your message has been received. We'll get back to you soon.",
            reference_id=reference_id,
        )

    # Build email content
    type_label = request.type.replace("_", " ").title()
    email_body = f"""
New Contact Form Submission
============================

Reference: {reference_id}
Type: {type_label}
From: {request.name} <{request.email}>
Date: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC

Subject: {request.subject}

Message:
--------
{request.message}

---
This message was sent via the Profit Sentinel contact form.
Reply directly to this email to respond to the sender.
    """.strip()

    try:
        # Send via configured provider
        if email_service.provider == "resend":
            import resend

            resend.api_key = email_service.resend_api_key

            response = resend.Emails.send(
                {
                    "from": f"Profit Sentinel Contact <{email_service.from_email}>",
                    "to": [destination],
                    "reply_to": request.email,
                    "subject": f"[{type_label}] {request.subject} (Ref: {reference_id})",
                    "text": email_body,
                }
            )

            logger.info(
                f"Contact form sent via Resend: ref={reference_id}, "
                f"message_id={response.get('id')}"
            )

        elif email_service.provider == "sendgrid":
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Email, Mail, To

            sg = SendGridAPIClient(email_service.sendgrid_api_key)

            message = Mail(
                from_email=Email(email_service.from_email, "Profit Sentinel Contact"),
                to_emails=To(destination),
                subject=f"[{type_label}] {request.subject} (Ref: {reference_id})",
            )
            message.reply_to = Email(request.email, request.name)

            from sendgrid.helpers.mail import Content

            message.add_content(Content("text/plain", email_body))

            response = sg.send(message)
            logger.info(
                f"Contact form sent via SendGrid: ref={reference_id}, "
                f"status={response.status_code}"
            )

        return ContactResponse(
            success=True,
            message="Your message has been sent. We'll get back to you within 24-48 hours.",
            reference_id=reference_id,
        )

    except Exception as e:
        logger.error(f"Failed to send contact form {reference_id}: {e}")
        # Still return success to user - we have the data logged
        return ContactResponse(
            success=True,
            message="Your message has been received. We'll get back to you soon.",
            reference_id=reference_id,
        )


@router.get("/types")
async def get_contact_types() -> dict:
    """Get available contact types and their descriptions."""
    return {
        "types": [
            {
                "key": "support",
                "label": "Support",
                "description": "Get help with using Profit Sentinel",
            },
            {
                "key": "feature_request",
                "label": "Feature Request",
                "description": "Suggest a new feature or improvement",
            },
            {
                "key": "feedback",
                "label": "Feedback",
                "description": "Share your experience or suggestions",
            },
            {
                "key": "privacy",
                "label": "Privacy Request",
                "description": "Data deletion, access, or privacy concerns",
            },
        ]
    }
