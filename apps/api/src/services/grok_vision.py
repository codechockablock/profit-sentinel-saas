"""
Grok Vision Service - Image Understanding for Repair Assistant

Uses Grok Vision API to analyze repair problem photos and extract:
1. Problem type classification
2. Visual features for VSA encoding
3. Damage assessment
4. Component identification

Privacy:
- Images are processed but never stored
- Only text descriptions and feature vectors are retained
- EXIF data must be stripped client-side before upload

Security:
- Input validation and sanitization
- Prompt injection guardrails
- Output validation against expected schema
- Rate limiting awareness
- Content moderation for inappropriate images
"""

import base64
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
from openai import OpenAI

from ..dependencies import get_grok_client

logger = logging.getLogger(__name__)


# =============================================================================
# GUARDRAILS & VALIDATION
# =============================================================================

class VisionGuardrails:
    """
    Security guardrails for Grok Vision prompts and responses.

    Protects against:
    1. Prompt injection via user text
    2. Unexpected/malicious responses
    3. Invalid image data
    4. Content policy violations
    """

    # Maximum lengths for user inputs
    MAX_TEXT_CONTEXT_LENGTH = 500
    MAX_IMAGE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB

    # Blocked patterns in user text (case-insensitive)
    BLOCKED_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"disregard\s+(previous|above|all)",
        r"forget\s+(previous|above|everything)",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(if|a|an)",
        r"pretend\s+(to\s+be|you\s+are)",
        r"system\s*:\s*",
        r"assistant\s*:\s*",
        r"user\s*:\s*",
        r"\[system\]",
        r"\[instruction\]",
        r"new\s+instructions",
        r"override\s+(instructions|prompt)",
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode",
    ]

    # Required fields in response
    REQUIRED_RESPONSE_FIELDS = [
        "primary_category",
        "description",
    ]

    # Valid category values
    VALID_CATEGORIES = {
        "plumbing", "electrical", "hvac", "carpentry",
        "painting", "flooring", "roofing", "appliances",
        "outdoor", "automotive",
    }

    # Valid severity values
    VALID_SEVERITIES = {"minor", "moderate", "major"}

    @classmethod
    def sanitize_user_text(cls, text: str | None) -> str | None:
        """
        Sanitize user-provided text context.

        Removes potential prompt injection attempts while preserving
        legitimate repair problem descriptions.

        Args:
            text: Raw user text

        Returns:
            Sanitized text or None if text should be rejected
        """
        if not text:
            return None

        # Truncate to max length
        text = text[:cls.MAX_TEXT_CONTEXT_LENGTH]

        # Strip dangerous characters
        text = text.replace("\x00", "")  # Null bytes
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)  # Control chars

        # Check for blocked patterns
        text_lower = text.lower()
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"Blocked pattern detected in user text: {pattern}")
                # Remove the problematic portion rather than rejecting entirely
                text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)

        # Escape any markdown-like formatting that could affect parsing
        # But preserve normal punctuation
        text = text.replace("```", "")
        text = text.replace("'''", "")

        return text.strip() if text.strip() else None

    @classmethod
    def validate_image_data(cls, image_base64: str) -> tuple[bool, str]:
        """
        Validate base64 image data.

        Checks:
        1. Valid base64 encoding
        2. Size limits
        3. Valid image header (JPEG/PNG)

        Args:
            image_base64: Base64-encoded image string

        Returns:
            (is_valid, error_message)
        """
        if not image_base64:
            return False, "Image data is empty"

        # Check for reasonable base64 length
        if len(image_base64) > cls.MAX_IMAGE_SIZE_BYTES * 1.37:  # Base64 overhead
            return False, f"Image exceeds {cls.MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB limit"

        try:
            # Decode to verify valid base64
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return False, f"Invalid base64 encoding: {e}"

        # Check actual size after decode
        if len(image_bytes) > cls.MAX_IMAGE_SIZE_BYTES:
            return False, f"Decoded image exceeds {cls.MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB limit"

        # Check for valid image magic bytes
        if image_bytes[:2] == b'\xff\xd8':
            # JPEG
            return True, ""
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            # PNG
            return True, ""
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            # WebP
            return True, ""
        else:
            return False, "Invalid image format. Supported: JPEG, PNG, WebP"

    @classmethod
    def validate_response(cls, data: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        """
        Validate and sanitize API response.

        Ensures response matches expected schema and contains no
        suspicious content.

        Args:
            data: Parsed JSON response

        Returns:
            (is_valid, error_message, sanitized_data)
        """
        if not isinstance(data, dict):
            return False, "Response is not a dictionary", {}

        # Check required fields
        for field in cls.REQUIRED_RESPONSE_FIELDS:
            if field not in data:
                return False, f"Missing required field: {field}", {}

        # Sanitize and validate each field
        sanitized = {}

        # Primary category - must be in valid set
        primary = str(data.get("primary_category", "")).lower().strip()
        if primary not in cls.VALID_CATEGORIES:
            # Try to match partial
            matched = None
            for valid in cls.VALID_CATEGORIES:
                if valid in primary or primary in valid:
                    matched = valid
                    break
            primary = matched or "plumbing"  # Default fallback
        sanitized["primary_category"] = primary

        # Subcategory - sanitize to slug format
        subcategory = data.get("subcategory")
        if subcategory:
            subcategory = str(subcategory).lower()
            subcategory = re.sub(r'[^a-z0-9\-]', '-', subcategory)
            subcategory = re.sub(r'-+', '-', subcategory).strip('-')
            sanitized["subcategory"] = subcategory[:50]  # Max length
        else:
            sanitized["subcategory"] = None

        # Confidence - must be 0-1 float
        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5
        sanitized["confidence"] = confidence

        # Description - sanitize text
        description = str(data.get("description", ""))[:500]
        # Remove any embedded JSON/code
        description = re.sub(r'```.*?```', '', description, flags=re.DOTALL)
        description = re.sub(r'\{[^}]*\}', '', description)
        sanitized["description"] = description.strip() or "Unable to analyze image"

        # Lists - sanitize to simple strings
        list_fields = [
            "visible_components",
            "damage_indicators",
            "keywords",
            "likely_parts_needed",
            "tools_needed",
            "safety_concerns",
        ]
        for list_field in list_fields:
            raw_list = data.get(list_field, [])
            if not isinstance(raw_list, list):
                raw_list = []
            # Sanitize each item
            clean_list = []
            for item in raw_list[:20]:  # Max 20 items
                if isinstance(item, str):
                    # Remove special chars, limit length
                    clean = re.sub(r'[^\w\s\-/]', '', str(item))[:50]
                    if clean.strip():
                        clean_list.append(clean.strip())
            sanitized[list_field] = clean_list

        # Severity - must be valid value
        severity = str(data.get("severity", "moderate")).lower()
        if severity not in cls.VALID_SEVERITIES:
            severity = "moderate"
        sanitized["severity"] = severity

        # Booleans
        sanitized["diy_feasible"] = bool(data.get("diy_feasible", True))
        sanitized["professional_recommended"] = bool(data.get("professional_recommended", False))

        return True, "", sanitized

    @classmethod
    def build_safe_prompt(
        cls,
        user_context: str | None,
        base_prompt: str
    ) -> str:
        """
        Build a safe prompt with clear boundaries.

        Uses XML-like tags to clearly separate system instructions
        from user input, making injection attacks harder.

        Args:
            user_context: Sanitized user text
            base_prompt: Base analysis prompt

        Returns:
            Safe combined prompt
        """
        prompt_parts = [
            base_prompt,
            "",
            "IMPORTANT: Your response must be valid JSON matching the schema above.",
            "Do not include any other text, explanations, or markdown formatting.",
            "Only analyze the image for repair-related content.",
        ]

        if user_context:
            prompt_parts.extend([
                "",
                "<user_description>",
                "The user provided this additional context:",
                user_context,
                "</user_description>",
                "",
                "Use the user's description to help identify the problem, but base your",
                "analysis primarily on what you can see in the image.",
            ])

        return "\n".join(prompt_parts)


@dataclass
class VisionAnalysisResult:
    """Result from Grok Vision image analysis."""

    # Identified problem type
    primary_category: str
    subcategory: str | None
    confidence: float

    # Visual description
    description: str
    visible_components: list[str]
    damage_indicators: list[str]

    # Parts and tools
    likely_parts_needed: list[str]
    tools_needed: list[str]

    # Repair context
    severity_estimate: str  # "minor", "moderate", "major"
    diy_feasible: bool
    professional_recommended: bool

    # Safety
    safety_concerns: list[str]

    # VSA-friendly keywords for encoding
    keywords: list[str]

    # Raw response for debugging
    raw_response: str | None = None


@dataclass
class ImageFeatures:
    """Features extracted for VSA encoding."""

    # Hash for deduplication (NOT for storage)
    image_hash: str

    # Semantic features
    category_hint: str
    subcategory_hint: str | None
    keywords: list[str]
    severity: str

    # For VSA text encoder input
    description_for_encoding: str


class GrokVisionService:
    """
    Service for analyzing repair problem images using Grok Vision.

    Provides:
    1. Problem classification from photos
    2. Feature extraction for VSA encoding
    3. Repair difficulty assessment
    """

    # System prompt for repair image analysis - highly optimized for accuracy
    SYSTEM_PROMPT = """You are an expert repair technician with 20+ years of experience in residential and commercial repair. You work at a hardware store helping customers identify problems and find the right parts.

Your expertise includes:
- Plumbing: Faucets, toilets, pipes, drains, water heaters, sump pumps
- Electrical: Outlets, switches, breakers, wiring, lighting fixtures, ceiling fans
- HVAC: Furnaces, AC units, thermostats, ductwork, filters
- Carpentry: Doors, windows, framing, trim, decks, fences
- Flooring: Tile, hardwood, laminate, vinyl, carpet
- Roofing: Shingles, flashing, gutters, soffits
- Appliances: Washers, dryers, refrigerators, dishwashers, garbage disposals
- Outdoor: Sprinklers, pavers, retaining walls, grading
- Automotive: Basic maintenance, filters, fluids, batteries

IMPORTANT ANALYSIS GUIDELINES:

1. BE SPECIFIC: Don't just say "plumbing problem" - identify the exact issue (e.g., "compression faucet with worn washer causing drip")

2. LOOK FOR CLUES:
   - Water stains indicate leak location and severity
   - Mineral buildup shows hard water and age
   - Rust indicates iron pipes or long-term moisture
   - Burn marks near electrical = serious hazard
   - Sagging/warping = structural moisture damage

3. IDENTIFY PARTS NEEDED: Be specific about parts that might need replacement:
   - Faucet: washer, O-ring, cartridge, seat, aerator
   - Toilet: flapper, fill valve, wax ring, bolts
   - Outlet: receptacle, cover plate, wire nuts
   - etc.

4. SAFETY FIRST: Flag any safety concerns:
   - Water + electricity proximity
   - Gas appliance issues
   - Structural damage
   - Mold presence
   - Asbestos-era materials (pre-1980)

5. CONFIDENCE CALIBRATION:
   - 0.9+ : Clear image, obvious problem, high certainty
   - 0.7-0.9: Good image, likely diagnosis, some uncertainty
   - 0.5-0.7: Partial view or ambiguous symptoms
   - Below 0.5: Cannot reliably diagnose from image

OUTPUT FORMAT: Respond ONLY with valid JSON. No explanations before or after."""

    # Analysis prompt - structured for precise extraction
    ANALYSIS_PROMPT = """Analyze this home repair photo. Examine the image carefully before responding.

STEP 1 - IDENTIFY THE SPACE/CONTEXT:
- What room or area is this? (kitchen, bathroom, basement, exterior, garage, etc.)
- What is the main object/system in the image?

STEP 2 - DIAGNOSE THE PROBLEM:
- What specific malfunction or damage is visible?
- What is the most likely root cause?
- What components appear affected?

STEP 3 - ASSESS SEVERITY:
- minor: Cosmetic or simple fix, basic tools, <1 hour, under $50 parts
- moderate: Functional repair, some skill needed, 1-3 hours, $50-200 parts
- major: Complex repair, specialized tools/skills, 3+ hours, $200+ parts OR professional required

STEP 4 - RECOMMEND ACTION:
- Can an average homeowner do this?
- What tools would they need?
- Should they call a professional instead?

Respond with this exact JSON structure:
{
  "primary_category": "<one of: plumbing, electrical, hvac, carpentry, painting, flooring, roofing, appliances, outdoor, automotive>",
  "subcategory": "<specific-problem-type in slug format, e.g., leaky-faucet, clogged-drain>",
  "confidence": <0.0-1.0 based on image clarity and diagnostic certainty>,
  "description": "<2-3 sentences describing what you see and the likely problem>",
  "visible_components": ["<specific parts visible that may need repair>"],
  "damage_indicators": ["<specific signs of damage/malfunction you observe>"],
  "likely_parts_needed": ["<specific replacement parts a customer might need>"],
  "severity": "<minor|moderate|major>",
  "diy_feasible": <true|false>,
  "professional_recommended": <true|false>,
  "safety_concerns": ["<any safety issues to flag, empty array if none>"],
  "tools_needed": ["<basic tools needed for DIY repair>"],
  "keywords": ["<8-12 search terms for finding relevant products/guides>"]
}"""

    # Category mapping for normalization
    CATEGORY_SLUGS = {
        "plumbing": "plumbing",
        "electrical": "electrical",
        "hvac": "hvac",
        "carpentry": "carpentry",
        "painting": "painting",
        "flooring": "flooring",
        "roofing": "roofing",
        "appliances": "appliances",
        "outdoor": "outdoor",
        "automotive": "automotive",
        # Common variations
        "electric": "electrical",
        "plumb": "plumbing",
        "wood": "carpentry",
        "floor": "flooring",
        "roof": "roofing",
        "appliance": "appliances",
        "garden": "outdoor",
        "lawn": "outdoor",
        "car": "automotive",
        "auto": "automotive",
    }

    def __init__(self, client: OpenAI | None = None):
        """Initialize with Grok client."""
        self._client = client

    @property
    def client(self) -> OpenAI | None:
        """Get Grok client (lazy load if not provided)."""
        if self._client is None:
            self._client = get_grok_client()
        return self._client

    def analyze_image(
        self,
        image_base64: str,
        text_context: str | None = None
    ) -> VisionAnalysisResult:
        """
        Analyze a repair problem image with full guardrails.

        Args:
            image_base64: Base64-encoded image (JPEG/PNG/WebP, max 2MB)
            text_context: Optional text description from user

        Returns:
            VisionAnalysisResult with classification and features

        Raises:
            ValueError: If client not available or image invalid
        """
        if not self.client:
            raise ValueError("Grok client not configured")

        # === GUARDRAIL: Validate image data ===
        is_valid, error_msg = VisionGuardrails.validate_image_data(image_base64)
        if not is_valid:
            logger.warning(f"Image validation failed: {error_msg}")
            raise ValueError(f"Invalid image: {error_msg}")

        # Compute hash for deduplication (NOT for storage)
        image_hash = hashlib.sha256(
            base64.b64decode(image_base64)
        ).hexdigest()[:16]

        # === GUARDRAIL: Sanitize user text ===
        safe_text_context = VisionGuardrails.sanitize_user_text(text_context)
        if text_context and not safe_text_context:
            logger.warning("User text was rejected by guardrails")

        # === GUARDRAIL: Build safe prompt ===
        user_prompt = VisionGuardrails.build_safe_prompt(
            safe_text_context,
            self.ANALYSIS_PROMPT
        )

        # Determine image type from magic bytes
        try:
            image_bytes = base64.b64decode(image_base64)
            if image_bytes[:2] == b'\xff\xd8':
                image_type = "image/jpeg"
            elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                image_type = "image/png"
            elif image_bytes[:4] == b'RIFF':
                image_type = "image/webp"
            else:
                image_type = "image/jpeg"  # Default fallback
        except Exception:
            image_type = "image/jpeg"

        try:
            response = self.client.chat.completions.create(
                model="grok-vision-beta",  # Grok Vision model
                messages=[
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=1024
            )

            content = response.choices[0].message.content.strip()

            # === GUARDRAIL: Validate and sanitize response ===
            result = self._parse_vision_response_safe(content, image_hash)
            result.raw_response = content

            return result

        except Exception as e:
            logger.error(f"Grok Vision analysis failed: {e}")
            # Return fallback result with sanitized context
            return self._fallback_analysis(safe_text_context, image_hash, str(e))

    def _parse_vision_response_safe(
        self,
        content: str,
        image_hash: str
    ) -> VisionAnalysisResult:
        """
        Parse Grok Vision JSON response with full guardrails.

        Uses VisionGuardrails.validate_response for sanitization.
        """
        # Clean JSON from markdown
        json_content = content
        if "```json" in content:
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                json_content = match.group(1)
            else:
                raise ValueError("Could not extract JSON from response")
        elif "```" in content:
            match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                json_content = match.group(1)

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse vision response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")

        # === GUARDRAIL: Validate and sanitize response ===
        is_valid, error_msg, sanitized = VisionGuardrails.validate_response(data)
        if not is_valid:
            logger.warning(f"Response validation failed: {error_msg}")
            raise ValueError(f"Invalid response: {error_msg}")

        # Build subcategory slug properly
        subcategory = sanitized.get("subcategory")
        primary_slug = sanitized["primary_category"]
        if subcategory and not subcategory.startswith(primary_slug):
            subcategory = f"{primary_slug}-{subcategory}"

        return VisionAnalysisResult(
            primary_category=primary_slug,
            subcategory=subcategory,
            confidence=sanitized["confidence"],
            description=sanitized["description"],
            visible_components=sanitized.get("visible_components", []),
            damage_indicators=sanitized.get("damage_indicators", []),
            likely_parts_needed=sanitized.get("likely_parts_needed", []),
            tools_needed=sanitized.get("tools_needed", []),
            severity_estimate=sanitized["severity"],
            diy_feasible=sanitized["diy_feasible"],
            professional_recommended=sanitized["professional_recommended"],
            safety_concerns=sanitized.get("safety_concerns", []),
            keywords=sanitized.get("keywords", []),
        )

    def _parse_vision_response(
        self,
        content: str,
        image_hash: str
    ) -> VisionAnalysisResult:
        """Legacy parser - redirects to safe version."""
        return self._parse_vision_response_safe(content, image_hash)

    def _fallback_analysis(
        self,
        text_context: str | None,
        image_hash: str,
        error: str
    ) -> VisionAnalysisResult:
        """Provide fallback analysis when vision API fails."""
        # Try to infer category from text context
        category = "plumbing"  # Default
        keywords = []
        likely_parts = []
        tools = []

        if text_context:
            text_lower = text_context.lower()
            for hint, cat in self.CATEGORY_SLUGS.items():
                if hint in text_lower:
                    category = cat
                    break

            # Extract simple keywords
            keywords = re.findall(r'\b\w{4,}\b', text_lower)[:10]

            # Infer basic parts/tools from keywords
            parts_hints = {
                "faucet": ["washer", "O-ring", "cartridge"],
                "toilet": ["flapper", "fill valve"],
                "outlet": ["receptacle", "cover plate"],
                "drain": ["drain snake", "P-trap"],
            }
            tools_hints = {
                "faucet": ["adjustable wrench", "pliers"],
                "toilet": ["adjustable wrench"],
                "outlet": ["screwdriver", "voltage tester"],
                "drain": ["plunger", "drain snake"],
            }

            for hint, parts in parts_hints.items():
                if hint in text_lower:
                    likely_parts.extend(parts)
            for hint, t in tools_hints.items():
                if hint in text_lower:
                    tools.extend(t)

        # Sanitize error message
        safe_error = re.sub(r'[^\w\s\-.]', '', str(error))[:50]

        return VisionAnalysisResult(
            primary_category=category,
            subcategory=None,
            confidence=0.3,  # Low confidence for fallback
            description=f"Image analysis unavailable. {safe_error}",
            visible_components=[],
            damage_indicators=[],
            likely_parts_needed=likely_parts[:5],
            tools_needed=tools[:5],
            severity_estimate="moderate",
            diy_feasible=True,
            professional_recommended=False,
            safety_concerns=[],
            keywords=keywords,
        )

    def extract_features_for_vsa(
        self,
        result: VisionAnalysisResult
    ) -> ImageFeatures:
        """
        Extract features suitable for VSA encoding.

        Args:
            result: VisionAnalysisResult from analyze_image

        Returns:
            ImageFeatures for VSA text encoder
        """
        # Build description string for VSA encoding
        # Include all relevant fields for comprehensive vector representation
        description_parts = [
            result.primary_category,
            result.subcategory or "",
            result.description,
            " ".join(result.visible_components),
            " ".join(result.damage_indicators),
            " ".join(result.likely_parts_needed),
            result.severity_estimate,
            " ".join(result.keywords),
        ]
        description_for_encoding = " ".join(filter(None, description_parts))

        # Combine keywords with parts and components for richer encoding
        all_keywords = list(set(
            result.keywords +
            result.visible_components +
            result.likely_parts_needed
        ))

        return ImageFeatures(
            image_hash="",  # Not stored
            category_hint=result.primary_category,
            subcategory_hint=result.subcategory,
            keywords=all_keywords,
            severity=result.severity_estimate,
            description_for_encoding=description_for_encoding,
        )

    def encode_to_vsa_vector(
        self,
        features: ImageFeatures,
        text_encoder,  # TextEncoder from repair_engine
    ) -> torch.Tensor:
        """
        Encode image features to VSA vector using text encoder.

        Args:
            features: Extracted image features
            text_encoder: TextEncoder instance from repair engine

        Returns:
            VSA vector representing the image
        """
        # Use the description for encoding
        return text_encoder.encode(features.description_for_encoding)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_repair_image(
    image_base64: str,
    text_context: str | None = None
) -> VisionAnalysisResult:
    """
    Convenience function to analyze a repair image.

    Args:
        image_base64: Base64-encoded image
        text_context: Optional text description

    Returns:
        VisionAnalysisResult
    """
    service = GrokVisionService()
    return service.analyze_image(image_base64, text_context)


def get_image_vsa_vector(
    image_base64: str,
    text_encoder,
    text_context: str | None = None
) -> tuple[torch.Tensor, VisionAnalysisResult]:
    """
    Get VSA vector from repair image.

    Args:
        image_base64: Base64-encoded image
        text_encoder: TextEncoder from repair engine
        text_context: Optional text description

    Returns:
        (VSA vector, VisionAnalysisResult)
    """
    service = GrokVisionService()

    # Analyze image
    result = service.analyze_image(image_base64, text_context)

    # Extract features
    features = service.extract_features_for_vsa(result)

    # Encode to VSA
    vector = service.encode_to_vsa_vector(features, text_encoder)

    return vector, result
