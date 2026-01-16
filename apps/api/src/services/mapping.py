"""
Column Mapping Service.

Provides intelligent column mapping using AI with heuristic fallback.
"""

import json
import logging
from typing import Dict, List

import pandas as pd

from ..config import STANDARD_FIELDS, get_settings
from ..dependencies import get_grok_client

logger = logging.getLogger(__name__)


class MappingService:
    """Service for intelligent column mapping."""

    def suggest_mapping(self, df: pd.DataFrame, filename: str) -> Dict:
        """
        Suggest column mappings for uploaded data.

        Uses Grok AI when available, falls back to heuristics.

        Args:
            df: DataFrame with sample data
            filename: Original filename

        Returns:
            Mapping suggestions with confidence scores
        """
        columns = list(df.columns)
        sample = df.head(10).to_dict(orient='records')

        grok_client = get_grok_client()
        if grok_client:
            suggestions = self._ai_mapping(grok_client, columns, sample, filename)
        else:
            suggestions = self._heuristic_mapping(columns)
            suggestions["notes"] = "No GROK_API_KEY - heuristic fallback"

        # Ensure confidence scores exist
        if "confidence" not in suggestions:
            suggestions["confidence"] = {
                col: 1.0 if suggestions["mapping"].get(col) else 0.0
                for col in columns
            }

        return {
            "original_columns": columns,
            "sample_data": sample,
            "suggestions": suggestions["mapping"],
            "confidences": suggestions["confidence"],
            "notes": suggestions.get("notes", "")
        }

    def _ai_mapping(
        self,
        client,
        columns: List[str],
        sample: List[Dict],
        filename: str
    ) -> Dict:
        """Use Grok AI for intelligent column mapping."""
        prompt = f"""
You are Profit Sentinel's expert semi-agentic column mapper for messy POS/ERP exports.
Task: Suggest mapping from uploaded columns to our standard fields using BOTH column names AND sample values.

Uploaded file: {filename}
Columns: {columns}
First 10 rows sample:
{json.dumps(sample, indent=2)}

Standard fields (preferred first, with common aliases):
{json.dumps(STANDARD_FIELDS, indent=2)}

Rules:
- Use sample values to disambiguate (e.g., dates -> date, $-numbers -> revenue/cost, negative qty -> return_flag).
- Common tricks: "Ext Price"/"Line Total"/"Amount" -> revenue, "Avg Cost" -> cost.
- Only map if confident (>0.6 internally).
- Use EXACT standard field name or null.

Return ONLY valid JSON:
{{
  "mapping": {{"Uploaded Column Name": "standard_field" or null}},
  "confidence": {{"Uploaded Column Name": 0.0-1.0}},
  "notes": "Brief explanation of guesses/unmapped"
}}
"""

        try:
            response = client.chat.completions.create(
                model="grok-3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            content = response.choices[0].message.content.strip()

            # Clean JSON from markdown code blocks
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            return json.loads(content)

        except Exception as e:
            logger.warning(f"Grok mapping failed: {e}")
            result = self._heuristic_mapping(columns)
            result["notes"] = f"Grok failed ({str(e)}) - heuristic fallback"
            return result

    def _heuristic_mapping(self, columns: List[str]) -> Dict:
        """Fallback heuristic mapping based on column names."""
        mapping = {}
        confidence = {}

        for col in columns:
            clean = col.strip().lower().replace(' ', '').replace('$', '').replace('.', '')
            matched = False

            for std, examples in STANDARD_FIELDS.items():
                if any(ex.replace(' ', '') in clean for ex in examples):
                    mapping[col] = std
                    confidence[col] = 0.8
                    matched = True
                    break

            if not matched:
                mapping[col] = None
                confidence[col] = 0.0

        return {
            "mapping": mapping,
            "confidence": confidence,
            "notes": "Heuristic keyword match"
        }
