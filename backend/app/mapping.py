# backend/app/mapping.py
import pandas as pd
import io
import json
import os
from openai import OpenAI
from typing import Dict

STANDARD_FIELDS = { ... }  # your existing dict

def get_df_from_s3(key: str) -> pd.DataFrame:
    obj = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=key)
    contents = obj['Body'].read()
    
    if key.lower().endswith('.csv'):
        return pd.read_csv(io.BytesIO(contents), nrows=50, dtype=str, keep_default_na=False)  # sample only
    else:  # xlsx/xls
        return pd.read_excel(io.BytesIO(contents), nrows=50, dtype=str)

def suggest_column_mapping(df: pd.DataFrame, filename: str) -> Dict:
    columns = list(df.columns)
    sample = df.head(10).to_dict(orient='records')
    
    client = OpenAI(api_key=os.getenv(""), base_url="https://api.x.ai/v1")
    
    prompt = f"""
You are Profit Sentinel's expert column mapper for messy POS/ERP exports.

Task: Map uploaded columns to our standard fields using both column names AND actual sample values.

Uploaded file: {filename}
Columns: {columns}
First 10 rows sample:
{json.dumps(sample, indent=2)}

Standard fields (preferred name first):
{json.dumps({k: v for k, v in STANDARD_FIELDS.items()}, indent=2)}

Rules:
- Match semantically, not just keyword (e.g., "Ext Amt" or "Line Total" → revenue).
- Look at sample values for clues (dates → date, numbers with $ → revenue/cost, negative qty → return_flag).
- Be conservative: only map if confidence >0.6 internally.
- Use EXACT standard field name or null.

Return ONLY valid JSON:
{{
  "mapping": {{"Uploaded Column Name": "standard_field" or null}},
  "confidence": {{"Uploaded Column Name": 0.0-1.0}},
  "notes": "Brief explanation of tricky guesses or unmapped columns"
}}
"""

    response = client.chat.completions.create(
        model="grok-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )
    
    try:
        suggestions = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        suggestions = {"mapping": {}, "confidence": {}, "notes": f"Parse failed: {e}"}

    # Fallback heuristic if Grok bombs
    if not suggestions["mapping"]:
        fallback = {}
        conf = {}
        for col in columns:
            clean = col.strip().lower().replace(' ', '').replace('$', '')
            matched = False
            for std, examples in STANDARD_FIELDS.items():
                if any(ex.replace(' ', '') in clean for ex in examples):
                    fallback[col] = std
                    conf[col] = 0.6
                    matched = True
                    break
            if not matched:
                fallback[col] = None
                conf[col] = 0.0
        suggestions = {"mapping": fallback, "confidence": conf, "notes": "Used fallback heuristic"}

    return {
        "original_columns": columns,
        "sample_data": sample,
        "suggestions": suggestions["mapping"],
        "confidences": suggestions.get("confidence", {}),
        "notes": suggestions.get("notes", "")
    }