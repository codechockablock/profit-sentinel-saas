"""
PROFIT SENTINEL - SEMANTIC FLAGGING MODULE
===========================================

Flags queries for employer review without blocking execution.
Designed to surface potentially sensitive patterns while maintaining
full system capability.

Architecture:
    Query → VSA Processing → LLM Interpretation → Response (immediate)
                                   ↓
                          Semantic Flag Check (async)
                                   ↓
                          [If flagged: log + queue for review]
"""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class FlagCategory(Enum):
    """Categories of flagged content for review"""
    INDIVIDUAL_SURVEILLANCE = "individual_surveillance"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    PRICING_COORDINATION = "pricing_coordination"
    ACCUMULATION_PATTERN = "accumulation_pattern"
    DATA_EXFILTRATION = "data_exfiltration"
    UNUSUAL_SCOPE = "unusual_scope"


class FlagSeverity(Enum):
    """Severity levels - all get logged, severity affects review priority"""
    LOW = "low"          # Informational, batch review weekly
    MEDIUM = "medium"    # Review within 48 hours
    HIGH = "high"        # Review within 24 hours
    CRITICAL = "critical"  # Immediate notification


@dataclass
class SemanticFlag:
    """A single flag raised on a query"""
    category: FlagCategory
    severity: FlagSeverity
    reason: str
    matched_patterns: list[str]
    confidence: float  # 0.0 - 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d['category'] = self.category.value
        d['severity'] = self.severity.value
        return d


@dataclass
class FlaggedQuery:
    """Complete record of a flagged query for review"""
    query_id: str
    user_id: str
    session_id: str
    timestamp: str
    raw_query: str
    vsa_patterns_accessed: list[str]
    llm_response_summary: str
    flags: list[SemanticFlag]
    context: dict  # Additional metadata
    reviewed: bool = False
    review_notes: str | None = None
    reviewer_id: str | None = None
    review_timestamp: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['flags'] = [f.to_dict() if isinstance(f, SemanticFlag) else f for f in self.flags]
        return d


class SemanticFlagDetector:
    """
    Detects potentially sensitive query patterns for employer review.

    Design principles:
    - Never blocks queries, only flags
    - Low false positive tolerance (don't cry wolf)
    - Employer has full context, we just surface patterns
    - Accumulation detection catches slow-burn concerns
    """

    def __init__(self, config: dict | None = None):
        self.config = config or self._default_config()
        self.user_query_history: dict[str, list[dict]] = defaultdict(list)
        self.flagged_queries: list[FlaggedQuery] = []

    def _default_config(self) -> dict:
        return {
            # Individual surveillance detection
            "individual_patterns": {
                "enabled": True,
                "id_patterns": [
                    r"ASSOCIATE[_-]?\d+",
                    r"EMPLOYEE[_-]?\d+",
                    r"EMP[_-]?\d+",
                    r"STAFF[_-]?\d+",
                    r"USER[_-]?\d+",
                    r"CASHIER[_-]?\d+",
                ],
                "behavioral_keywords": [
                    "movement", "location", "proximity", "association",
                    "correlation", "pattern", "behavior", "activity",
                    "transaction gap", "void rate", "break", "deviation",
                    "anomaly", "relationship", "contact", "interaction"
                ],
                "threshold_mentions": 2,  # Flag if ID + N behavioral keywords
            },

            # Competitive intelligence detection
            "competitive_patterns": {
                "enabled": True,
                "competitor_indicators": [
                    r"COMPETITOR[_-]?[A-Z0-9]+",
                    r"competitor'?s?",
                    r"rival",
                    r"competing\s+store",
                ],
                "sensitive_topics": [
                    "supplier", "vendor", "contract", "renewal",
                    "vulnerability", "weakness", "account", "customer",
                    "poach", "target", "undercut", "steal",
                    "reverse engineer", "infer", "source from"
                ],
                "threshold_matches": 2,
            },

            # Pricing coordination detection
            "pricing_patterns": {
                "enabled": True,
                "coordination_language": [
                    "equilibrium", "stability", "floor price", "ceiling",
                    "coordinate", "signal", "match", "follow",
                    "defector", "retaliate", "punish", "pressure",
                    "tacit", "implicit", "without direct",
                    "market stability", "regional pricing"
                ],
                "threshold_matches": 2,
            },

            # Accumulation detection (slow-burn patterns)
            "accumulation_patterns": {
                "enabled": True,
                "window_hours": 168,  # 7 days
                "related_query_threshold": 5,  # N similar queries in window
                "similarity_threshold": 0.6,
            },

            # Unusual scope detection
            "scope_patterns": {
                "enabled": True,
                "bulk_indicators": [
                    "all employees", "every associate", "full export",
                    "complete list", "entire database", "bulk download"
                ],
            },

            # Review settings
            "review": {
                "immediate_notify_severities": ["critical", "high"],
                "digest_frequency_hours": 24,
                "retention_days": 90,
            }
        }

    def analyze_query(
        self,
        query: str,
        user_id: str,
        session_id: str,
        vsa_patterns: list[str] | None = None,
        llm_response: str | None = None,
        context: dict | None = None
    ) -> tuple[list[SemanticFlag], FlaggedQuery | None]:
        """
        Analyze a query for sensitive patterns. Returns flags and optionally
        a complete FlaggedQuery record if any flags were raised.

        This runs AFTER the query executes - it never blocks.
        """
        flags = []
        vsa_patterns = vsa_patterns or []
        context = context or {}

        # Check each detection category
        if self.config["individual_patterns"]["enabled"]:
            flag = self._check_individual_surveillance(query, vsa_patterns)
            if flag:
                flags.append(flag)

        if self.config["competitive_patterns"]["enabled"]:
            flag = self._check_competitive_intelligence(query, vsa_patterns)
            if flag:
                flags.append(flag)

        if self.config["pricing_patterns"]["enabled"]:
            flag = self._check_pricing_coordination(query, vsa_patterns)
            if flag:
                flags.append(flag)

        if self.config["accumulation_patterns"]["enabled"]:
            flag = self._check_accumulation_pattern(query, user_id)
            if flag:
                flags.append(flag)

        if self.config["scope_patterns"]["enabled"]:
            flag = self._check_unusual_scope(query)
            if flag:
                flags.append(flag)

        # Record query in history for accumulation detection
        self._record_query(query, user_id)

        # Create flagged query record if any flags
        flagged_query = None
        if flags:
            query_id = self._generate_query_id(query, user_id, session_id)
            flagged_query = FlaggedQuery(
                query_id=query_id,
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.utcnow().isoformat(),
                raw_query=query,
                vsa_patterns_accessed=vsa_patterns,
                llm_response_summary=self._summarize_response(llm_response),
                flags=flags,
                context=context
            )
            self.flagged_queries.append(flagged_query)

        return flags, flagged_query

    def _check_individual_surveillance(
        self,
        query: str,
        vsa_patterns: list[str]
    ) -> SemanticFlag | None:
        """Detect queries that may constitute individual employee surveillance"""
        config = self.config["individual_patterns"]
        combined_text = f"{query} {' '.join(vsa_patterns)}".lower()

        # Find individual identifiers
        id_matches = []
        for pattern in config["id_patterns"]:
            matches = re.findall(pattern, query, re.IGNORECASE)
            id_matches.extend(matches)

        if not id_matches:
            return None

        # Count behavioral keywords
        behavioral_matches = []
        for keyword in config["behavioral_keywords"]:
            if keyword.lower() in combined_text:
                behavioral_matches.append(keyword)

        if len(behavioral_matches) >= config["threshold_mentions"]:
            # Determine severity based on specificity
            if len(id_matches) > 3 or "relationship" in behavioral_matches:
                severity = FlagSeverity.HIGH
            elif len(behavioral_matches) > 4:
                severity = FlagSeverity.MEDIUM
            else:
                severity = FlagSeverity.LOW

            return SemanticFlag(
                category=FlagCategory.INDIVIDUAL_SURVEILLANCE,
                severity=severity,
                reason=f"Query references specific individual(s) ({', '.join(id_matches[:3])}) with behavioral analysis keywords",
                matched_patterns=id_matches[:5] + behavioral_matches[:5],
                confidence=min(0.5 + (len(behavioral_matches) * 0.1), 0.95)
            )

        return None

    def _check_competitive_intelligence(
        self,
        query: str,
        vsa_patterns: list[str]
    ) -> SemanticFlag | None:
        """Detect queries that may constitute competitive espionage"""
        config = self.config["competitive_patterns"]
        combined_text = f"{query} {' '.join(vsa_patterns)}".lower()

        # Find competitor references
        competitor_matches = []
        for pattern in config["competitor_indicators"]:
            matches = re.findall(pattern, query, re.IGNORECASE)
            competitor_matches.extend(matches)

        if not competitor_matches:
            return None

        # Check sensitive topics
        topic_matches = []
        for topic in config["sensitive_topics"]:
            if topic.lower() in combined_text:
                topic_matches.append(topic)

        if len(topic_matches) >= config["threshold_matches"]:
            # High severity for account targeting or reverse engineering
            high_severity_topics = {"poach", "target", "vulnerability", "reverse engineer", "steal"}
            if any(t in high_severity_topics for t in topic_matches):
                severity = FlagSeverity.HIGH
            else:
                severity = FlagSeverity.MEDIUM

            return SemanticFlag(
                category=FlagCategory.COMPETITIVE_INTELLIGENCE,
                severity=severity,
                reason="Query combines competitor references with sensitive business intelligence topics",
                matched_patterns=competitor_matches[:3] + topic_matches[:5],
                confidence=min(0.4 + (len(topic_matches) * 0.15), 0.9)
            )

        return None

    def _check_pricing_coordination(
        self,
        query: str,
        vsa_patterns: list[str]
    ) -> SemanticFlag | None:
        """Detect queries that may indicate price fixing intent"""
        config = self.config["pricing_patterns"]
        combined_text = f"{query} {' '.join(vsa_patterns)}".lower()

        # Check coordination language
        matches = []
        for term in config["coordination_language"]:
            if term.lower() in combined_text:
                matches.append(term)

        if len(matches) >= config["threshold_matches"]:
            # Critical severity for explicit coordination language
            critical_terms = {"coordinate", "signal", "defector", "retaliate", "tacit"}
            if len(set(matches) & critical_terms) >= 2:
                severity = FlagSeverity.CRITICAL
            elif any(t in critical_terms for t in matches):
                severity = FlagSeverity.HIGH
            else:
                severity = FlagSeverity.MEDIUM

            return SemanticFlag(
                category=FlagCategory.PRICING_COORDINATION,
                severity=severity,
                reason="Query contains language associated with pricing coordination",
                matched_patterns=matches[:7],
                confidence=min(0.5 + (len(matches) * 0.1), 0.95)
            )

        return None

    def _check_accumulation_pattern(
        self,
        query: str,
        user_id: str
    ) -> SemanticFlag | None:
        """Detect users building up sensitive analysis over multiple queries"""
        config = self.config["accumulation_patterns"]
        window = timedelta(hours=config["window_hours"])
        cutoff = datetime.utcnow() - window

        # Get recent queries for this user
        recent = [
            q for q in self.user_query_history[user_id]
            if datetime.fromisoformat(q["timestamp"]) > cutoff
        ]

        if len(recent) < config["related_query_threshold"]:
            return None

        # Simple similarity check - count shared significant words
        query_words = set(self._extract_significant_words(query))
        similar_count = 0
        related_queries = []

        for past in recent:
            past_words = set(self._extract_significant_words(past["query"]))
            if query_words and past_words:
                overlap = len(query_words & past_words) / max(len(query_words), len(past_words))
                if overlap >= config["similarity_threshold"]:
                    similar_count += 1
                    related_queries.append(past["query"][:50])

        if similar_count >= config["related_query_threshold"]:
            return SemanticFlag(
                category=FlagCategory.ACCUMULATION_PATTERN,
                severity=FlagSeverity.MEDIUM,
                reason=f"User has made {similar_count} similar queries in the past {config['window_hours']} hours",
                matched_patterns=related_queries[:5],
                confidence=min(0.3 + (similar_count * 0.1), 0.8)
            )

        return None

    def _check_unusual_scope(self, query: str) -> SemanticFlag | None:
        """Detect queries requesting unusually broad data access"""
        config = self.config["scope_patterns"]
        query_lower = query.lower()

        matches = []
        for indicator in config["bulk_indicators"]:
            if indicator.lower() in query_lower:
                matches.append(indicator)

        if matches:
            return SemanticFlag(
                category=FlagCategory.UNUSUAL_SCOPE,
                severity=FlagSeverity.LOW,
                reason="Query requests broad data scope",
                matched_patterns=matches,
                confidence=0.6
            )

        return None

    def _extract_significant_words(self, text: str) -> list[str]:
        """Extract significant words for similarity comparison"""
        # Remove common words, keep nouns/verbs that indicate intent
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "what", "which", "who", "whom", "whose",
            "where", "when", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "and", "but", "or", "if", "because", "as", "until",
            "while", "of", "at", "by", "for", "with", "about", "against",
            "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "any", "please", "show", "me", "get",
            "find", "tell", "give", "analyze", "analysis", "data", "report"
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in stopwords]

    def _record_query(self, query: str, user_id: str):
        """Record query for accumulation detection"""
        self.user_query_history[user_id].append({
            "query": query,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Prune old entries
        window = timedelta(hours=self.config["accumulation_patterns"]["window_hours"])
        cutoff = datetime.utcnow() - window
        self.user_query_history[user_id] = [
            q for q in self.user_query_history[user_id]
            if datetime.fromisoformat(q["timestamp"]) > cutoff
        ]

    def _generate_query_id(self, query: str, user_id: str, session_id: str) -> str:
        """Generate unique ID for flagged query"""
        content = f"{query}{user_id}{session_id}{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _summarize_response(self, response: str | None) -> str:
        """Create brief summary of LLM response for review context"""
        if not response:
            return "[No response captured]"
        if len(response) <= 200:
            return response
        return response[:200] + "..."

    # === REVIEW INTERFACE ===

    def get_pending_reviews(
        self,
        severity_filter: list[FlagSeverity] | None = None,
        category_filter: list[FlagCategory] | None = None,
        limit: int = 50
    ) -> list[FlaggedQuery]:
        """Get flagged queries pending review"""
        pending = [q for q in self.flagged_queries if not q.reviewed]

        if severity_filter:
            pending = [
                q for q in pending
                if any(f.severity in severity_filter for f in q.flags)
            ]

        if category_filter:
            pending = [
                q for q in pending
                if any(f.category in category_filter for f in q.flags)
            ]

        # Sort by highest severity flag
        severity_order = {
            FlagSeverity.CRITICAL: 0,
            FlagSeverity.HIGH: 1,
            FlagSeverity.MEDIUM: 2,
            FlagSeverity.LOW: 3
        }

        def max_severity(q):
            return min(severity_order[f.severity] for f in q.flags)

        pending.sort(key=max_severity)
        return pending[:limit]

    def mark_reviewed(
        self,
        query_id: str,
        reviewer_id: str,
        notes: str,
        action_taken: str | None = None
    ):
        """Mark a flagged query as reviewed"""
        for q in self.flagged_queries:
            if q.query_id == query_id:
                q.reviewed = True
                q.reviewer_id = reviewer_id
                q.review_notes = notes
                q.review_timestamp = datetime.utcnow().isoformat()
                if action_taken:
                    q.context["action_taken"] = action_taken
                break

    def generate_digest(
        self,
        hours: int = 24,
        include_reviewed: bool = False
    ) -> dict:
        """Generate summary digest for employer review"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        relevant = [
            q for q in self.flagged_queries
            if datetime.fromisoformat(q.timestamp) > cutoff
            and (include_reviewed or not q.reviewed)
        ]

        # Aggregate by category
        by_category = defaultdict(list)
        for q in relevant:
            for flag in q.flags:
                by_category[flag.category.value].append({
                    "query_id": q.query_id,
                    "user_id": q.user_id,
                    "severity": flag.severity.value,
                    "reason": flag.reason,
                    "timestamp": q.timestamp
                })

        # Count by severity
        severity_counts = defaultdict(int)
        for q in relevant:
            max_sev = max(f.severity.value for f in q.flags)
            severity_counts[max_sev] += 1

        # Identify repeat users
        user_counts = defaultdict(int)
        for q in relevant:
            user_counts[q.user_id] += 1
        repeat_users = {u: c for u, c in user_counts.items() if c > 1}

        return {
            "period_hours": hours,
            "generated_at": datetime.utcnow().isoformat(),
            "total_flagged_queries": len(relevant),
            "pending_review": len([q for q in relevant if not q.reviewed]),
            "severity_breakdown": dict(severity_counts),
            "category_breakdown": {k: len(v) for k, v in by_category.items()},
            "repeat_flagged_users": repeat_users,
            "queries_by_category": dict(by_category),
            "immediate_attention": [
                q.to_dict() for q in relevant
                if any(f.severity in [FlagSeverity.CRITICAL, FlagSeverity.HIGH] for f in q.flags)
                and not q.reviewed
            ]
        }

    def export_for_review(self, filepath: str):
        """Export all flagged queries to JSON for external review"""
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "total_records": len(self.flagged_queries),
            "flagged_queries": [q.to_dict() for q in self.flagged_queries]
        }
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# === INTEGRATION HELPER ===

class ProfitSentinelFlagIntegration:
    """
    Drop-in integration for Profit Sentinel query pipeline.

    Usage:
        flagger = ProfitSentinelFlagIntegration()

        # In your query handler:
        response = process_query(query)  # Normal processing
        flagger.check_and_log(query, user_id, session_id, vsa_patterns, response)
        return response  # Response always returned, flagging is async
    """

    def __init__(self, config: dict | None = None):
        self.detector = SemanticFlagDetector(config)
        self.notification_callbacks = []

    def register_notification_callback(self, callback):
        """Register callback for immediate notifications (high/critical severity)"""
        self.notification_callbacks.append(callback)

    def check_and_log(
        self,
        query: str,
        user_id: str,
        session_id: str,
        vsa_patterns: list[str] | None = None,
        llm_response: str | None = None,
        context: dict | None = None
    ) -> list[SemanticFlag]:
        """
        Check query for flags and log if needed.
        Returns flags but NEVER blocks the query.
        """
        flags, flagged_query = self.detector.analyze_query(
            query=query,
            user_id=user_id,
            session_id=session_id,
            vsa_patterns=vsa_patterns,
            llm_response=llm_response,
            context=context
        )

        # Immediate notification for high severity
        if flagged_query:
            high_severity = any(
                f.severity in [FlagSeverity.CRITICAL, FlagSeverity.HIGH]
                for f in flags
            )
            if high_severity:
                for callback in self.notification_callbacks:
                    try:
                        callback(flagged_query)
                    except Exception:
                        pass  # Don't let notification failures affect query

        return flags

    def get_review_dashboard_data(self) -> dict:
        """Get data for employer review dashboard"""
        return {
            "digest_24h": self.detector.generate_digest(hours=24),
            "digest_7d": self.detector.generate_digest(hours=168),
            "pending_critical": self.detector.get_pending_reviews(
                severity_filter=[FlagSeverity.CRITICAL, FlagSeverity.HIGH],
                limit=10
            ),
            "pending_all": self.detector.get_pending_reviews(limit=50)
        }


# === DEMONSTRATION ===

if __name__ == "__main__":
    print("=" * 70)
    print("PROFIT SENTINEL SEMANTIC FLAGGING MODULE - DEMONSTRATION")
    print("=" * 70)

    flagger = ProfitSentinelFlagIntegration()

    # Test queries
    test_queries = [
        {
            "query": "Show me transaction patterns for ASSOCIATE_4471 including break times and which other employees they interact with",
            "user_id": "manager_01",
            "session_id": "sess_001",
            "vsa_patterns": ["TEMPORAL_CLUSTERING", "BEHAVIORAL_CORRELATION"],
            "expected_flag": "INDIVIDUAL_SURVEILLANCE"
        },
        {
            "query": "Analyze COMPETITOR_A's supplier relationships and identify when their contracts renew so we can approach their vendors",
            "user_id": "buyer_02",
            "session_id": "sess_002",
            "vsa_patterns": ["SUPPLIER_INFERENCE", "CONTRACT_TIMING"],
            "expected_flag": "COMPETITIVE_INTELLIGENCE"
        },
        {
            "query": "What's the regional price equilibrium for lumber and how can we signal to competitors to maintain floor pricing?",
            "user_id": "pricing_mgr",
            "session_id": "sess_003",
            "vsa_patterns": ["PRICE_EQUILIBRIUM", "MARKET_SIGNALING"],
            "expected_flag": "PRICING_COORDINATION"
        },
        {
            "query": "Show me the margin analysis for SKU 1234567",
            "user_id": "analyst_01",
            "session_id": "sess_004",
            "vsa_patterns": ["MARGIN_LEAK"],
            "expected_flag": None  # Should NOT flag - normal query
        },
        {
            "query": "What are our top short-ship vendors this quarter?",
            "user_id": "analyst_01",
            "session_id": "sess_005",
            "vsa_patterns": ["VENDOR_PERFORMANCE"],
            "expected_flag": None  # Should NOT flag - normal query
        }
    ]

    print("\nRunning test queries...\n")

    for test in test_queries:
        flags = flagger.check_and_log(
            query=test["query"],
            user_id=test["user_id"],
            session_id=test["session_id"],
            vsa_patterns=test["vsa_patterns"]
        )

        print(f"Query: {test['query'][:60]}...")
        print(f"  Expected flag: {test['expected_flag']}")
        print(f"  Actual flags: {[f.category.value for f in flags] if flags else 'None'}")

        if flags:
            for flag in flags:
                print(f"    - {flag.category.value}: {flag.severity.value}")
                print(f"      Reason: {flag.reason}")
                print(f"      Confidence: {flag.confidence:.2f}")

        expected = test["expected_flag"]
        actual = [f.category.value for f in flags] if flags else []

        if (expected is None and not actual) or (expected and expected.lower() in [a.lower() for a in actual]):
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
        print()

    print("=" * 70)
    print("GENERATING REVIEW DIGEST")
    print("=" * 70)

    digest = flagger.detector.generate_digest(hours=24)
    print(json.dumps(digest, indent=2, default=str))

    print("\n" + "=" * 70)
    print("MODULE READY FOR INTEGRATION")
    print("=" * 70)
