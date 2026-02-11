"""POS System Integrations.

Provides connection management and data sync configuration for
Square, Lightspeed, Clover, and Shopify POS systems.

Architecture:
    - PosConnection: stored OAuth credentials + sync config per integration
    - PosConnector: abstract interface for each POS system
    - Concrete connectors: SquareConnector, LightspeedConnector, etc.
    - Sync engine: orchestrates data pull → normalize → analyze pipeline

This module defines the connection lifecycle:
    1. User selects POS system
    2. OAuth redirect → get access token
    3. Store connection config
    4. Test connection / pull sample data
    5. Schedule automatic syncs

Note: actual OAuth tokens are never stored in this layer in production —
they go to Supabase vault. This module manages the connection metadata
and provides the interface for the sync pipeline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

logger = logging.getLogger("sentinel.pos")


# ---------------------------------------------------------------------------
# Supported POS systems
# ---------------------------------------------------------------------------


class PosSystem(str, Enum):
    SQUARE = "square"
    LIGHTSPEED = "lightspeed"
    CLOVER = "clover"
    SHOPIFY = "shopify"

    @property
    def display_name(self) -> str:
        return {
            PosSystem.SQUARE: "Square POS",
            PosSystem.LIGHTSPEED: "Lightspeed Retail",
            PosSystem.CLOVER: "Clover POS",
            PosSystem.SHOPIFY: "Shopify POS",
        }[self]

    @property
    def auth_type(self) -> str:
        return {
            PosSystem.SQUARE: "oauth2",
            PosSystem.LIGHTSPEED: "oauth2",
            PosSystem.CLOVER: "oauth2",
            PosSystem.SHOPIFY: "oauth2",
        }[self]

    @property
    def docs_url(self) -> str:
        return {
            PosSystem.SQUARE: "https://developer.squareup.com/docs",
            PosSystem.LIGHTSPEED: "https://developers.lightspeedhq.com",
            PosSystem.CLOVER: "https://docs.clover.com",
            PosSystem.SHOPIFY: "https://shopify.dev",
        }[self]

    @property
    def inventory_fields(self) -> list[str]:
        """Key inventory fields this POS provides."""
        base = ["sku", "description", "qty_on_hand", "unit_cost", "retail_price"]
        extras = {
            PosSystem.SQUARE: ["category", "vendor", "barcode"],
            PosSystem.LIGHTSPEED: ["category", "vendor", "reorder_point", "vendor_id"],
            PosSystem.CLOVER: ["category", "barcode", "price_type"],
            PosSystem.SHOPIFY: ["vendor", "product_type", "tags", "barcode"],
        }
        return base + extras.get(self, [])


class SyncFrequency(str, Enum):
    MANUAL = "manual"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()


class ConnectionStatus(str, Enum):
    PENDING = "pending"
    CONNECTED = "connected"
    ERROR = "error"
    EXPIRED = "expired"
    DISCONNECTED = "disconnected"


# ---------------------------------------------------------------------------
# Connection data classes
# ---------------------------------------------------------------------------


@dataclass
class PosConnection:
    """A configured POS system connection."""

    connection_id: str
    user_id: str
    pos_system: PosSystem
    store_name: str
    status: ConnectionStatus
    sync_frequency: SyncFrequency
    created_at: datetime
    last_sync_at: datetime | None = None
    last_sync_status: str | None = None
    last_sync_rows: int = 0
    location_id: str | None = None  # POS-specific location/store ID
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "connection_id": self.connection_id,
            "user_id": self.user_id,
            "pos_system": self.pos_system.value,
            "pos_system_display": self.pos_system.display_name,
            "store_name": self.store_name,
            "status": self.status.value,
            "sync_frequency": self.sync_frequency.value,
            "created_at": self.created_at.isoformat(),
            "last_sync_at": (
                self.last_sync_at.isoformat() if self.last_sync_at else None
            ),
            "last_sync_status": self.last_sync_status,
            "last_sync_rows": self.last_sync_rows,
            "location_id": self.location_id,
            "inventory_fields": self.pos_system.inventory_fields,
        }


@dataclass
class SyncResult:
    """Result of a data sync operation."""

    connection_id: str
    success: bool
    rows_synced: int
    errors: list[str]
    duration_seconds: float
    analysis_triggered: bool
    analysis_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "connection_id": self.connection_id,
            "success": self.success,
            "rows_synced": self.rows_synced,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
            "analysis_triggered": self.analysis_triggered,
            "analysis_id": self.analysis_id,
        }


@dataclass
class PosSystemInfo:
    """Information about a supported POS system."""

    system: PosSystem
    display_name: str
    auth_type: str
    inventory_fields: list[str]
    docs_url: str
    setup_steps: list[str]

    def to_dict(self) -> dict:
        return {
            "system": self.system.value,
            "display_name": self.display_name,
            "auth_type": self.auth_type,
            "inventory_fields": self.inventory_fields,
            "docs_url": self.docs_url,
            "setup_steps": self.setup_steps,
        }


# ---------------------------------------------------------------------------
# Setup steps per POS system
# ---------------------------------------------------------------------------

_SETUP_STEPS: dict[PosSystem, list[str]] = {
    PosSystem.SQUARE: [
        "Click 'Connect Square' to authorize Profit Sentinel",
        "Select which Square location(s) to sync",
        "Choose sync frequency (daily recommended)",
        "Run initial sync to import current inventory",
    ],
    PosSystem.LIGHTSPEED: [
        "Click 'Connect Lightspeed' to authorize via OAuth",
        "Select your Lightspeed account and store",
        "Map inventory categories (auto-detected)",
        "Choose sync frequency and run initial import",
    ],
    PosSystem.CLOVER: [
        "Click 'Connect Clover' to authorize",
        "Select your Clover merchant account",
        "Verify inventory categories match",
        "Choose sync frequency and run initial import",
    ],
    PosSystem.SHOPIFY: [
        "Install Profit Sentinel from the Shopify App Store",
        "Authorize inventory data access",
        "Select which locations to include",
        "Choose sync frequency and run initial import",
    ],
}


# ---------------------------------------------------------------------------
# Connection store (in-memory for dev, Supabase for production)
# ---------------------------------------------------------------------------


class InMemoryPosConnectionStore:
    """In-memory POS connection store."""

    def __init__(self):
        self._connections: dict[str, PosConnection] = {}
        self._user_connections: dict[str, list[str]] = defaultdict(list)
        self._id_counter = 0

    def create_connection(
        self,
        user_id: str,
        pos_system: PosSystem,
        store_name: str,
        sync_frequency: SyncFrequency = SyncFrequency.DAILY,
        location_id: str | None = None,
    ) -> PosConnection:
        """Create a new POS connection."""
        self._id_counter += 1
        conn_id = f"pos_{self._id_counter:04d}"

        conn = PosConnection(
            connection_id=conn_id,
            user_id=user_id,
            pos_system=pos_system,
            store_name=store_name,
            status=ConnectionStatus.CONNECTED,
            sync_frequency=sync_frequency,
            created_at=datetime.now(UTC),
            location_id=location_id,
        )

        self._connections[conn_id] = conn
        self._user_connections[user_id].append(conn_id)
        return conn

    def list_connections(self, user_id: str) -> list[PosConnection]:
        """List all connections for a user."""
        conn_ids = self._user_connections.get(user_id, [])
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]

    def get_connection(self, connection_id: str, user_id: str) -> PosConnection | None:
        """Get a specific connection."""
        conn = self._connections.get(connection_id)
        if conn and conn.user_id == user_id:
            return conn
        return None

    def update_sync_status(
        self,
        connection_id: str,
        status: str,
        rows: int = 0,
    ) -> bool:
        """Update the last sync status for a connection."""
        conn = self._connections.get(connection_id)
        if not conn:
            return False
        conn.last_sync_at = datetime.now(UTC)
        conn.last_sync_status = status
        conn.last_sync_rows = rows
        return True

    def disconnect(self, connection_id: str, user_id: str) -> bool:
        """Disconnect a POS integration."""
        conn = self._connections.get(connection_id)
        if not conn or conn.user_id != user_id:
            return False
        conn.status = ConnectionStatus.DISCONNECTED
        return True

    def delete_connection(self, connection_id: str, user_id: str) -> bool:
        """Delete a connection entirely."""
        conn = self._connections.get(connection_id)
        if not conn or conn.user_id != user_id:
            return False
        del self._connections[connection_id]
        self._user_connections[user_id].remove(connection_id)
        return True


# ---------------------------------------------------------------------------
# Global store
# ---------------------------------------------------------------------------

_store: InMemoryPosConnectionStore | None = None


def init_pos_store(store: InMemoryPosConnectionStore | None = None) -> None:
    """Initialize the global POS connection store."""
    global _store
    _store = store or InMemoryPosConnectionStore()


def _get_store() -> InMemoryPosConnectionStore:
    """Get the global store, initializing if needed."""
    global _store
    if _store is None:
        _store = InMemoryPosConnectionStore()
    return _store


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_supported_systems() -> list[PosSystemInfo]:
    """Get information about all supported POS systems."""
    return [
        PosSystemInfo(
            system=system,
            display_name=system.display_name,
            auth_type=system.auth_type,
            inventory_fields=system.inventory_fields,
            docs_url=system.docs_url,
            setup_steps=_SETUP_STEPS.get(system, []),
        )
        for system in PosSystem
    ]


def create_pos_connection(
    user_id: str,
    pos_system: str,
    store_name: str,
    sync_frequency: str = "daily",
    location_id: str | None = None,
) -> PosConnection:
    """Create a new POS connection."""
    return _get_store().create_connection(
        user_id=user_id,
        pos_system=PosSystem(pos_system),
        store_name=store_name,
        sync_frequency=SyncFrequency(sync_frequency),
        location_id=location_id,
    )


def list_pos_connections(user_id: str) -> list[PosConnection]:
    """List all POS connections for a user."""
    return _get_store().list_connections(user_id)


def get_pos_connection(connection_id: str, user_id: str) -> PosConnection | None:
    """Get a specific POS connection."""
    return _get_store().get_connection(connection_id, user_id)


def disconnect_pos(connection_id: str, user_id: str) -> bool:
    """Disconnect a POS integration."""
    return _get_store().disconnect(connection_id, user_id)


def delete_pos_connection(connection_id: str, user_id: str) -> bool:
    """Delete a POS connection."""
    return _get_store().delete_connection(connection_id, user_id)


def trigger_sync(
    connection_id: str,
    user_id: str,
) -> SyncResult:
    """Trigger a manual data sync for a POS connection.

    In production, this would:
    1. Pull inventory data from the POS API
    2. Normalize to Profit Sentinel format
    3. Upload to S3 and trigger analysis
    4. Return the sync result + analysis_id

    Currently returns a simulated result for the connection lifecycle.
    """
    conn = get_pos_connection(connection_id, user_id)
    if not conn:
        return SyncResult(
            connection_id=connection_id,
            success=False,
            rows_synced=0,
            errors=["Connection not found"],
            duration_seconds=0,
            analysis_triggered=False,
        )

    if conn.status != ConnectionStatus.CONNECTED:
        return SyncResult(
            connection_id=connection_id,
            success=False,
            rows_synced=0,
            errors=[f"Connection is {conn.status.value}, not connected"],
            duration_seconds=0,
            analysis_triggered=False,
        )

    # Simulated sync (production would call the actual POS API)
    _get_store().update_sync_status(connection_id, "success", rows=0)

    return SyncResult(
        connection_id=connection_id,
        success=True,
        rows_synced=0,
        errors=[],
        duration_seconds=0.0,
        analysis_triggered=False,
        analysis_id=None,
    )
