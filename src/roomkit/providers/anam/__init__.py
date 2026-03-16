"""Anam AI avatar provider."""

from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.anam.realtime import AnamRealtimeProvider

__all__ = ["AnamConfig", "AnamRealtimeProvider", "AnamSIPBridge"]


def __getattr__(name: str) -> object:
    if name == "AnamSIPBridge":
        from roomkit.providers.anam.sip_bridge import AnamSIPBridge

        return AnamSIPBridge
    raise AttributeError(f"module 'roomkit.providers.anam' has no attribute {name}")
