"""Provide missing experimental torch_xla modules for tests."""

import sys
from types import ModuleType

# If torch_xla lacks splash_attention, provide a minimal stub
try:
    pass
except Exception:  # noqa: BLE001
    experimental = ModuleType('torch_xla.experimental')
    splash = ModuleType('torch_xla.experimental.splash_attention')
    def splash_attention(*args, **kwargs):  # noqa: D401,D417
        """Placeholder for unavailable kernel."""
        raise NotImplementedError(
            "Splash attention kernel is not available in this environment"
        )
    class SplashAttentionConfig:  # noqa: D401,D417
        def __init__(self, *args, **kwargs):
            self.mesh = None
            self.qkv_partition_spec = None
            self.segment_ids_partition_spec = None
        def to_json(self):
            return "{}"
    splash.splash_attention = splash_attention
    splash.SplashAttentionConfig = SplashAttentionConfig
    experimental.splash_attention = splash
    sys.modules['torch_xla.experimental'] = experimental
    sys.modules['torch_xla.experimental.splash_attention'] = splash
