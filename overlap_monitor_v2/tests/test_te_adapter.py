from __future__ import annotations

import unittest

from overlap_monitor_v2.core.events import EventType
from overlap_monitor_v2.te_adapter import TEAdapter


class TEAdapterTests(unittest.TestCase):
    def test_classifies_te_gemm(self):
        info = TEAdapter().classify_kernel("transformer_engine::gemm_fp8")
        self.assertEqual(info.event_type, EventType.GEMM)
        self.assertEqual(info.precision, "fp8")
        self.assertTrue(info.is_transformer_engine)

    def test_classifies_attention(self):
        info = TEAdapter().classify_kernel("nvte_fused_attn_bf16")
        self.assertEqual(info.event_type, EventType.ATTENTION)
        self.assertEqual(info.precision, "bf16")


if __name__ == "__main__":
    unittest.main()
