from __future__ import annotations

import unittest

from overlap_monitor.core.events import EventType
from overlap_monitor.te_adapter import TEAdapter


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

    def test_does_not_treat_word_suffix_as_te_marker(self):
        info = TEAdapter().classify_kernel("update_tensor_kernel")
        self.assertEqual(info.event_type, EventType.UNKNOWN)
        self.assertFalse(info.is_transformer_engine)


if __name__ == "__main__":
    unittest.main()
