## Summary

What changed?

## Validation

```bash
python3 -m unittest discover -s overlap_monitor/tests -p 'test_*.py'
```

## Measurement Semantics

- [ ] No new global CUDA synchronization in runtime recorders.
- [ ] Work-handle upper-bound semantics are preserved where applicable.
- [ ] GPU/profiler claims include validation evidence.

## Compatibility

- [ ] Core modules remain free of Megatron, torch, and transformer_engine imports.
- [ ] Public API or output changes are documented.
