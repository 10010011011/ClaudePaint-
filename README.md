# ClaudePaint (no-slop-painter)

Deterministic painting tools. The LLM only directs: it reads numbered region maps and writes small JSON decisions. It never issues per-stroke calls.

## Status
- Step 1 (done): perceptual flatten + numbered region map
- Planned: 2 edge blend (lost/found edges by dE), 3 grid critic, 4 stroke texture, 5 pixel mode, 6 skill + MCP wrapper
