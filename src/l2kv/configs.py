from __future__ import annotations

def get_default_skip_layers() -> tuple[int, ...]:
    """Layers left uncompressed by the experiments."""

    return (0, 1)
