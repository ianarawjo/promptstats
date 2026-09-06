"""Color constants shared across the vis/ plotting modules.

Each plotting module defines its own ``_PALETTE`` dict (different plots need
different keys), but a few colors -- axis text, secondary/tick text, grid
lines -- are the same everywhere. Defined once here so they stay in sync.
"""

TEXT = "#2D333B"            # dark slate — axis labels
TEXT_SECONDARY = "#6B7280"  # muted gray — secondary/tick text
GRID = "#EEF1F4"            # very light gray — grid lines
