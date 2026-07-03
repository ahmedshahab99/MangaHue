# Pipeline

1. Load and validate the page.
2. Resize to the configured maximum dimension.
3. Normalize grayscale line art.
4. Remove small speckles.
5. Create line, speech bubble, text, and protected masks.
6. Estimate panels and split the page into overlapping tiles.
7. Load style, project memory, character profiles, and user hints.
8. Rasterize hint maps and masks.
9. Run the selected backend per tile.
10. Merge tiles into the full page.
11. Apply palette correction.
12. Clean speech bubbles.
13. Reduce color near ink boundaries.
14. Restore original manga ink.
15. Save final output, debug images, and JSON reports.

