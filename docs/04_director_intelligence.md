# Director Intelligence

MangaHue is designed to behave like a manga color director, not a filter. The current engine records:

- Page classification: portrait page, spread/panel, or webtoon strip.
- Ink profile: adaptive threshold, line density, connected ink components, large ink regions, and removed speckles.
- Texture profile: screen-tone score and page complexity score.
- Protected zones: speech bubbles, detected text strokes, and protected coverage.
- Hint behavior: style hints, character-memory hints, user hints, and hint coverage.
- Output gates: line preservation, bubble cleanliness, protected-region violation score, hint adherence, and sharpness estimate.

Debug mode writes `director_overlay.png`, a compact visual map:

- Black: restored ink.
- Green: speech bubbles.
- Blue: text strokes.
- Amber: tone-like texture.
- Hint colors: active guidance regions.

Character memory labels support:

- `hero.hair`
- `hero:skin`
- `hair` when one character profile is active

Memory guidance is applied softly before user hints, so the user's direct instruction remains the highest-priority color signal.
