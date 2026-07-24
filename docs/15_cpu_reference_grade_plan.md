# CPU Reference-Grade Colorization Plan

Status: approved direction for the July 29 infrastructure upgrade.

## Goal

Build the strongest practical manga colorization system that can be operated from
a 6 GB RAM CPU VPS without renting a permanent GPU. The target is the supplied
professionally colored page: deliberate object colors, warm lighting, convincing
skin, stable characters, readable materials, preserved ink, and untouched text.

The target is not achievable by saturation or postprocessing alone. The CPU plan
combines compact pretrained perception and color models with MangaHue's region
graph, reference memory, deterministic rendering, candidate ranking, and a small
human correction loop.

## Honest Product Contract

MangaHue will expose two CPU modes:

1. `preview`: the current compact ONNX path, optimized for low latency.
2. `studio_cpu`: a slower reference-guided path, optimized for quality and allowed
   to spend minutes on a page.

`studio_cpu` must never claim to recover the original artist's unknown colors.
Without a colored reference or user palette, several valid colorings can exist.
The quality target becomes realistic when the user supplies a cover, character
sheet, previously colored page, or a few region corrections.

The target page must remain a private benchmark input unless the user confirms
that it can legally be redistributed. Do not commit copyrighted page images or
third-party model weights to this repository.

## CPU Architecture

```text
source page
  -> page and panel analysis
  -> protected ink, text, bubble, and screentone layers
  -> compact semantic perception
  -> region adjacency graph
  -> character and palette reference matching
  -> base color candidates
  -> deterministic material and lighting renderer
  -> original-resolution ink and tone reconstruction
  -> quality scoring and local repair
  -> best candidate
  -> chapter memory and Telegram delivery
```

All heavyweight stages run sequentially. Models are loaded on demand and unloaded
before the next memory-heavy stage. Only one `studio_cpu` page may execute at a
time on a 6 GB server.

## 6 GB RAM Budget

| Component | Target working set |
| --- | ---: |
| Linux, system services, and safety reserve | 1.2 GB |
| MangaHue bot, queue, and API | 0.4 GB |
| Current ONNX colorizer | 0.8 GB |
| One compact perception model | 0.8 GB |
| Full-resolution image layers and masks | 1.0 GB |
| Candidate compositor and quality metrics | 0.6 GB |
| Required free headroom | 1.2 GB |

Hard requirements:

- [ ] Enforce one quality worker.
- [ ] Load only one large model at a time.
- [ ] Use memory-mapped weights where supported.
- [ ] Use ONNX Runtime CPU with bounded threads.
- [ ] Prefer INT8 weights after measured quality validation.
- [ ] Store intermediate layers on disk, not all in memory.
- [ ] Limit decoded page dimensions before model inference.
- [ ] Render final output in strips or tiles at original resolution.
- [ ] Reject archive bombs and oversized decoded images.
- [ ] Abort cleanly before the kernel out-of-memory killer is triggered.

## Pretrained Model Policy

No model is accepted because its demonstration looks attractive. Every candidate
must pass license, memory, runtime, structure, and quality gates.

### Candidate roles

- Current MangaHue hint ONNX: mandatory fast fallback and one candidate source.
- DDColor tiny: benchmark as an alternate chroma proposal, not as the final manga
  renderer.
- Compact face detection and face parsing: locate face, skin, hair, eyes, and mouth
  masks for local correction.
- Compact image embedding model: match recurring character and reference crops.
- Compact segmentation model: propose object boundaries where connected-component
  and screentone analysis are uncertain.
- Manga-specific reference model: retain behind the existing model gateway for
  future hardware, free notebook sessions, or an optional external worker.

### Admission checklist

- [ ] Record repository, exact revision, weight source, and SHA-256.
- [ ] Record code license and weight license separately.
- [ ] Confirm commercial use before production deployment.
- [ ] Reject pickle-based weights when a safer format is available.
- [ ] Scan model archives before extraction.
- [ ] Convert to ONNX or SafeTensors where technically possible.
- [ ] Benchmark FP32, FP16 where supported, dynamic INT8, and static INT8.
- [ ] Measure peak RSS from a cold process.
- [ ] Measure warm page latency.
- [ ] Test portrait, landscape, dense screentone, and multi-panel pages.
- [ ] Test faces, hands, white clothing, dark clothing, and backgrounds.
- [ ] Reject models that alter dialogue, faces, anatomy, or panel geometry.
- [ ] Reject models that exceed the 4.8 GB application memory ceiling.
- [ ] Keep the current backend available for immediate rollback.

## Phase 0: Ground-Truth Benchmark

- [ ] Create a private benchmark directory outside Git.
- [ ] Add at least 30 legally usable grayscale and colored page pairs.
- [ ] Include the attached visual target as a local comparison only.
- [ ] Align every grayscale and colored pair.
- [ ] Add page metadata: series, chapter, scene, characters, and lighting.
- [ ] Mark speech bubbles, text, faces, skin, hair, clothes, and backgrounds.
- [ ] Separate easy, medium, and hard pages.
- [ ] Create a held-out set that is never used for tuning.
- [ ] Record the current production output for every page.
- [ ] Record runtime and peak memory for every stage.
- [ ] Generate side-by-side and difference reports.

Baseline measurements from the supplied comparison:

- Professional target vivid-color coverage: approximately 37.1%.
- Previous MangaHue output vivid-color coverage: approximately 5.8%.
- Current screentone-safe build vivid-color coverage: approximately 12.6%.

These figures are diagnostics, not the only success metric. Increasing saturation
without semantic correctness is a failure.

## Phase 1: Protected-Layer Decomposition

- [ ] Produce immutable original ink and panel-border masks.
- [ ] Produce high-confidence text and speech-bubble masks.
- [ ] Separate screentone frequency from base luminance.
- [ ] Detect flat white regions that are objects, not speech bubbles.
- [ ] Detect panel-local backgrounds.
- [ ] Detect open contours and likely leakage points.
- [ ] Build multi-scale connected components.
- [ ] Store masks in a versioned page-layer manifest.
- [ ] Add visual debug exports for every layer.
- [ ] Add regression fixtures for thin ink and dense screentones.
- [ ] Require exact text restoration before accepting a result.

## Phase 2: Semantic Region Graph

- [ ] Represent each paintable component as a stable region ID.
- [ ] Store area, centroid, boundary, tone histogram, and texture descriptors.
- [ ] Store neighboring regions and shared boundary strength.
- [ ] Store panel membership and depth hints.
- [ ] Add semantic probabilities for skin, hair, eyes, mouth, clothing,
      background, furniture, effects, and unknown.
- [ ] Fuse classical geometry with compact-model predictions.
- [ ] Keep uncertainty instead of forcing low-confidence labels.
- [ ] Merge fragmented regions using boundary and texture evidence.
- [ ] Split regions that cross strong semantic boundaries.
- [ ] Prevent colors from crossing panel borders or protected layers.
- [ ] Serialize the graph for retries and manual correction.
- [ ] Add deterministic graph hashing for cache reuse.

## Phase 3: Character and Reference Memory

- [ ] Accept colored covers, character sheets, and earlier colored pages.
- [ ] Extract reference crops without modifying the originals.
- [ ] Build face and appearance embeddings.
- [ ] Match recurring characters with confidence scores.
- [ ] Store canonical skin, hair, eye, clothing, and accessory swatches.
- [ ] Store alternate outfits separately.
- [ ] Store palette provenance and user confirmation.
- [ ] Transfer colors only above a confidence threshold.
- [ ] Require review when two characters are visually ambiguous.
- [ ] Support point correspondences for difficult reference transfer.
- [ ] Preserve chapter-level palette locks.
- [ ] Learn from accepted user corrections.
- [ ] Never learn from automatically generated output without approval.

## Phase 4: Palette Director

- [ ] Build scene palettes from character locks and reference images.
- [ ] Choose warm, neutral, cool, night, sunset, or custom illumination.
- [ ] Reserve neutral white for speech bubbles and intentional white materials.
- [ ] Maintain perceptual distance between adjacent regions.
- [ ] Use plausible skin ranges without erasing diverse skin tones.
- [ ] Generate coordinated shadow and highlight ramps per material.
- [ ] Preserve black hair and dark fabric chroma instead of collapsing to gray.
- [ ] Generate three to five controlled candidates, not random color noise.
- [ ] Record every palette decision in the page report.
- [ ] Allow one-click palette replacement without rerunning perception.
- [ ] Keep character colors stable across panels and pages.

## Phase 5: CPU Base Color Ensemble

- [ ] Run the compact hint ONNX candidate.
- [ ] Run DDColor tiny only if it passes the admission gates.
- [ ] Generate a graph-fill candidate from semantic locks.
- [ ] Generate reference-transfer candidates where references exist.
- [ ] Normalize all candidates into a common linear-light color space.
- [ ] Extract candidate chroma separately from original luminance.
- [ ] Reject pixels that violate protected layers.
- [ ] Blend candidates per region, not with one global opacity.
- [ ] Prefer deterministic graph fill in high-confidence locked regions.
- [ ] Prefer model proposals in uncertain organic regions.
- [ ] Keep candidate generation resumable and cacheable.
- [ ] Unload each model before the next heavyweight stage.

## Phase 6: Material and Lighting Renderer

This stage closes part of the gap between flat tinting and the supplied target.
It must enhance color without redrawing manga content.

- [ ] Estimate panel-local light direction from grayscale gradients.
- [ ] Convert screentones into controlled luminance modulation.
- [ ] Build soft ambient shadows from region adjacency.
- [ ] Build contact shadows near strong occluding boundaries.
- [ ] Add cel-shaded shadow bands where the source supports them.
- [ ] Add restrained skin warmth and cheek variation.
- [ ] Add hair color variation while preserving black ink masses.
- [ ] Add eye iris color only inside high-confidence eye masks.
- [ ] Add fabric-specific contrast and color response.
- [ ] Preserve existing plaid, hatch, and texture patterns.
- [ ] Add diffuse background illumination without coloring bubbles.
- [ ] Add highlights only when supported by source luminance or a style rule.
- [ ] Use edge-aware smoothing inside regions.
- [ ] Prevent halos around ink.
- [ ] Process all calculations in linear light where appropriate.
- [ ] Gamut-map the result without clipping skin or highlights.
- [ ] Expose lighting strength as a bounded style parameter.

## Phase 7: Original-Resolution Reconstruction

- [ ] Reconstruct chroma at original page resolution.
- [ ] Restore original luminance detail without desaturating chroma.
- [ ] Restore exact ink and panel borders.
- [ ] Restore speech bubbles and dialogue.
- [ ] Restore screentones as luminance texture.
- [ ] Remove edge color fringes.
- [ ] Remove isolated color speckles.
- [ ] Repair tiny uncolored holes.
- [ ] Preserve antialiasing at text and ink boundaries.
- [ ] Export lossless PNG before optional delivery compression.

## Phase 8: Quality Controller

- [ ] Score ink preservation.
- [ ] Score text and bubble preservation.
- [ ] Score region leakage.
- [ ] Score grayscale residue outside intentional neutral regions.
- [ ] Score vivid-color coverage without rewarding oversaturation.
- [ ] Score skin plausibility and within-character consistency.
- [ ] Score character palette consistency.
- [ ] Score adjacent-region distinguishability.
- [ ] Score reference adherence when a reference exists.
- [ ] Score chapter-level palette drift.
- [ ] Detect damaged faces and eyes.
- [ ] Detect implausible color islands.
- [ ] Detect crushed blacks and clipped highlights.
- [ ] Rank candidates with hard safety gates before aesthetic scores.
- [ ] Repair only failing regions.
- [ ] Stop retries after a strict CPU budget.
- [ ] Preserve all scores and decisions in the job manifest.

## Phase 9: Correction Loop

- [ ] Add brush, fill, erase, undo, and mask editing to Studio.
- [ ] Add click-to-select region editing.
- [ ] Add semantic labels to selected regions.
- [ ] Add before-and-after comparison.
- [ ] Add character assignment for selected regions.
- [ ] Save corrections as vector or region operations.
- [ ] Re-render corrected pages without rerunning every model.
- [ ] Propagate confirmed locks to later pages.
- [ ] Add correction confidence and provenance.
- [ ] Export corrected pairs for future training.

The correction loop is part of the CPU solution, not an admission of failure. One
confirmed hair, eye, skin, and outfit palette can guide an entire chapter.

## Phase 10: Telegram Product Flow

- [ ] Add `/preview` and `/studio` modes.
- [ ] Explain that Studio mode is slower before accepting a job.
- [ ] Accept optional reference images before chapter processing.
- [ ] Accept a chapter link, ZIP, CBZ, PDF, or page images.
- [ ] Validate ownership and supported sources.
- [ ] Queue one Studio page at a time.
- [ ] Persist progress before every expensive stage.
- [ ] Send each accepted page as soon as it completes.
- [ ] Show current stage without promising an exact completion time.
- [ ] Support cancellation and resume.
- [ ] Retry after process restart.
- [ ] Return a correction link for uncertain pages.
- [ ] Package final PNG, ZIP, and CBZ outputs.
- [ ] Retain outputs according to a configured policy.

## Phase 11: Operations

- [ ] Add a durable SQLite job queue for the single-server edition.
- [ ] Use transactional job claiming.
- [ ] Add per-user quotas and maximum chapter sizes.
- [ ] Add watchdogs for stalled model workers.
- [ ] Enforce process RSS and wall-time limits.
- [ ] Add disk-space admission checks.
- [ ] Add structured logs without secrets.
- [ ] Add daily database and configuration backups.
- [ ] Add model and application rollback.
- [ ] Add deployment health checks.
- [ ] Add failure-injection tests.
- [ ] Verify unrelated VPS services before and after deployment.

## July 29 Execution Order

1. Secure and baseline the new VPS.
2. Deploy the existing `preview` engine unchanged.
3. Build and run the private benchmark.
4. Measure pretrained CPU candidates one at a time.
5. Select only models that pass license and memory gates.
6. Implement protected-layer manifests and the semantic region graph.
7. Implement reference and character memory.
8. Implement graph fill, material rendering, and candidate ranking.
9. Add Studio correction tools.
10. Add `/studio` Telegram jobs and resumability.
11. Tune against development pages.
12. Run the untouched held-out benchmark.
13. Release only after every hard safety gate passes.

## Quality Gates

Hard gates:

- [ ] Dialogue pixels remain unchanged within tolerance.
- [ ] Panel geometry remains unchanged.
- [ ] Ink preservation score is at least 0.97.
- [ ] Protected-region violations are below 0.5%.
- [ ] No model-generated anatomy or rewritten artwork is accepted.
- [ ] Peak application RSS remains below 4.8 GB.
- [ ] A killed worker can resume without losing the chapter.

Target gates:

- [ ] Vivid-color coverage approaches 25% to 35% on appropriate target pages.
- [ ] Skin, hair, and clothing are semantically distinct.
- [ ] Main-character palette drift is below the configured perceptual threshold.
- [ ] Backgrounds receive intentional colors rather than a global tint.
- [ ] At least 80% of private benchmark pages are preferred over the current
      production output in blind human comparison.
- [ ] A corrected character palette is reused reliably across the chapter.

The 37.1% target-page vivid coverage is a reference, not a mandatory value for
every manga page. Quiet, night, monochromatic, or intentionally neutral scenes
must not be forced to that saturation.

## Definition of Done

The CPU plan is complete only when:

- The system runs on the measured 6 GB VPS without swap thrashing.
- Preview and Studio modes share the same protected-layer and memory contracts.
- Studio mode can use at least one legally deployable compact pretrained model.
- Reference images control recurring character colors.
- Deterministic rendering adds material-aware color and lighting without changing
  the drawing.
- Automatic ranking rejects unsafe candidates.
- Corrections can be propagated through a chapter.
- The private held-out benchmark beats the current production engine.
- Telegram jobs survive restart and deliver pages incrementally.
- Every deployed source change, config, model manifest, and migration is committed
  to GitHub.

## Deferred Upgrade Path

If free notebook GPU time or affordable GPU hardware becomes available later, a
reference-guided diffusion backend can be added behind the existing inference
interface. It will replace only candidate generation. The region graph, references,
material renderer, safety gates, corrections, Telegram flow, and chapter memory
remain useful and unchanged.
