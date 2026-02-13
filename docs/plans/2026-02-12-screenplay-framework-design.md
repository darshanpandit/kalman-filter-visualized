# Screenplay Framework Design

## Problem

Scene files interleave creative content (narration, visual beats) with implementation (Manim code, filter setup). This makes it hard to iterate on the storytelling without touching code. We need a higher-level format for authoring scenes.

## Solution

A Fountain-inspired DSL (`.screenplay` files) with two Claude Code skills:

1. **`screenplay-writer`** — Brainstorms and writes a screenplay with the user
2. **`screenplay-to-manim`** — Generates a ManimCE scene file from a screenplay

The DSL serves as structured spec for LLM-assisted code generation (not compiled or interpreted directly).

## DSL Syntax (`.screenplay`)

```
PART N: Title
SCENE NN: Title — "Subtitle"

REFERENCES:
  - Author (Year) — description

DATA:
  source: eth/pedestrian #171
  # or: generator/mode_switching(n_steps=80, seed=42)
  measurement_noise: 0.6
  max_steps: 60
  filter: KalmanFilter(Q=0.1*I4, R=0.36*I2)
  # or multi-filter:
  filters:
    - KalmanFilter(name="CV", F=F_cv, Q=0.01*I4)
    - KalmanFilter(name="CT", F=F_ct, Q=0.05*I4)
  run: IMMFilter(filters=[CV, CT], Pi=[[0.95,0.05],[0.05,0.95]])

= BEAT 1: Description =

TITLE: "Text" (position)
SHOW: mobject_or_description
ANIMATE: animation description
TEXT: "On-screen text" (position, color)
PAUSE: short | medium | long
NOTE: "Citation text"
FADEOUT: all | specific_mobject

> Voiceover narration text goes in blockquotes.
> Can span multiple lines.
```

### Syntax Elements

| Element | Syntax | Purpose |
|---------|--------|---------|
| Headers | `PART N:` / `SCENE NN:` | Video and scene identity |
| References | `REFERENCES:` block | Papers/sources cited |
| Data | `DATA:` block | Data source, filters, parameters |
| Beat | `= BEAT N: desc =` | Narrative segment boundary |
| Voiceover | `> lines` | Narrator script |
| Visual | `SHOW:` | Create/display mobject |
| Animation | `ANIMATE:` | Animate over time |
| Title/Text | `TITLE:` / `TEXT:` | On-screen text |
| Pause | `PAUSE:` | Timing break |
| Note | `NOTE:` | observation_note citation |
| Transition | `FADEOUT:` | Scene cleanup |
| Style refs | `$COLOR_NAME` | Reference style.py constants |

### Style Variable References

Use `$COLOR_NAME` to reference constants from `kalman_manim/style.py`:
- `$COLOR_PREDICTION`, `$COLOR_MEASUREMENT`, `$COLOR_POSTERIOR`
- `$COLOR_FILTER_KF`, `$COLOR_FILTER_EKF`, `$COLOR_FILTER_UKF`, `$COLOR_FILTER_PF`
- `$COLOR_FILTER_TF`, `$COLOR_FILTER_KALMANNET`, `$COLOR_FILTER_IMM`, `$COLOR_FILTER_PHD`
- `$COLOR_SSM`, `$COLOR_SOCIAL`, `$COLOR_HIGHLIGHT`, `$COLOR_TEXT`

### Data Source Syntax

Real data:
```
source: eth/pedestrian #171
source: ucy/zara1/pedestrian #42
```

Generators:
```
source: generator/pedestrian(n_steps=100, seed=42)
source: generator/mode_switching(n_steps=80, seed=42)
source: generator/lorenz(n_steps=500, dt=0.01)
source: generator/multi_target(n_targets=5, n_steps=60)
```

## Skill 1: `screenplay-writer`

### Purpose
Brainstorm a video scene with the user and produce a `.screenplay` file.

### Workflow
1. Understand the teaching goal (what should the viewer learn?)
2. Propose visual metaphors and narrative arc
3. Draft beats with voiceover + visuals
4. Write to `screenplays/partN/sceneNN_name.screenplay`

### Knowledge Required
- DSL syntax (this document)
- Available mobjects: GaussianEllipse, StateSpace, PedestrianPath, JacobianTangent, SigmaPointCloud, ParticleCloud, ComparisonTable, TransformerDiagram, KalmanNetDiagram, SSMDiagram, AttentionHeatmap, MultiTrackPlot, IntensityHeatmap, PredictionFan, ModeProbabilityBar, RSSMDiagram, GraphicalModel, VectorFieldPlot, PhaseSpacePlot, GrandTaxonomyDiagram, RMSELineChart, FilterBarChart, ErrorHistogram
- Available generators: pedestrian, linear, nonlinear, sharp_turn, multimodal, multi_target, mode_switching, lorenz, pendulum
- Available filters: KalmanFilter, EKF, UKF, ParticleFilter, IMMFilter, GMPHDFilter
- Available datasets: ETH (eth, hotel), UCY (univ, zara1, zara2)
- Pedagogical arc: hook -> build intuition -> formalize -> demo -> takeaway
- Series structure (Parts 1-9 content)

### Validation
- Every beat has voiceover (at least one `>` line)
- Every beat has at least one visual (`SHOW:`, `TITLE:`, `TEXT:`, or `ANIMATE:`)
- DATA block is complete if scene uses data
- Scene has at least 3 beats
- Scene ends with `FADEOUT: all`

## Skill 2: `screenplay-to-manim`

### Purpose
Read a `.screenplay` file and generate a complete ManimCE scene `.py` file.

### Workflow
1. Parse the screenplay
2. Generate scene file following project conventions
3. Apply layout constraints
4. Run self-validation checklist

### Project Conventions (mandatory)
- Base classes: `VoiceoverScene, MovingCameraScene`
- Speech service: `GTTSService()`
- Background: `self.camera.background_color = BG_COLOR`
- Boilerplate: `sys.path.insert(0, ...)`, `from __future__ import annotations`
- Style: import from `kalman_manim.style`
- Text: use `Text()` never `MathTex()` (no LaTeX available)
- Rectangles: use `RoundedRectangle` not `Rectangle(corner_radius=...)`
- Citations: use `make_observation_note()`
- Scene cleanup: `self.play(*[FadeOut(mob) for mob in self.mobjects])`

### Layout Constraints (from ManimAgentPrompts lessons)

**Frame safety:**
- Safe frame: 16:9, content within `[-6, 6] x [-3.4, 3.4]`
- Title band: top 1.0 units (y > 2.4)
- Diagram band: middle (y between -2.0 and 2.4)
- Note/citation band: bottom (y < -2.0)

**Text safety:**
- No overlapping text — ever
- Text blocks max 3 lines, 4-6 words per line
- Boundary check: `abs(label.get_x()) <= 5`
- Fallback: if RIGHT placement exceeds bounds, place DOWN instead
- Font sizes: titles 36-48, headings 28-32, body 22-24, small 16-18

**Grouping and scaling:**
- Multi-part groups: `VGroup` + `arrange()` + `scale_to_fit_width(12)` + `move_to(ORIGIN)`
- Always use relative positioning: `next_to`, `to_edge`, `arrange`
- No raw coordinates beyond safe frame bounds
- Every `VGroup` with 3+ items must have explicit arrangement

**Animation timing:**
- Map `PAUSE: short` -> `PAUSE_SHORT`, `medium` -> `PAUSE_MEDIUM`, `long` -> `PAUSE_LONG`
- Map animation speeds to `FAST_ANIM`, `NORMAL_ANIM`, `SLOW_ANIM`
- Title animations: `Write()` or `FadeIn(shift=DOWN*0.3)`
- Content animations: `FadeIn()`, `Create()`, `LaggedStart()`
- Exit animations: `FadeOut()`

### Self-Validation Checklist
1. All mobjects use relative positioning (`next_to`, `to_edge`, `arrange`)
2. No raw coordinates beyond safe frame
3. All text blocks <= 3 lines
4. Every VGroup with 3+ items has explicit arrangement
5. No `MathTex` usage
6. No `Rectangle(corner_radius=...)` usage
7. Every voiceover block has matching animations
8. Scene ends with FadeOut(all)
9. `from __future__ import annotations` present
10. `sys.path.insert` boilerplate present
11. Docstring with scene description present
