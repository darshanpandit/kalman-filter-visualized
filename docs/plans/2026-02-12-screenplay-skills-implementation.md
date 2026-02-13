# Screenplay Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create two Claude Code skills — `screenplay-writer` (brainstorm + write `.screenplay` files) and `screenplay-to-manim` (generate ManimCE scene code from screenplays) — plus a DSL reference and sample screenplay.

**Architecture:** Skills live in `~/.claude/skills/` as SKILL.md files. The DSL reference doc lives in the project at `docs/screenplay-dsl.md`. A `screenplays/` directory holds `.screenplay` files organized by part. No runtime code — the skills are pure prompts that guide Claude's generation.

**Tech Stack:** Claude Code skills (Markdown), Fountain-inspired DSL (custom format)

---

### Task 1: Create DSL reference document

**Files:**
- Create: `docs/screenplay-dsl.md`

**Step 1: Write the DSL reference**

This is the canonical syntax reference that both skills will point to. It defines every construct in the `.screenplay` format.

```markdown
# Screenplay DSL Reference

## File Format

Files use `.screenplay` extension. Stored in `screenplays/partN/sceneNN_name.screenplay`.

## Header Block

```
PART N: Part Title
SCENE NN: Scene Title — "Subtitle"
```

Required. Identifies which video and scene this is.

## References Block (optional)

```
REFERENCES:
  - Author (Year) — description
  - Author (Year) — description
```

Papers or sources cited. Generates `make_observation_note()` at scene end.

## Data Block

```
DATA:
  source: eth/pedestrian #171
  measurement_noise: 0.6
  max_steps: 60
  filter: KalmanFilter(Q=0.1*I4, R=0.36*I2)
```

Or for multi-filter scenes:

```
DATA:
  source: generator/mode_switching(n_steps=80, seed=42)
  filters:
    - KalmanFilter(name="CV", F=F_cv, Q=0.01*I4, R=0.25*I2)
    - KalmanFilter(name="CT", F=F_ct, Q=0.05*I4, R=0.25*I2)
  run: IMMFilter(filters=[CV, CT], Pi=[[0.95,0.05],[0.05,0.95]])
```

### Data Sources

Real data:
- `eth/pedestrian #N` — ETH dataset (eth or hotel sequence)
- `hotel/pedestrian #N` — ETH hotel sequence
- `ucy/SCENE/pedestrian #N` — UCY dataset (univ, zara1, zara2)

Generators:
- `generator/pedestrian(n_steps=N, seed=S)`
- `generator/linear(n_steps=N, seed=S)`
- `generator/nonlinear(n_steps=N, seed=S)`
- `generator/sharp_turn(n_steps=N, seed=S)`
- `generator/multimodal(seed=S)`
- `generator/multi_target(n_targets=N, n_steps=N, seed=S)`
- `generator/mode_switching(n_steps=N, seed=S)`
- `generator/lorenz(n_steps=N, dt=D)`
- `generator/pendulum(n_steps=N, dt=D)`

### Filters

- `KalmanFilter(F=..., Q=..., R=...)`
- `EKF(f=..., h=..., F_jac=..., H_jac=..., Q=..., R=...)`
- `UKF(f=..., h=..., Q=..., R=..., alpha=..., beta=..., kappa=...)`
- `ParticleFilter(f=..., h=..., Q=..., R=..., N=...)`
- `IMMFilter(filters=[...], Pi=...)`
- `GMPHDFilter(F=..., H=..., Q=..., R=..., p_s=..., p_d=...)`

Shorthand: `I4` = `np.eye(4)`, `I2` = `np.eye(2)`

## Beat Block

```
= BEAT N: Short Description =
```

A narrative segment. Each beat pairs voiceover with visuals. A scene should have 3-8 beats.

## Voiceover

```
> Narrator text goes in blockquotes.
> Can span multiple lines.
> This becomes the GTTS voiceover text.
```

Every beat must have at least one voiceover block.

## Visual Directives

### TITLE / TEXT
```
TITLE: "Text" (top)
TEXT: "Text" (bottom, $COLOR_HIGHLIGHT)
```

Position: `top`, `bottom`, `center`. Optional color as `$COLOR_NAME`.

### SHOW
```
SHOW: measurement_dots appearing in batches of 8
SHOW: true_path as dashed line
SHOW: axes(x_range=[0,10], y_range=[0,5], size=7x4)
SHOW: ModeProbabilityBar(models=["CV","CT"], width=5.0) at bottom
SHOW: ComparisonTable(headers=[...], rows=[...])
```

Describes what mobject to create and display. Can reference project mobjects by name or describe custom visuals.

### ANIMATE
```
ANIMATE: step through model_probabilities at steps [10, 25, 45, 65]
ANIMATE: filter running on measurements, showing ellipses growing/shrinking
ANIMATE: dots fading from $COLOR_MEASUREMENT to $COLOR_POSTERIOR
```

Describes dynamic behavior over time.

### PAUSE
```
PAUSE: short    # PAUSE_SHORT
PAUSE: medium   # PAUSE_MEDIUM
PAUSE: long     # PAUSE_LONG
```

### NOTE
```
NOTE: "Author (Year): Key finding.\nSecond line."
```

Creates `make_observation_note()` citation box, typically near scene end.

### FADEOUT
```
FADEOUT: all
FADEOUT: title, subtitle
```

Scene cleanup. `FADEOUT: all` is required as the last directive.

## Style References

Use `$COLOR_NAME` to reference `kalman_manim/style.py` constants:

| Variable | Color | Use |
|----------|-------|-----|
| `$COLOR_PREDICTION` | Swiss red #e63946 | Predictions, KF |
| `$COLOR_MEASUREMENT` | Royal blue #457b9d | Measurements |
| `$COLOR_POSTERIOR` | Gold #f4a261 | Posteriors, estimates |
| `$COLOR_PROCESS_NOISE` | Teal #2a9d8f | Process noise, UKF |
| `$COLOR_HIGHLIGHT` | Gold #f4a261 | Emphasis |
| `$COLOR_TEXT` | Cream #f1faee | Default text |
| `$COLOR_FILTER_KF` | Red #e63946 | KF in comparisons |
| `$COLOR_FILTER_EKF` | Orange #e07c42 | EKF |
| `$COLOR_FILTER_UKF` | Teal #2a9d8f | UKF |
| `$COLOR_FILTER_PF` | Gold #f4a261 | PF |
| `$COLOR_FILTER_TF` | Violet #9b59b6 | Transformers |
| `$COLOR_FILTER_KALMANNET` | Bright red #e74c3c | KalmanNet |
| `$COLOR_FILTER_IMM` | Purple #8e44ad | IMM |
| `$COLOR_FILTER_PHD` | Green #27ae60 | PHD |
| `$COLOR_SSM` | Emerald #1abc9c | State-space models |
| `$COLOR_SOCIAL` | Amber #f39c12 | Social prediction |

## Complete Example

```
PART 1: Kalman Filters
SCENE 01: Hook — "Where is the pedestrian?"

REFERENCES:
  - Pellegrini et al. (2009) — ETH pedestrian dataset

DATA:
  source: eth/pedestrian #171
  measurement_noise: 0.6
  max_steps: 60
  filter: KalmanFilter(Q=0.1*I4, R=0.36*I2)

= BEAT 1: Noisy measurements =

TITLE: "Where is the pedestrian?" (top)

SHOW: measurement dots appearing in batches of 8, $COLOR_MEASUREMENT

> Where is the pedestrian? These are real coordinates
> from the ETH Zurich pedestrian dataset. Each ping
> gives a position estimate, but look at how noisy
> these measurements are.

PAUSE: medium

= BEAT 2: Reveal truth =

TEXT: "Your phone says you're here... but are you really?" (below title)
SHOW: true path as dashed line (Create animation)

> Your phone says you're somewhere around here, but
> where are you really? Here's the actual path — smooth
> and continuous. But all we get are these scattered,
> error-prone observations.

= BEAT 3: Kalman tease =

TEXT: "Can we recover the true trajectory?" (bottom, $COLOR_HIGHLIGHT)
SHOW: filtered estimate path ($COLOR_POSTERIOR, Create animation)
TEXT: "Yes — with a Kalman Filter." (replace previous, $COLOR_POSTERIOR)

> Can we recover the true trajectory from noisy
> measurements? Yes — with a Kalman Filter. By
> intelligently combining predictions with measurements,
> we can reconstruct something remarkably close to the truth.

PAUSE: long

> Over the next few videos, I'll show you exactly how this works.

FADEOUT: all
```
```

**Step 2: Commit**

```bash
git add docs/screenplay-dsl.md
git commit -m "Add screenplay DSL reference document"
```

---

### Task 2: Create `screenplay-writer` skill

**Files:**
- Create: `~/.claude/skills/screenplay-writer/SKILL.md`

**Step 1: Write the skill**

```markdown
---
name: screenplay-writer
description: Use when creating or brainstorming a new ManimCE video scene, writing narration scripts, or planning visual beats for the Kalman Filter video series
---

# Writing Screenplays

## Overview

Brainstorm and write `.screenplay` files for ManimCE video scenes. The screenplay is a Fountain-inspired DSL that specifies narrative, visuals, and data — a complete scene spec in readable format.

## When to Use

- User wants to create a new scene
- User wants to plan what a video should show/say
- User wants to iterate on narration or visual flow
- User says "write a script", "plan a scene", "screenplay"

## Process

1. **Understand the teaching goal** — What should the viewer learn?
2. **Identify the data** — Real dataset or generator? Which filter?
3. **Draft the narrative arc** — Hook, build intuition, formalize, demo, takeaway
4. **Write beats** — Each beat = voiceover + visuals, 3-8 beats per scene
5. **Save** to `screenplays/partN/sceneNN_name.screenplay`

## DSL Quick Reference

| Element | Syntax |
|---------|--------|
| Header | `PART N:` / `SCENE NN:` |
| References | `REFERENCES:` block |
| Data | `DATA:` block with source, filter params |
| Beat | `= BEAT N: desc =` |
| Voiceover | `> narrator text` |
| Visual | `SHOW:` / `ANIMATE:` |
| Text | `TITLE:` / `TEXT:` with position and color |
| Timing | `PAUSE: short/medium/long` |
| Citation | `NOTE: "text"` |
| Cleanup | `FADEOUT: all` |
| Colors | `$COLOR_NAME` refs |

Full syntax: `docs/screenplay-dsl.md`

## Available Project Assets

**Mobjects:** GaussianEllipse, StateSpace, PedestrianPath, JacobianTangent, SigmaPointCloud, ParticleCloud, ComparisonTable, AttentionHeatmap, MultiTrackPlot, IntensityHeatmap, PredictionFan, ModeProbabilityBar, RSSMDiagram, GraphicalModel, VectorFieldPlot, PhaseSpacePlot, GrandTaxonomyDiagram, RMSELineChart, FilterBarChart, ErrorHistogram, TransformerDiagram, KalmanNetDiagram, SSMDiagram

**Generators:** pedestrian, linear, nonlinear, sharp_turn, multimodal, multi_target, mode_switching, lorenz, pendulum

**Filters:** KalmanFilter, EKF, UKF, ParticleFilter, IMMFilter, GMPHDFilter

**Datasets:** ETH (eth, hotel), UCY (univ, zara1, zara2)

## Validation Checklist

Before saving, verify:
- [ ] Every beat has voiceover (`>` lines)
- [ ] Every beat has at least one visual directive
- [ ] DATA block is complete (source + filter if needed)
- [ ] Scene has 3-8 beats
- [ ] Voiceover is conversational, not academic
- [ ] Last directive is `FADEOUT: all`
- [ ] Total voiceover ~2-4 minutes when read aloud

## Pedagogical Patterns

**Hook scene:** Mystery/question → reveal → tease solution
**Concept scene:** Intuition → formalize → "why it works"
**Demo scene:** Setup → run filter → interpret results
**Comparison scene:** Side-by-side → metrics → decision guide
**Capstone scene:** Recap → taxonomy → open questions

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Too much text on screen | Max 3 lines per TEXT, 4-6 words per line |
| No visual for a beat | Every beat needs SHOW, ANIMATE, or TEXT |
| Academic narration | Write conversationally — "look at how..." not "we observe that..." |
| Missing data spec | Always specify source and filter parameters |
| Too many beats | 3-8 beats. If more, split into two scenes |
```

**Step 2: Commit**

```bash
git add ~/.claude/skills/screenplay-writer/SKILL.md
git commit -m "Add screenplay-writer skill"
```

---

### Task 3: Create `screenplay-to-manim` skill

**Files:**
- Create: `~/.claude/skills/screenplay-to-manim/SKILL.md`

**Step 1: Write the skill**

```markdown
---
name: screenplay-to-manim
description: Use when converting a .screenplay file to ManimCE Python code, generating scene files from screenplays, or when user says render or generate code from a screenplay
---

# Generating Manim from Screenplays

## Overview

Read a `.screenplay` file and generate a complete ManimCE scene `.py` file. Apply layout constraints to ensure frame-safe, non-overlapping output.

## When to Use

- User has a `.screenplay` file and wants scene code
- User says "generate", "render", "convert" a screenplay
- After `screenplay-writer` produces a `.screenplay`

## Process

1. **Read** the `.screenplay` file
2. **Read** `docs/screenplay-dsl.md` for syntax reference
3. **Read** existing scene files for patterns (e.g., `part1_kalman_filter/scene01_hook.py`)
4. **Generate** complete scene `.py` file
5. **Self-validate** against layout checklist
6. **Save** to `partN_name/sceneNN_name.py`

## Mandatory Boilerplate

Every generated scene MUST include:

```python
from __future__ import annotations

from manim import *
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_manim.style import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class SceneName(VoiceoverScene, MovingCameraScene):
    """Docstring from SCENE header."""

    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = BG_COLOR
        # ... scene content ...
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=NORMAL_ANIM)
```

## DSL-to-Code Mapping

| DSL | Code |
|-----|------|
| `TITLE: "X" (top)` | `Text("X", color=COLOR_TEXT, font_size=TITLE_FONT_SIZE).to_edge(UP, buff=0.3)` |
| `TEXT: "X" (bottom)` | `Text("X", ...).to_edge(DOWN, buff=0.5)` |
| `SHOW: axes(...)` | `Axes(x_range=..., y_range=..., x_length=N, y_length=N)` |
| `PAUSE: short` | `self.wait(PAUSE_SHORT)` |
| `PAUSE: medium` | `self.wait(PAUSE_MEDIUM)` |
| `PAUSE: long` | `self.wait(PAUSE_LONG)` |
| `NOTE: "X"` | `make_observation_note("X")` |
| `FADEOUT: all` | `self.play(*[FadeOut(mob) for mob in self.mobjects])` |
| `> voiceover text` | `with self.voiceover(text="...") as tracker:` |
| `$COLOR_NAME` | Direct constant reference from style.py |

## Data Source Mapping

| DSL | Code |
|-----|------|
| `source: eth/pedestrian #N` | `load_eth_trajectory(sequence="eth", pedestrian_id=N, ...)` |
| `source: hotel/pedestrian #N` | `load_eth_trajectory(sequence="hotel", pedestrian_id=N, ...)` |
| `source: ucy/SCENE/pedestrian #N` | `load_trajectory(dataset="ucy", sequence="SCENE", pedestrian_id=N, ...)` |
| `source: generator/NAME(...)` | `generate_NAME_trajectory(...)` |

## Layout Constraints (MANDATORY)

### Frame Safety
- Safe frame: content within `[-6, 6] x [-3.4, 3.4]`
- Title band: top (y > 2.4), use `.to_edge(UP, buff=0.3)`
- Diagram band: middle
- Note/citation band: bottom (y < -2.0)

### Text Safety
- **NO OVERLAPPING TEXT** — absolute rule
- Max 3 lines per text block, 4-6 words per line
- Always `Text()` never `MathTex()` (no LaTeX available)
- Boundary check: if text might exceed x=5, place DOWN instead of RIGHT
- Font sizes: title 36-48, heading 28-32, body 22-24, small 16-18

### Grouping
- Multi-part groups: `VGroup` + `arrange()` + `scale_to_fit_width(12)`
- Always relative positioning: `next_to`, `to_edge`, `arrange`
- No raw coordinates beyond safe frame
- Every VGroup with 3+ items needs explicit arrangement

### Animation Timing
- Title: `Write()` or `FadeIn(shift=DOWN*0.3)`, `run_time=NORMAL_ANIM`
- Content appear: `FadeIn()`, `run_time=FAST_ANIM` or `NORMAL_ANIM`
- Path drawing: `Create()`, `run_time=SLOW_ANIM`
- Batch appear: `LaggedStart(*[FadeIn(d) for d in batch], lag_ratio=0.15)`
- Exit: `FadeOut()`, `run_time=NORMAL_ANIM` or `FAST_ANIM`

### Known Pitfalls
- Use `RoundedRectangle` not `Rectangle(corner_radius=...)`
- Parallel manim renders corrupt GTTS voiceover JSON cache
- `set_z_index(10)` on titles to keep them above other content

## Self-Validation Checklist

Run mentally before saving generated code:

1. [ ] `from __future__ import annotations` present
2. [ ] `sys.path.insert` boilerplate present
3. [ ] Docstring with scene description
4. [ ] All text uses `Text()` not `MathTex()`
5. [ ] No `Rectangle(corner_radius=...)` — use `RoundedRectangle`
6. [ ] All mobjects use relative positioning
7. [ ] No raw coordinates beyond `[-6, 6] x [-3.4, 3.4]`
8. [ ] All text blocks <= 3 lines
9. [ ] Every VGroup with 3+ items has explicit arrangement
10. [ ] Every voiceover block wraps matching animations
11. [ ] Scene ends with `FadeOut(all)`
12. [ ] All imports are correct and used

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using `MathTex` | Use `Text()` — no LaTeX installed |
| `Rectangle(corner_radius=0.1)` | `RoundedRectangle(corner_radius=0.1)` |
| Text overflows frame | Check `abs(label.get_x()) <= 5`, fallback to DOWN |
| Missing voiceover wrapper | Every beat's animations go inside `with self.voiceover(...)` |
| Hardcoded coordinates | Use `next_to()`, `to_edge()`, `arrange()` |
| Title hidden behind content | Add `.set_z_index(10)` to title |
```

**Step 2: Commit**

```bash
git add ~/.claude/skills/screenplay-to-manim/SKILL.md
git commit -m "Add screenplay-to-manim skill"
```

---

### Task 4: Create screenplays directory and sample screenplay

**Files:**
- Create: `screenplays/part1/scene01_hook.screenplay`

**Step 1: Write the sample screenplay**

Transcribe the existing `part1_kalman_filter/scene01_hook.py` into screenplay format. This validates the DSL can express a real scene.

**Step 2: Create directory structure**

```bash
mkdir -p screenplays/part1
```

**Step 3: Commit**

```bash
git add screenplays/
git commit -m "Add screenplays directory with sample scene01 hook"
```

---

### Task 5: Validate round-trip with sample screenplay

**Step 1: Use screenplay-to-manim skill on sample**

Invoke the `screenplay-to-manim` skill on `screenplays/part1/scene01_hook.screenplay` and generate a scene file to a temp location.

**Step 2: Compare generated output with existing scene**

Diff the generated file against `part1_kalman_filter/scene01_hook.py` to verify the DSL captures enough information for faithful code generation.

**Step 3: Adjust DSL or skills if gaps found**

If the generated code is missing important details, update `docs/screenplay-dsl.md` and the relevant skill.

**Step 4: Commit any fixes**

```bash
git add docs/ screenplays/
git commit -m "Refine screenplay DSL based on round-trip validation"
```

---

### Task 6: Final commit and push

**Step 1: Run tests to verify nothing broken**

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

Expected: 127 tests passing (no test changes in this plan)

**Step 2: Push**

```bash
git push
```
