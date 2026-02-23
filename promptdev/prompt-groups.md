# ManimGL Explorer — Prompt Groups

Six prompt groups shaped this build. Each is a self-contained context that informed a specific part of the system.

---

## 1. Environment Bootstrap

**What**: Python 3.14 venv at `/Users/jarkko/_dev/manimgl`, ManimGL v1.7.2 from source at `/tmp/manim`, 3b1b videos at `/tmp/3b1b-videos`.

**Key fixes**: `setuptools<82` (pkg_resources removed in 82+), `audioop-lts` (removed in 3.13+), `trimesh`/`pywavefront` for dev manimlib. MacTeX at `/Library/TeX/texbin/`, FFmpeg 8.0.1.

**Run command**: `cd /tmp/3b1b-videos && PYTHONPATH="/tmp/3b1b-videos:$PYTHONPATH" PATH="/Library/TeX/texbin:$PATH" /Users/jarkko/_dev/manimgl/bin/manimgl <file.py> <SceneName>`

**Gotchas**: macOS `/tmp` → `/private/tmp` (path matching). Symlink `_2025/hairy_ball` → `_2026/hairy_ball`. `custom_config.yml` paths changed from `/Users/grant/...` to `/private/tmp/3b1b-videos/`.

---

## 2. Codebase Catalog

**Scope**: 59 Python files across 16 projects in `_2024/`, `_2025/`, `_2026/`. Total ~80K lines.

**Biggest projects**: transformers (12 files, 17K lines), laplace (9 files, 13K lines), cosmic_distance (5 files, 7K lines), grover (5 files, 6K lines).

**Discovery**: Glob `_20{24,25,26}/*/` → filter directories with `.py` files → skip `_2025/hairy_ball` (symlink to `_2026`).

**Each file has**: scene classes (InteractiveScene subclasses), helper functions, mathematical constants, imported ManimGL primitives.

---

## 3. ManimGL Architecture

**Core pattern**: Scene subclass → `construct()` method → `self.play(Animation)` → `self.wait()`.

**Key classes**: `InteractiveScene` (3D camera + mouse), `Scene` (2D), `Mobject`/`VMobject` (geometry), `Group`/`VGroup` (containers).

**Animation pipeline**: `self.play()` → creates `AnimationGroup` → calls `begin()` on each → interpolates `alpha` 0→1 → calls `finish()`. Key animations: `Transform`, `Write`, `FadeIn`, `MoveAlongPath`, `Indicate`.

**Coordinate system**: `RIGHT = [1,0,0]`, `UP = [0,1,0]`, `OUT = [0,0,1]`. Constants: `TAU = 2π`, `PI`, `DEGREES = TAU/360`.

**The animate builder**: `mob.animate.method().set_anim_args(run_time=2)` creates a `_MethodAnimation` that interpolates from current state to the result of calling `.method()`.

---

## 4. Explorer Web App Architecture

**Stack**: Vanilla HTML/CSS/JS, no build step, no frameworks. Serve with `python3 -m http.server 8765`.

**Layout**: 3-panel — file tree sidebar (left), code viewer (center), AI chat panel (right, toggleable).

**Data architecture**: `data.js` is a manifest (tree + fileIndex + searchIndex + architectureContext). 16 chunk files in `data/` are IIFEs that register into `window.MANIM_DATA.files` on demand.

**Lazy loading**: `loadFile()` checks `fileIndex[path]` → injects `<script>` tag → waits for load → renders. `loadedChunks` Set prevents double-loading.

**Search**: Two-tier — loaded files get live code scan (line-by-line regex), unloaded files use pre-built `searchIndex` (classes/functions/methods extracted at generation time).

**Code rendering**: Custom Python syntax highlighter with ManimGL-aware token sets. Annotations rendered as inline expandable blocks.

**AI chat**: Anthropic API (user provides key), sends current file context + architectureContext as system prompt.

---

## 5. Design System

**Theme**: Dark mode only. Background `#0d1117`, panels `#161b22`, borders `#30363d`. Accent `#3b82f6`.

**Fonts**: JetBrains Mono (code), Inter (UI). Both from Google Fonts.

**Syntax colors**: keywords `#ff7b72`, strings `#a5d6ff`, numbers `#79c0ff`, comments `#8b949e`, decorators `#d2a8ff`, builtins `#ffa657`, ManimGL classes `#7ee787`.

**Components**: Collapsible file tree with project/file icons. Breadcrumb navigation. Annotation badges with line numbers. Loading spinner for chunk fetches. Search with category grouping.

---

## 6. Generator Script

**File**: `gen_explorer.py` (849 lines Python).

**Input**: `/tmp/3b1b-videos/_20{24,25,26}/*/` Python files.

**Output**: 16 chunk JS files + 1 manifest `data.js`.

**Annotation engine**: 6 pattern layers applied per-line:
1. Imports (`from manim_imports_ext import *`, etc.)
2. Class definitions (base class lookup: InteractiveScene, Scene, etc.)
3. ManimGL object creation (Tex, MathTex, Arrow, Surface, etc.)
4. API patterns (`self.play`, `self.wait`, `self.add`, `.animate.`, `.to_edge`, etc.)
5. Animation classes (Transform, FadeIn, Write, etc.)
6. Math operations (np.array, rotate_vector, interpolate, etc.)

**Annotation density**: Target ~15-25% for small files (<200 lines), ~2-5% for large files.

**Escaping**: Python → JS template literal: `\` → `\\`, `` ` `` → `\`​`, `${` → `\${`.

**Overrides**: Hand-written annotations for `lorenz.py` and `ladybug.py` replace auto-generated ones.

**Search index extraction**: Regex `^(\s*)def\s+(\w+)\s*\(` and `^class\s+(\w+)` per file. Indent > 0 → method, else → function.
