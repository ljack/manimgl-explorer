# ManimGL Explorer — Master Bootstrap Prompt

> Feed this to Claude Code in an empty directory with access to `/tmp/3b1b-videos` (the [3b1b/videos](https://github.com/3b1b/videos) repo). It will produce the complete explorer.

---

## Prompt

Build a single-page web app called **ManimGL Explorer** that lets users browse annotated source code from 3Blue1Brown's animation library (ManimGL).

### Source data

The 3b1b/videos repo is at `/tmp/3b1b-videos`. Process all Python files in `_2024/`, `_2025/`, `_2026/` (skip `_2025/hairy_ball` which is a symlink to `_2026`). There are 59 files across 16 projects.

### Architecture

**No build step. Vanilla HTML/CSS/JS. Serve with `python3 -m http.server`.**

Data architecture uses lazy-loaded per-project chunks:

- `data.js` — manifest only: file tree structure, `fileIndex` (path → chunk URL), `searchIndex` (pre-extracted classes/functions/methods per file), `architectureContext` (ManimGL reference text for AI chat)
- `data/*.js` — 16 IIFE chunk files, one per project. Each registers its files into `window.MANIM_DATA.files` with `{ description, code, annotations }`.

### Layout

Three-panel layout:
1. **Left sidebar**: Collapsible file tree grouped by year → project → file
2. **Center**: Code viewer with Python syntax highlighting, inline annotations, breadcrumb nav
3. **Right** (toggleable): AI chat panel (Anthropic API, user provides key)

### Code viewer features

- Custom Python syntax highlighter aware of ManimGL tokens (scene classes, mobject types, animation names)
- Annotations displayed as expandable inline blocks at specific line numbers
- Loading spinner when fetching chunks on demand
- Two-tier search: live code scan for loaded files, pre-built searchIndex for unloaded files

### Design

Dark theme only. Background `#0d1117`, panels `#161b22`, borders `#30363d`, accent `#3b82f6`. Fonts: JetBrains Mono (code), Inter (UI) from Google Fonts. Syntax colors: keywords `#ff7b72`, strings `#a5d6ff`, comments `#8b949e`, ManimGL classes `#7ee787`.

### Generator script

Write a Python script `gen_explorer.py` that:

1. Scans `_2024/`, `_2025/`, `_2026/` for Python files
2. Groups files by project (directory name)
3. For each file, generates:
   - A 2-3 sentence description (what it does, key concepts)
   - The full source code (escaped for JS template literals: `\` → `\\`, backtick → escaped backtick, `${` → `\${`)
   - Line-level annotations using pattern matching (not LLM):
     - Imports (what they provide)
     - Class definitions (base class purpose)
     - ManimGL object creation (Tex, Surface, Arrow, etc.)
     - API patterns (self.play, .animate., .to_edge, etc.)
     - Animation classes (Transform, FadeIn, Write, etc.)
     - Math operations (numpy, rotate_vector, interpolate, etc.)
   - Target annotation density: ~15-25% for small files, ~2-5% for large
4. Writes 16 chunk JS files to `data/`
5. Writes `data.js` manifest with tree, fileIndex, searchIndex, architectureContext

### File structure

```
manimgl-explorer/
  index.html
  styles.css
  app.js
  data.js          (manifest: tree + indexes)
  data/
    _2024_antp.js
    _2024_holograms.js
    ... (16 chunk files)
  gen_explorer.py
```

Build the web app first (index.html, styles.css, app.js with lazy loading), then write the generator script, then run it to produce all data files. Verify everything works: all 59 files load, search works across loaded and unloaded files, annotations display correctly.
