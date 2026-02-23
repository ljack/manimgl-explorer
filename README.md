# ManimGL Explorer

Browse annotated source code from [3Blue1Brown](https://www.3blue1brown.com/)'s animation library ([ManimGL](https://github.com/3b1b/manim)).

**[Live Demo →](#)** *(coming soon)*

## What's inside

59 Python files across 16 projects from the 3b1b/videos repo (2024–2026), with:

- Full source code with syntax highlighting (ManimGL-aware)
- Line-level annotations explaining imports, API patterns, math, and animation logic
- Two-tier search across all files (live code scan + pre-built index)
- AI chat assistant (bring your own Anthropic API key)

### Projects covered

| Year | Projects |
|------|----------|
| 2024 | Transformers (GPT deep dive), Holograms, Inscribed Rectangle, Puzzles, Linear Algebra, ANTP, Manim Demo |
| 2025 | Laplace Transform, Cosmic Distance Ladder, Grover's Algorithm, Colliding Blocks v2, Spheres, Guest Videos, Zeta |
| 2026 | Hairy Ball Theorem, Monthly Mindbenders |

## Running locally

```bash
cd manimgl-explorer
python3 -m http.server 8765
# Open http://localhost:8765
```

No build step. No dependencies. Just a static site.

## Architecture

- **Vanilla HTML/CSS/JS** — no frameworks, no bundler
- **Lazy-loaded data chunks** — 16 per-project JS files loaded on demand (~3MB total, but only fetched when needed)
- **`data.js`** — manifest with file tree, chunk index, search index, and ManimGL architecture context
- **`gen_explorer.py`** — Python generator script that reads source files and produces all data

## Regenerating data

If you have the [3b1b/videos](https://github.com/3b1b/videos) repo cloned:

```bash
# Clone videos repo
git clone https://github.com/3b1b/videos.git /tmp/3b1b-videos

# Generate all data files
python3 gen_explorer.py
```

The generator reads Python files from `_2024/`, `_2025/`, `_2026/` and produces annotated chunk files with auto-generated annotations.

## Prompt engineering

See [`promptdev/`](promptdev/) for the prompts used to build this project with Claude Code, including a master prompt that can bootstrap the entire project from scratch.

## License

The source code displayed in the explorer comes from [3b1b/videos](https://github.com/3b1b/videos) (MIT License). The explorer itself is MIT licensed.
