# Prompt Development Notes

This directory contains the prompts and context documents used to build the ManimGL Explorer with Claude Code. These are preserved for reproducibility — you can use the master prompt to regenerate or extend the project.

## Files

- **`master-prompt.md`** — Single prompt that bootstraps the entire project from scratch
- **`prompt-groups.md`** — TL;DR of the 6 prompt groups that shaped the build

## Usage

To regenerate from scratch:
1. Clone the [3b1b/videos](https://github.com/3b1b/videos) repo to `/tmp/3b1b-videos`
2. Feed `master-prompt.md` to Claude Code in an empty directory
3. The generator script produces all data files; the rest is hand-crafted
