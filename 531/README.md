# 5/3/1 Companion

A lightweight, mobile-first MVP web app for lifters following Jim Wendler-style 5/3/1 programs.

## Included

- `index.html` — app entry point
- `app.js` — local-first app logic and workout generation engine
- `styles.css` — mobile-first UI styles
- `docs/research/531-program-research.md` — distilled research on how the program works, what varies, and common failure points
- `docs/product/531-companion-prd.md` — product requirements, user flows, information architecture, and MVP scope
- `docs/product/531-companion-data-model.md` — suggested domain model for workouts, cycles, templates, and progression

## MVP features implemented

- onboarding with units, rounding, experience, template, and per-lift max inputs
- conservative training max suggestions
- cycle generation for Vanilla, Beginner, FSL 5x5, and BBB 5x10
- workout view with exact main and supplemental sets
- logging for plus sets, assistance categories, notes, and session flags
- cycle review with increase / hold / reset recommendations
- local history, export, and localStorage persistence

## Run locally

This MVP is static and has no build step.

### Option 1: open directly

Open `index.html` in a browser.

### Option 2: serve locally

```bash
cd 531
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Product concept

A lightweight coaching-and-tracking app for lifters running 5/3/1 variants such as:

- Vanilla 5/3/1
- Beginner 5/3/1
- First Set Last (FSL)
- Boring But Big (BBB)

The app should help users:

- calculate a conservative training max
- see the exact prescribed weights for today
- log AMRAP / PR sets and assistance work
- progress upper and lower lifts correctly between cycles
- avoid common mistakes like starting too heavy or adding too much junk volume

## Important note

Some 5/3/1 material is proprietary or book-specific. This repo focuses on:

- publicly documented concepts
- user-entered template parameters
- app behavior that supports the method without reproducing paywalled content
