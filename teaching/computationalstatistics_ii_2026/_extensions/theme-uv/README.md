# UV theme

**Work in progress.** This is an experimental teaching theme, developed while
preparing the lectures. Layout and styling are still being adjusted, and some
features are incomplete. See the known issues below before reusing it.

A Quarto Reveal.js theme for Universidad de Valparaíso lectures, with a
Beamer-inspired layout: UV blues, a custom title slide, a gradient footer,
mathematical blocks, collapsible proofs, and clickable explanations.

## Start a presentation

With `_extensions/theme-uv` in the project, use this YAML in a lecture `.qmd`:

```yaml
---
course: "DOCE-202 Computational Statistics II"
title: "Unit 1: Advanced Optimization"
author: "Nicolás Rivera"
exercise-list: true
format:
  theme-uv-revealjs:
    slide-level: 2
    width: 1600
    height: 900
    center: false
    toc: true
    toc-depth: 1
---
```

Run `quarto preview lecture_1.qmd` while editing, or `quarto render lecture_1.qmd`
to produce the HTML. The dimensions above match the current lecture; change
them to adjust the slide canvas. Keep `center: false` for the intended layout.

The title page displays the course, title, author, and two logos. Other slides
show the author on the left of the footer, the presentation title in the middle,
and the slide number on the right. The footer is hidden on the title page.

**Logo paths:** `title-slide.html` currently uses `../assets/logo_uv.png` and
`../assets/imagotipo.svg`. This fits lectures rendered inside `lectures/` with
`assets/` beside that folder. Adjust the paths if you move the output elsewhere.

## Organise your slides

Use `# Topic` for section dividers, `## Slide title` for individual slides, and
`### Subheading` within a slide. A dense slide can use
`## A longer discussion {.scrollable}`. Standard Quarto columns and fragments
remain available.

For a little emphasis, use `[Important]{.red}` or `[Comment]{.blue}`. These two
classes use ordinary red and blue; the theme's UV palette styles the headings,
footer, bullets, and boxes.

## Definitions, theorems, and exercises

Use Quarto's numbered mathematical blocks. Give each one a unique ID; `name`
is optional. Refer to it elsewhere with `@` followed by its ID.

```markdown
::: {#def-convex name="Convex function"}
A function is convex if...
:::

See @def-convex.
```

| Block | ID prefix |
|---|---|
| Definition | `def-` |
| Theorem / lemma / corollary | `thm-` / `lem-` / `cor-` |
| Proposition / conjecture | `prp-` / `cnj-` |
| Example / exercise | `exm-` / `exr-` |

Quarto supplies the numbering and cross-references; the theme supplies the
coloured backgrounds, borders, and titles. A regular proof uses `::: {.proof}`.

## Boxes and proofs you can open

The custom boxes use the same fenced-block syntax:

```markdown
::: {.proofbox title="Why does this work?"}
Write the proof here, including any mathematics.
:::
```

| Opening fence | What it does |
|---|---|
| `::: {.box title="Remark"}` | A tinted box with a title and blue border. |
| `::: {.box .turquoise title="Remember"}` | A lighter accent colour. |
| `::: {.proofbox}` | A collapsible proof, titled “Proof”, with a square at the end when open. |
| `::: {.proof_idea}` | A purple collapsible box, titled “Proof Idea”. |
| `::: {.eqbox title="Key equation"}` | An equation container with spacing and an optional blue title. |
| `::: {.exercise title="Quick question"}` | A styled, unnumbered exercise box. |

Close each block with `:::`. Use `title="..."` to rename a proof box. Prefer
giving `.box` a title. The `.eqbox` styling is unfinished; do not expect a fully
styled equation highlight yet.
For numbered exercises and the automatic index, use `#exr-...` instead of a
plain `.exercise` box.

## Clickable explanations

Place a small link beside something students might want explained, then write
the explanation as Markdown with a matching ID:

```markdown
The Hessian is positive semidefinite.
[Explain](#positive-hessian){.explanation-link}

::: {#positive-hessian .explanation title="Positive semidefinite"}
This means $v^\top H v\geq 0$ for every vector $v$.
:::
```

The link opens a speech bubble beside it. Click again, click outside, or press
Escape to close it; changing slides closes it too. For a display equation and
its link, wrap their paragraph in `::: {.explained-equation}`. An explanation
without an explicit link gets a link using its title. In print/PDF view, the
explanations appear open in the slide content.

## Automatic exercise list

Set `exercise-list: true` to append an **Exercises** slide at the end. It is off
by default; `false` disables it. There is no placeholder to maintain.

```markdown

## Practice

::: {#exr-gradient name="Compute the gradient"}
Find the gradient of $f(x)=x^\top x$.
:::
```

The entry reads “Exercise 1: Compute the gradient”, using Quarto's actual
number. Without `name`, it reads “Exercise 1”. Each link opens the containing
slide, so several exercises can share a slide, including exercises in columns.
Subquestions stay together under their exercise's entry.

Only numbered `exr-` blocks are collected. A heading that says “Exercise” is
not enough. Put exercises after a slide heading (`##` in this setup), including
after any horizontal-rule slide break. No index slide is added if there are no
numbered exercises. Long lists are scrollable.

## Where to make changes

| File | Edit it to change… |
|---|---|
| [_extension.yml](_extension.yml) | Format defaults and which theme files are loaded. |
| [theme.scss](theme.scss) | Colours, fonts, footer, headings, bullets, and box appearance. The main UV colours are at the top. |
| [title-slide.html](title-slide.html) | Title-page layout, displayed metadata, and logo paths. |
| [beamer-blocks.lua](beamer-blocks.lua) | Custom box titles and collapsible proof markup. |
| [explanation.lua](explanation.lua) | Conversion of explanation blocks while retaining Markdown mathematics. |
| [explanation.html](explanation.html) | Bubble positioning, opening, closing, and print behaviour. |
| [exercise-index.lua](exercise-index.lua) | Collection of numbered exercises and generation of the final list. |

`custom.scss` is a spare stylesheet: it is **not loaded** by `_extension.yml`,
so editing it alone will not change the slides. The bundled `logo.svg` is also
not the logo used by the current title template.

## Known issues and TODO

The working notes in [lectures/AGENTS.md](../../lectures/AGENTS.md) record these
unfinished parts:

- **Frame titles:** the title is currently part of the slide layout rather than
  an independently positioned Beamer-style frame title. Centering a slide also
  moves its title. Keep `center: false` until this is redesigned.
- **Boxes without titles:** `.box` does not look right without a title. Supply
  `title="..."` for now.
- **Equation highlighting:** `.eqbox` is marked as not yet working in the notes.
  The current code provides a container and optional title, but its intended
  equation-highlight appearance remains unfinished.
- **Font Size:** I wish to have a simpler way to change the fon't size of my slides.
  

There are also a few implementation details to keep in mind: some CSS rules
are repeated and many use `!important`, so check for later overrides when an
edit seems to have no effect. The `.uv-subtitle` style exists, but the title
template does not currently render `subtitle`. The custom interactive features
are designed for Reveal.js HTML; they are not a separate LaTeX Beamer theme.
