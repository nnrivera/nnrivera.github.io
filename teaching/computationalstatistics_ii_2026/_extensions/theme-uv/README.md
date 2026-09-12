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
subtitle: "Gradient methods and Newton’s method"
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
them to adjust the slide canvas. Use `center: false` for top-aligned content, or `center: true` to centre
regular slide bodies beneath their fixed headings.

The title page displays the course, title, optional subtitle, author, and optional title-page images.
Set `subtitle: "..."` in the lecture metadata to display it between the title and
author. Omit it (or use `subtitle: ""`) for no subtitle and no extra space. Other slides
show the author on the left of the footer, the presentation title in the middle,
and the slide number on the right. The footer is hidden on the title page.

### Lecture or talk title page

The default `title-layout: lecture` keeps the current teaching layout. For a
classic centred talk opening, set `title-layout: talk` under
`format: theme-uv-revealjs:`:

```yaml
title: "Optimisation beyond smoothness"
subtitle: "Geometry, algorithms and statistical applications"
author:
  - name: Nicolás Rivera
    affiliation: Universidad de Valparaíso
venue: "Research seminar · Host university"
date: 2026-10-01
date-format: "D MMMM YYYY"
title-logo: images/host-logo.png
title-logo-alt: "Host university"
format:
  theme-uv-revealjs:
    title-layout: talk
```

`title-layout` is a theme-specific format option. Presentation metadata such as
`title`, `author`, `venue` and `date` stays at the top level.

The talk layout places a larger title above the authors, with each author's
Quarto affiliations directly below their name. Add more entries to `author`
for multiple authors; their groups appear side by side and wrap if needed.
The default slide footer lists their names, separated by commas.
A simple `author: "Name"` also works without an affiliation. The title uses
institutional colour 1, with the remaining text in colour 2.

`subtitle`, `venue`, `date` and the images are optional. The venue and date appear
below the authors; host logos are centred beneath them using the existing
`title-logo` and `title-wordmark` fields. The talk layout has no large gradient
band and does not display `course`. It affects only the opening slide; normal
slides keep the existing theme. Omit `title-layout` or set it to `lecture` to
return to the teaching layout. Invalid layout names produce a render error.

### Optional title-page images

Keep image files with your lecture or course, outside `_extensions`. Set either
or both paths in the lecture metadata:

```yaml
title-logo: images/logo.png
title-wordmark: images/wordmark.svg
title-logo-alt: "University name"
title-wordmark-alt: "University name"
```

Paths are relative to the lecture `.qmd`. For a shared `assets/` folder beside
`lectures/`, use `../assets/logo_uv.png` and `../assets/imagotipo.svg`, as the
current lectures do. In the lecture layout, a logo appears on the left, a wordmark on the right;
either can be used alone. Omit both paths (or set them to empty strings) to
omit the entire logo area. The lecture layout retains its gradient; talk layouts centre the supplied images
beneath the event details.
There are no institution-specific image defaults in the extension.

The optional `*-alt` fields describe the images for assistive technology.
Without them, images have empty alternative text and are treated as decorative.

## Customise the institutional colours

The theme uses three Sass variables, defined at the top of [theme.scss](theme.scss):

```scss
/*-- scss:defaults --*/
$institutional_colour_1: #232754 !default; // Dark colour: headings and text
$institutional_colour_2: #0F4494 !default; // Main colour: borders and links
$institutional_colour_3: #009FE3 !default; // Bright accent
```

These replace the earlier `$uv-dark-blue`, `$uv-blue` and `$uv-light-blue` names.
The UV colours are the defaults; colleagues can supply their own palette.
Keep colour 1 dark enough for readable text on pale backgrounds.
Headings, footer, bullets, institutional box borders and tints, and numbered
styles 1–4 all derive from these variables. Styles 5–10 are independent palettes;
explicit equation `colour` values and the default proof/idea colours remain separate.

To override the defaults without editing the extension, put those three variables
with your chosen values in `my-colours.scss` beside your lecture, then append it
to the theme list:

```yaml
format:
  theme-uv-revealjs:
    theme: [default, ../_extensions/theme-uv/theme.scss, ../_extensions/theme-uv/block-styles.scss, my-colours.scss]
```

The paths above assume the lecture is in `lectures/`. Numbered palettes and their
internal CSS classes now use `block-style`; replace the experimental `uv-style`
attribute in older sources with `block-style`.

## Slide text size

Set `fontsize` in pixels using Quarto's format options:

```yaml
format:
  theme-uv-revealjs:
    fontsize: 28px
```

The default is `24px`. Slide headings, mathematical blocks and their spacing
scale with the body text. Popup text also follows the setting, with the existing
viewport adjustment. Footer labels retain their compact screen-dependent size;
slide numbers scale with the slide font. The footer bar remains 45px tall.
Larger text does not automatically fit more content into the same slide: check
dense slides and long footer text after changing the size. Browser paper-print
mode retains Reveal's own print typography; Reveal PDF uses the slide font size.

## Organise your slides

Use `# Topic` for section dividers, `## Slide title` for individual slides, and
`### Subheading` within a slide. A dense slide can use
`## A longer discussion {.scrollable}`. Standard Quarto columns and fragments
remain available.

For a little emphasis, use `[Important]{.red}`, `[Comment]{.blue}` or
`[Key assumption]{.purple}`. These optional styling utilities use red, blue and
purple (`#7d3c98`), independently of the institutional palette and numbered block
styles. The class syntax is native Quarto; these colour definitions belong to
this extension.

### Centre content beneath a fixed slide heading

Use `.center` on an individual slide:

```markdown
## A key result {.center}

Content to centre vertically beneath the heading.
```

Alternatively, set `center: true` under `theme-uv-revealjs` to centre regular
slide bodies throughout. The title and its underline stay at the top, including
when a long title wraps. Centring is vertical; horizontal text alignment stays
unchanged. Ordinary slides with `center: false` keep their existing layout.

This feature follows the extension's `slide-level: 2` structure. Section
dividers, the title page, contents and `.scrollable` slides retain their own
layouts. Oversized content starts below the heading rather than being centred
above it; use `.scrollable` or split the slide if it does not fit. Browser paper
printing uses document flow; Reveal PDF retains the centred slide layout.

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

## Ten numbered block styles

Add `block-style="1"` through `block-style="10"` to an individual environment. This is
an attribute, so it has **no leading dot**. The name `style` is reserved for
inline CSS and remains independent:

```markdown
::: {#thm-example name="A useful result" block-style="3"}
Write the theorem here.
:::

::: {.eqbox title="Key equation" block-style="3" alpha="0.15"}
$$x_{k+1}=x_k-\eta\nabla f(x_k)$$
:::
```

| Number | Palette |
|---|---|
| `1` | Institutional colour 1 (dark blue by default) |
| `2` | Institutional colour 2 (blue by default) |
| `3` | Institutional colour 3 (cyan by default) |
| `4` | Blend of the institutional colours |
| `5` | Crimson |
| `6` | Purple |
| `7` | Teal |
| `8` | Green |
| `9` | Amber |
| `10` | Slate |

Works with numbered definitions, theorems, lemmas, propositions, corollaries,
conjectures, examples and exercises, regular proofs, and custom `.box`,
`.proofbox`, `.proof_idea`, `.eqbox` and `.explanation` blocks.
Each environment keeps its own layout. Omit the attribute to keep its existing
default appearance. Invalid numbers produce a render error.

Equation capsules still accept `alpha`; an explicit `colour` overrides the
palette's capsule background colour. Put `block-style` on the **original** block
when using `.replicate`: replicas retain that block's palette and numbering.
The local `lectures/lecture_test.qmd` tutorial shows all ten styles and their code.

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
| `::: {.eqbox title="Key equation"}` | A pale blue equation capsule with an optional left-aligned caption. |

Close each block with `:::`. Use `title="..."` to rename a proof box.
Proof and proof-idea titles are plain text: `<`, `>` and `&` display literally;
HTML and Markdown are not interpreted in titles. Their bodies retain normal
Markdown and mathematics.
The title of a generic `.box` is optional.
Exercises use Quarto's native `#exr-...` blocks, with `name` for the optional
title and `block-style` for the palette:

```markdown
::: {#exr-stationary name="Try it" block-style="9"}
Find a stationary point of $f(x)=x^2$.
:::

See @exr-stationary.
```

The experimental `.exercise title="..."` form has been removed; migrate it to
`#exr-... name="..."` for native numbering, cross-references and the exercise index.

## Generic content boxes

A `.box` provides a background, border and padding around ordinary Quarto
content: paragraphs, lists, mathematics, images, tables or a combination.
It keeps the normal slide text size and grows with its content; it does not
add automatic fitting or scrolling. Content must still fit the slide or column.

```markdown
::: {#key-observation .box title="Remember" block-style="3"}
Check the assumptions before applying the result.

- The function must be differentiable.
- Choose a positive step size.
:::

See [the earlier observation](#key-observation).
```

- `title` is optional **plain text**, not Markdown or HTML. Omitted or empty
  titles produce a box without a heading or separator.
  Characters such as `<`, `>` and `&` are displayed literally.
- `block-style="1"`–`"10"` selects a palette; omitting it keeps the default.
- An optional unique ID such as `#key-observation` enables ordinary links and
  `.replicate`. Use an ordinary ID rather than a reserved Quarto prefix such
  as `thm-`, which would give the block theorem semantics.
- Boxes have no automatic numbering or `@` cross-reference type. Use Quarto's
  theorem environments when those features are needed.
- Collapsing and popups remain separate components: use `.proofbox` or
  `.explanation` where appropriate, and compose them with replicas.

Repeat the same labelled box, including its title and palette, with:

```markdown
::: {.replicate ref="key-observation"}
:::
```

See the generic-box examples and source snippets in
[lecture_test.qmd](../../lectures/lecture_test.qmd).

## Mathematics engine

The extension leaves `html-math-method` unset so Quarto uses its Reveal-specific
MathJax default. The installed Quarto uses MathJax 2 for Reveal. Explicitly setting
`html-math-method: mathjax` selects a different default URL in this Quarto version;
leave the option unset for these presentations.
Do not replace only the library URL with MathJax 3 or add a second manual loader:
the plugin and library APIs must match. Replication waits for the active engine
to finish typesetting. A future MathJax upgrade should update both sides together.

## Highlight an equation

Use `.eqbox` for a compact pale blue capsule, centred in its slide or column.
The equation retains the surrounding font size. An optional small, muted title
sits above the capsule, aligned with its left edge; omitting it leaves no title
space. There is no border, shadow or theorem-style title bar.

```markdown
::: {.eqbox title="Gradient descent"}
$$
x_{k+1} = x_k - \eta_k \nabla f(x_k)
$$
:::
```

### Optional colour and opacity

```markdown
::: {.eqbox title="Gradient ascent" colour="red" alpha="0.15"}
$$
x_{k+1} = x_k + \eta_k \nabla f(x_k)
$$
:::
```

- `colour`: `blue` (default), `red`, `purple`, or a six-digit hex value such as
  `#0F4494`. The spelling `color` is also accepted; `colour` takes precedence.
- `alpha`: background opacity from `0` (transparent) to `1` (opaque), default
  `0.10`. It affects only the fill, keeping the formula and caption fully opaque.
- Both settings are optional and independent. The default blue at `0.10`
  preserves the original pale blue appearance on white slides. On coloured
  slides, the transparent fill blends with the background.
- Use light fills for readable dark mathematics. Invalid colours or alpha
  values stop rendering with an explanatory error.

### Numbered equations and links

Use Quarto's native equation label **after the closing `$$`**, inside `.eqbox`:

```markdown
::: {.eqbox title="Gradient descent"}
$$
x_{k+1} = x_k - \eta_k \nabla f(x_k)
$$ {#eq-gradient-step}
:::

See @eq-gradient-step.
```

Quarto supplies the equation number and linked reference, including references
from other slides. Each label must be unique and start with `eq-`. The optional
`title` is a visual caption; it does not create an equation label. See
[Quarto's equation references](https://quarto.org/docs/authoring/cross-references.html#equations).

The examples in [lecture_test.qmd](../../lectures/lecture_test.qmd) include colour and
opacity comparisons and a link back to an equation on an earlier slide.

### Size and optional scrolling

The capsule grows to fit the mathematics in both width and height. It has no
internal scrollbar by default, including for `aligned` equations. Use an
`aligned` environment inside the display mathematics to break long formulas
into several lines.

To constrain a wide equation to its slide or column, explicitly opt in:

```markdown
::: {.eqbox title="A long equation" scroll="true"}
$$ ... $$
:::
```

`scroll="true"` enables horizontal scrolling only. Omit it (or use
`scroll="false"`) for natural sizing. A naturally sized equation can extend
beyond a narrow column; split it across lines or opt into scrolling in that case.
For print/PDF, capsules grow to show their contents even when scrolling is
requested; split equations that would exceed the page width.

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

The default title is **“Explanation”**; use `title="..."` to override it.
The link opens a speech bubble beside it. Click again, click outside, or press
Escape to close it; changing slides closes it too. For a display equation and
its link, wrap their paragraph in `::: {.explained-equation}`. An explanation
without an explicit link gets a link using its title. In print/PDF view, the
explanations appear open in the slide content.

## Replicate a labelled environment

Write the original once, then repeat it wherever needed:

```markdown
::: {.replicate ref="thm-descent"}
:::
```

`.replicate` is an empty destination block. It copies the original content,
including its number, title, mathematics and styling, without a popup or an
extra title/link. Put any commentary or `@thm-descent` link outside it.
`ref="thm-descent"` and `ref="#thm-descent"` are both accepted.

- Use any original label: equations, theorems, definitions, lemmas,
  propositions, corollaries, examples, exercises, proofs, figures, tables,
  code blocks and ordinary labelled divs.
- An equation label inside `.eqbox` copies the entire capsule, including its
  caption, colour and alpha. Give a custom environment an ID such as
  `::: {#notation .box title="Notation"}` to replicate the whole block.
- Replicas work directly on slides, in columns and inside other environments.
  Source content can appear earlier or later in the same presentation.
- Original numbering is retained. Replica IDs are renamed to avoid duplicate
  anchors; copies do not add entries to the exercise index. Links still lead
  to the originals. Proofs retain their collapsible behaviour.
- Changing the original updates all copies on the next render. Source fragments
  are visible immediately in replicas. Browser print/PDF output includes them.
- Reference original labels, not IDs attached to replicas. Missing labels and
  circular references show an error at the destination. A missing `ref` or a
  nonempty `.replicate` block is a render error.

This is an HTML/Reveal.js feature for static content. Interactive plots,
canvas drawings and embedded applications are not supported as live copies.
Replication runs in the browser after mathematics has been typeset; it does
not execute Python or R or create additional equation numbers.

### Put a replica in a popup

An explanation controls how content is shown; a replica supplies that content:

```markdown
[Remember the theorem](#theorem-reminder){.explanation-link}

:::: {#theorem-reminder .explanation title="Remember the theorem"}
::: {.replicate ref="thm-descent"}
:::

See @thm-descent for the original.
::::
```

Use four colons for the outer explanation and three for the replica inside it.
For equations, use `ref="eq-gradient-step"` instead. Long equations can widen a
popup up to 900 px, bounded by the viewport; larger content can be scrolled.
In print/PDF view, explanations and their replicas appear inline.

The earlier experimental `.explanation ref="..."` and direct popup links to
source labels have been replaced by this composition. `.explanation-link`
points to an explanation; `.replicate ref` points to the original content.
Handwritten explanation bodies continue to work.

See [lecture_test.qmd](../../lectures/lecture_test.qmd) for rendered examples with the
important source snippets beside them.

## Automatic exercise list

Set `exercise-list: true` to append a top-level **Exercise list** heading followed
by an **Exercises** slide at the end. The heading includes the section in the table
of contents even with `toc-depth: 1`. It is off by default; `false` disables it.
There is no placeholder to maintain.

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
| [theme.scss](theme.scss) | Colours, fonts, footer, headings, bullets, and box appearance. The three institutional colour defaults are at the top. |
| [block-styles.scss](block-styles.scss) | The ten numbered block palettes, including the four derived from institutional colours. |
| [block-styles.lua](block-styles.lua) | Validation of the `block-style` attribute and its CSS classes. |
| [title-page.lua](title-page.lua) | Title-layout selection and author names for the default footer. |
| [title-slide.html](title-slide.html) | Title-page layout, displayed metadata, and logo paths. |
| [beamer-blocks.lua](beamer-blocks.lua) | Custom box titles and collapsible proof markup. |
| [centred-content.html](centred-content.html) | Separation of fixed slide headings from vertically centred bodies. |
| [replicate.lua](replicate.lua) | Validation and markup for replica destinations. |
| [replicate.html](replicate.html) | Copying labelled content and preserving numbering and unique IDs. |
| [explanation.lua](explanation.lua) | Conversion of explanation blocks while retaining Markdown mathematics. |
| [explanation.html](explanation.html) | Bubble positioning, opening, closing, and print behaviour. |
| [exercise-index.lua](exercise-index.lua) | Collection of numbered exercises and generation of the final list. |

`custom.scss` is a spare stylesheet: it is **not loaded** by `_extension.yml`,
so editing it alone will not change the slides. The bundled `logo.svg` is also
not the logo used by the current title template.

### CSS overrides

Proof boxes and proof ideas share the `.beamer-box` component, with `.proof`
and `.idea` variants. Generic `.box` containers keep their own simpler markup;
they share palette support rather than a collapsible base class.

Component rules are scoped to `.reveal`, with numbered palettes loaded after
base styles. Proofs share `--proof-header`, `--proof-fill` and `--proof-qed`
variables, so colour variants do not duplicate layout rules. Custom rules loaded
later can override component styles using the same selector, without `!important`.
For example:

```scss
/*-- scss:rules --*/
.reveal .slides h2 { font-size: 1.1em; }
.reveal details.beamer-box > summary { padding: 0.5em 1em; }
```

Nine `!important` declarations remain in `theme.scss`, each with a reason:

- Hide the title-page footer and slide number despite inline visibility settings.
- Keep the title layout flex-based despite Reveal's inline `display: block`.
- Preserve title top/left/width and contents top positioning during PDF export.
- Preserve contents padding and printed section-heading alignment against
  Reveal's forced print rules.
- Override MathJax 2's forced `10000em` width for numbered equations inside
  naturally sized capsules, keeping the formula and number inside the box.

Do not remove these exceptions without checking both screen and print/PDF output.
The contents slide now follows Reveal's normal hiding behaviour when inactive.

## Known issues and TODO

This checklist consolidates the working notes in
[lectures/AGENTS.md](../../lectures/AGENTS.md), the teaching preferences in
[AGENTS.md](../../AGENTS.md), and a source review on 11 September 2026.
Unchecked items remain open; completed items describe the changes and checks.

### Reported layout issues

- [x] **Keep frame titles fixed when centring slide content.** Centred level-two
  slides use separate heading and body containers: the heading stays at the
  top, and the body centres in the remaining space. Supports per-slide `.center`
  and global `center: true`; scrollable slides retain their normal layout.
- [x] **Make untitled boxes look correct.** Omitted or empty
  titles now produce the same plain container. Edge margins are normalised for
  paragraphs and lists, while spacing between content blocks is retained.
- [x] **Finish equation highlighting.** `.eqbox` now uses a compact pale blue
  capsule with centred mathematics and an optional left-aligned caption above
  it. Supports untitled and multiline equations and placement inside columns.
- [x] **Keep labelled equations inside their capsules.** Override MathJax 2's
  forced full-width numbered display inside `.eqbox`, using the formula and
  tag's measured minimum width. Equation numbers and cross-references remain native.

### Issues found in the source

- [x] **Fix the turquoise box border.** The general border colour is set before
  the bright left accent, so it no longer overwrites it. See **Generic boxes:
  blue and turquoise** in [lecture_test.qmd](../../lectures/lecture_test.qmd).
- [x] **Render generic-box titles safely.** `.box` titles use plain-text Pandoc
  elements, so literal `<x>` or `&` cannot be interpreted as HTML markup.
- [x] **Escape proof titles before inserting HTML.** `.proofbox` and `.proof_idea`
  share one renderer which escapes their plain-text titles. Defaults, body
  mathematics, IDs, palettes and collapsible behaviour are retained.
- [x] **Preserve proof-box IDs and attributes.** The proof filter keeps the
  original outer div around `<details>`, preserving anchors, classes and
  attributes for styling and reference popups.
- [x] **Render subtitle metadata.** The title template displays an optional
  subtitle between the title and author, using the existing `.uv-subtitle`
  styling. Omitted or empty subtitles produce no extra element or spacing.
- [x] **Match the MathJax version to Reveal's maths plugin.** The extension uses
  Quarto's default MathJax configuration for Reveal (MathJax 2 with the installed
  Quarto version). Removed the incompatible MathJax 3 URLs from the lectures
  and the duplicate manual loader from `lecture_1.qmd`. Replication supports
  the matching MathJax 2 queue as well as MathJax 3's startup promise.

### Configuration and maintenance

- [ ] **Review YAML option placement (deferred).** Decide which settings belong
  under `format: theme-uv-revealjs:` and which belong in top-level document
  metadata. Review `title-layout`, title-page images and their alternative text,
  `exercise-list`, `course` and `venue`, alongside standard fields such as
  `title`, `author` and `date`. Document a consistent convention with a complete
  YAML example, and update the tutorial once the design is settled. Consider
  compatibility and precedence if both placements are accepted. No further
  configuration changes are planned until this review.
- [ ] **Consider YAML-defined colour shortcuts (future).** Allow authors to
  define a small mapping of names to colours in lecture metadata, then use
  those names as inline classes, such as `[Key assumption]{.highlight}`.
  Assess whether this convenience justifies the extra configuration and code;
  avoid turning it into a general styling language. If implemented, define
  validation, class-name collision handling and precedence relative to the
  existing `.red`, `.blue`, `.purple` and institutional palettes. Syntax and
  behaviour remain undecided; no implementation is planned yet.
- [ ] **Separate the visual theme from the functional extension (future).**
  Keep colours, typography, title-page layout, spacing and component appearance
  in the visual theme. Move replication, explanations, collapsible proofs,
  custom-block rendering and the exercise index into a functional extension
  with minimal styling so its components remain usable with other Quarto
  themes. The visual theme would supply their full appearance. Audit shared
  assumptions such as `slide-level: 2` and document how to use the two parts
  together or independently. Preserve current behaviour during migration;
  this is a future architectural task, not a prerequisite for using the extension.
- [x] **Provide a simple slide font-size setting.** Quarto's `fontsize` option
  controls the body size, defaulting to `24px`. Headings and boxes use relative
  sizes; popup typography follows the setting. See **Slide text size** above
  for footer and print behaviour.
- [x] **Consolidate repeated CSS rules and audit `!important`.** Proof and generic
  box layouts each have one set of rules; proof palettes use CSS variables.
  Reduced 135 declarations to eight documented exceptions for Reveal/Quarto
  inline styles and forced print rules. Checked title, contents, ordinary and
  centred slides, narrow screens, popups, paper print and Reveal PDF layouts.
  A ninth exception now handles MathJax 2's forced numbered-equation width;
  see **CSS overrides** above.
- [ ] **Review the eight remaining `!important` declarations.** The main source
  is the custom title page and its interaction with Quarto/Reveal's inline
  styles, especially during PDF export. The remaining declarations cover:
  - **Title-page footer and slide number:** one `display: none` declaration
    overrides their inline visibility settings.
  - **Title-page layout:** one `display: flex` declaration overrides Reveal's
    inline `display: block`; three declarations for `top`, `left` and `width`
    override PDF positioning to keep the title flush with the page.
  - **Contents slide:** one `top` declaration overrides the inline PDF offset,
    and one `padding-top` declaration preserves spacing against PDF and forced
    paper-print rules.
  - **Printed section headings:** one `text-align` declaration preserves centred
    headings against Reveal's forced left alignment.
  These are currently justified exceptions: browser comparisons showed layout
  changes when removing them. Investigate whether restructuring the title page
  and adjusting contents/print styling can remove them without changing the
  intended appearance. This requires layout or export changes, rather than
  simply deleting the declarations; recheck screen, paper print and Reveal PDF
  output. See **CSS overrides** above for the current behaviour.
- [x] **Make logo paths configurable.** Optional `title-logo` and `title-wordmark`
  metadata use lecture-relative paths, with optional alternative text. Either
  image can appear alone; omitting both removes the logo area. The existing
  lectures explicitly supply their shared `../assets/` paths.
- [x] **Add a purple emphasis class.** `.purple` is an optional text-colour
  utility alongside `.red` and `.blue`, independent of institutional colours.
- [x] **Keep the usage notes consistent.** The root `AGENTS.md` points to
  `_extensions/theme-uv/README.md`. Both this guide and the lecture notes
  document “Explanation” as the default explanation title.
- [ ] **Add a small visual example deck.** Include titled and untitled boxes,
  highlighted equations, centred content, long titles, explanations and the
  exercise index. Use it to check slide, scroll and print/PDF views after theme
  changes.

The custom interactive features target Reveal.js HTML; this extension is not a
separate LaTeX Beamer theme.
