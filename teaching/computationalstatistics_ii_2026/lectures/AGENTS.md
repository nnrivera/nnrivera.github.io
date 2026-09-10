# 1 Intro

This is the setup of the Quarto stuff I use for my slides. This is work in progress as this is the first time using quarto and my custom style. It is work in progress. This file is just a copy of an obsidian page with info about it

UPDATED: sept 10, 2026
# 2 UV-theme 

I am creating a UV-theme with the colours etc..

### 2.1.1 TODO
- [ ] Frame title (as in latex). Right know I essentially remade one as part of the slide. The problem is that if I center (align) the slide then it "centers" the title
### 2.1.2 BUGS
- [ ] the box works bad if no title is given


# 3 Quarto for math

## 3.1 Theorems and stuff

These are Quarto’s native mathematical blocks. They feature automatic cross-referencing (e.g., `@thm-label`), automatic sequential numbering, and standard LaTeX-style formatting.

| **Environment**                  | **Div Syntax**     | **Cross-Reference Tag** | **Description**                                     |
| -------------------------------- | ------------------ | ----------------------- | --------------------------------------------------- |
| **Theorem**                      | `::: {#thm-label}` | `@thm-label`            | Formal mathematical theorem                         |
| **Lemma**                        | `::: {#lem-label}` | `@lem-label`            | Auxiliary proposition to prove a theorem            |
| **Corollary**                    | `::: {#cor-label}` | `@cor-label`            | Direct consequence of a theorem                     |
| **Proposition**                  | `::: {#prp-label}` | `@prp-label`            | Standalone mathematical result                      |
| **Conjecture**                   | `::: {#cnj-label}` | `@cnj-label`            | Unproven mathematical proposition                   |
| **Definition**                   | `::: {#def-label}` | `@def-label`            | Formal mathematical definition                      |
| **Example**                      | `::: {#exm-label}` | `@exm-label`            | Concrete worked example                             |
| **Exercise**                     | `::: {#exr-label}` | `@exr-label`            | Problem posed to students                           |
| **Hypothesis**                   | `::: {#hyp-label}` | `@hyp-label`            | Scientific/experimental hypothesis                  |
| **Claim**                        | `::: {#clm-label}` | `@clm-label`            | Assertion within a larger proof                     |
| **Proof** <br>(Extensions below) | `::: {.proof}`     | _(None)_                | Unnumbered formal proof block (adds $\blacksquare$) |
| **Remark**                       | `::: {.remark}`    | _(None)_                | Unnumbered observation or n                         |

**Example:** 
```
::: {#thm-mvt}
## Mean Value Theorem
If $f$ is continuous on $[a,b]$ and differentiable on $(a,b)$, then $\exists c \in (a,b)$ such that $f'(c) = \frac{f(b)-f(a)}{b-a}$.
:::

::: {.proof}
Define $g(x) = f(x) - rx$ and apply Rolle's Theorem...
:::
```


### 3.1.1 Custom boxes

In the UV-theme that I created I did (with the chat)

| **Environment**            | **Div Syntax**                                                    | **Default Title** | **Description**                                                                                         |
| -------------------------- | ----------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| **Proof Idea**             | `::: {.proof_idea}`                                               | _Proof Idea_      | Proof Idea                                                                                              |
| **Collapsible Proof**      | `::: {.proofbox}`                                                 | _Proof_           | Proof that can be collapsed. Use with a scrollable slide                                                |
| **Custom Title Box**       | `::: {.proofbox title="..."}`                                     | _Your Title_      | same as above                                                                                           |
| **Equation Highlight Box** <br> (currently not working) | `::: {.eqbox}`                                                    | _(None)_          | just an equation                                                                                        |
| **Box**                    | ```::: {.box title = ..."}```                                     | _(None)_          | A box. It does not work well without the title                                                          |
| **explained-equation**     | ``::: {.explained-equation}``                                     | *(none)*, for now | It allows the explanation next to the equation with ``[Explain](#name-explanation){.explanation-link}`` |
| **explanation**            | ```::: {#name-explanation .explanation title="Curvature bound"}`` | *(none)*          | Allows for explanations (comic-like dialog box)                                                         |
