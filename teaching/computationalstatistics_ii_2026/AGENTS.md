# Introduction

* This is the folder of my Computational Statistics II class.
* It is a phd level class, but it is not for very strong students (like MIT or Berkeley), but it is still high-level
* Student are also taking a class on Measure Theoretical Probability and Statistical Inference, but I do not thing they have a strong command of those topics.

# Lectures

* Lectures are in /lectures
* They are done in quarto .qmd
* The theme is my personal theme-uv, which is in folder _extensions. There is a README.md there with some options and whatnot

### What type of lectures do I like?

* I don't like a lot of text, but still it is needed
* I like bullet points and enumerated points
* Use latex within markdown.
* Add excercises from time to time, to check definitions and applications of theorems, propositions, etc
* Important equations, and final results have to be highlighted (see _extensions/README.md for options), these include theorems, propositions, important equations
* I like colours, but usually forget to put colours, the same with bold text. Favourite colours: red, blue, purple
* I like to have pictures together with my definitions, usually in a 2 column fasion, sometimes an observable object can be even better, with interactivity
* For pseudo code use _extensions/leovan
* observable  code block should not be writen directly in the main .qmd but rather in an auxiliary .qmd that has to be called, so we keep the main .qmd clean. The same applies for Python and R code. Put the codes in the subfolder /codes of the main .qmd folder
* Both Python and R are used for my teaching, but use Python by default. Code never runs in the quarto file. If you need, you can put an screen shot of the output or copy it verbatim. 
* The lectures are going to be online, so the final html cannot rely on running Python or R computations (e.g. a server running python on real-time)
* images are in the subfolder /images of the main qmd folder



