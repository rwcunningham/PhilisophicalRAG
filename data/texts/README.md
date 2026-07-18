# Source Texts

Put philosophical source texts here as `.txt`, `.md`, or `.pdf`.

Good starting corpora for this project are public-domain works by authors such as Spinoza, Hume, Hobbes, Schopenhauer, Nietzsche, Kant, Mill, Plato, and Aristotle. For the free-will example, add texts that argue for necessity, determinism, compatibilism, skepticism about agency, or causal explanations of action.

For best citations, include frontmatter in text or Markdown files:

```markdown
---
title: A Treatise of Human Nature
author: David Hume
work: Book II, Part III
year: 1739
tags: necessity, liberty, causation, motives
---

Paste the text here.
```

You can also add a sidecar metadata file next to any source, for example `treatise.txt` plus `treatise.metadata.json`:

```json
{
  "title": "A Treatise of Human Nature",
  "author": "David Hume",
  "work": "Book II, Part III",
  "year": "1739",
  "tags": ["necessity", "liberty", "causation", "motives"]
}
```
