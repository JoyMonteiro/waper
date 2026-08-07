# Contributing

Thank you for considering reading this guide!
Contributions are welcome :)


* [Types of Contributions](#types-of-contributions)
* [Contributor Setup](#setting-up-the-code-for-local-development)
* [Contributor Guidelines](#contributor-guidelines)
* [Contributor Testing](#running-the-tests)
* [Building the Documentation](#building-the-documentation)
* [Core Committer Guide](#core-committer-guide)


## Types of Contributions

You can contribute in many ways:

### Report Bugs

Report bugs at [https://github.com/JoyMonteiro/waper/issues](https://github.com/JoyMonteiro/waper/issues).

A bug means encountering different behaviour than the expected or advertised one. There is no issue template; when you are reporting a bug, please include the following information.

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* If you can, provide detailed steps to reproduce the bug.
* If you don't have steps to reproduce the bug, just note your observations in as much detail as you can. Questions to start a discussion about the issue are welcome.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" is open to whoever wants to implement it. See [Contributor Setup](#setting-up-the-code-for-local-development) to get started.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "please-help" is open to whoever wants to implement it.

Please do not combine multiple feature enhancements into a single pull request.

See [Contributor Setup](#setting-up-the-code-for-local-development) to get started.

### Write Documentation

WAPER could always use more documentation, whether as part of the official WAPER docs, in docstrings, etc. See [Building the Documentation](#building-the-documentation) for how to build the site locally.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/JoyMonteiro/waper/issues](https://github.com/JoyMonteiro/waper/issues).

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Setting Up the Code for Local Development

Here's how to set up `waper` for local development.

1. Fork the `waper` repo on GitHub.
2. Clone your fork locally:

```bash
git clone git@github.com:JoyMonteiro/waper.git
```

3. Install your local copy into a virtualenv. Assuming you have virtualenv installed, this is how you set up your fork for local development:

```bash
cd waper
virtualenv env --python=python3
source env/bin/activate
pip install -e ".[dev]"
```

4. Install the notebook output filter. This is **per-clone** — git will not apply it
   otherwise, and checkouts will fail with "required filter nbstripout failed":

```bash
nbstripout --install --attributes .gitattributes
```

It keeps notebook outputs in your working copy while stopping them from entering git
history. Notebook output blobs of several MB each have already been committed here once.

5. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the tests and the
   lint/type checks locally. See [Running the Tests](#running-the-tests) for the commands
   — they are the same ones CI runs.

7. Ensure that your feature or commit is covered by tests. To see which lines your change
   left uncovered:

```bash
pytest -m "not slow" --cov --cov-report=term-missing
```

Add `--cov-report=html` for a browsable report in `htmlcov/`. That directory is
gitignored; please do not commit it.

8. Commit your changes and push your branch to GitHub:

```bash
git add -p
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

9. Submit a pull request through the GitHub website.

## Contributor Guidelines

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. The pull request should be contained: if it's too big consider splitting it into smaller pull requests.
3. If the pull request adds functionality, the docs should be updated.
4. The pull request must pass all CI/CD jobs before being ready for review.
5. If one CI/CD job is failing for unrelated reasons you may want to create another PR to fix that first.

### Coding Standards

* Single Responsibility of Units
* Modularity
* Composition over Inheritance


## Running the Tests

The suite is run with `pytest` directly; its configuration lives in `pyproject.toml` under
`[tool.pytest.ini_options]`. Install the development extra first (`pip install -e ".[dev]"`),
which brings in `pytest` and `pytest-cov`.

```bash
pytest -m "not slow"
```

The `slow` tests are deselected because they read large NetCDF files that are gitignored and
absent from a fresh clone. If you have those datasets locally, run the whole suite with
plain `pytest`; anything still missing will skip rather than fail.

Standard pytest selection syntax applies, e.g. to run only tests matching a substring:

```bash
pytest -m "not slow" -k 'smoke_test'
```

For further information please consult the [pytest usage docs](https://docs.pytest.org/en/latest/example/index.html).

### Lint and Type Checks

CI runs these on every push, so run them before you open a pull request:

```bash
ruff check .
mypy
```

Both are configured in `pyproject.toml`. `ruff check --fix .` applies the fixes it can
make automatically.


## Building the Documentation

The docs are a [Quarto](https://quarto.org) site under `docs/`, configured in
`docs/_quarto.yml`. `.github/workflows/docs.yml` builds it on every pull request into
`main` and publishes it to GitHub Pages on every push to `main`.

You need the Quarto CLI, which is not a Python package — install it from
[quarto.org/docs/get-started](https://quarto.org/docs/get-started/). The Python side is
the `docs` extra:

```bash
pip install -e ".[docs]"
```

Then build in two steps, from the repository root:

```bash
cd docs && quartodoc build   # regenerates docs/api/ by importing waper
cd .. && quarto render docs/ # renders the site into docs/_site/
```

Open `docs/_site/index.html` to view the result. Both `docs/api/` and `docs/_site/` are
generated and gitignored — do not commit them. Re-run `quartodoc build` whenever you
change a docstring or add something to the `quartodoc:` sections in `docs/_quarto.yml`;
`quarto render` alone will not pick those up.

The pages that are hand-written are `docs/index.qmd` (the landing page) and
`docs/algorithm.md` (how identification and tracking actually work). Everything under
`docs/api/` comes from the package's own docstrings, so that is where to fix an API
description.


## Core Committer Guide

### Vision and Scope

Core committers, use this section to:

* Guide your instinct and decisions as a core committer
* Limit the codebase from growing infinitely

#### API Accessible

* Modular API striving for statelessness
* Easy to use without having to think too hard
* Flexible for more complex use cases
* Easily extensible

#### Extensible

* Modular Design
* Aim for statelessness


#### Fast and Focused

WAPER is designed to do one thing, and do that one thing very well.

* Cover the important use cases and as little as possible beyond that :)


#### Inclusive

* Cross-platform and cross-version support

#### Stable

* Aim for high test coverage and covering corner cases
* No pull requests will be accepted that drop test coverage on any platform
* Stable APIs that tool builders can rely on


### Process: Pull Requests

How to prioritize pull requests, from most to least important:

* Fixes for broken tests. Broken means broken on any supported platform or Python version.
* Extra tests to cover corner cases.
* Minor edits to docs.
* Bug fixes.
* Major edits to docs.
* Features.

#### Pull Requests Review Guidelines
- Think carefully about the long-term implications of the change. How will it affect existing projects that are dependent on this? If this is complicated, do we really want to maintain it forever?
- Take the time to get things right, PRs almost always require additional improvements to meet the bar for quality. **Be very strict about quality.**
- When you merge a pull request take care of closing/updating every related issue explaining how they were affected by those changes. Credit the author in the merge commit; there is no separate authors file, and the `authors` list in `pyproject.toml` names the authors of the method the package implements, not every contributor.

### Process: Issues

If an issue is a bug that needs an urgent fix, mark it for the next patch release.
Then either fix it or mark as please-help.

For other issues: encourage friendly discussion, moderate debate, offer your thoughts.

### Process: Roadmap

The roadmap located [here](https://github.com/JoyMonteiro/waper/milestones?direction=desc&sort=due_date&state=open)

Due dates are flexible.

### Process: Release:

* Follow semantic versioning. Look at: [http://semver.org](http://semver.org)
