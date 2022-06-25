Waper

A rossby Wave Packet trackER
.. start-badges

| |build| |docs| |coverage| |maintainability| |better_code_hub| |tech-debt|
| |release_version| |wheel| |supported_versions| |gh-lic| |commits_since_specific_tag_on_master| |commits_since_latest_github_release|

|
| **Code:** https://github.com/JoyMonteiro/waper
| **Docs:** https://waper.readthedocs.io/en/master/
| **PyPI:** https://pypi.org/project/waper/
| **CI:** https://github.com/JoyMonteiro/waper/actions/


Features
========

1. **my_new_project** `python package`

   a. TODO Document a **Great Feature**
   b. TODO Document another **Nice Feature**
2. Tested against multiple `platforms` and `python` versions


Development
-----------
Here are some useful notes related to doing development on this project.

1. **Test Suite**, using `pytest`_, located in `tests` dir
2. **Parallel Execution** of Unit Tests, on multiple cpu's
3. **Documentation Pages**, hosted on `readthedocs` server, located in `docs` dir
4. **Automation**, using `tox`_, driven by single `tox.ini` file

   a. **Code Coverage** measuring
   b. **Build Command**, using the `build`_ python package
   c. **Pypi Deploy Command**, supporting upload to both `pypi.org`_ and `test.pypi.org`_ servers
   d. **Type Check Command**, using `mypy`_
   e. **Lint** *Check* and `Apply` commands, using `isort`_ and `black`_
5. **CI Pipeline**, running on `Github Actions`_, defined in `.github/`

   a. **Job Matrix**, spanning different `platform`'s and `python version`'s

      1. Platforms: `ubuntu-latest`, `macos-latest`
      2. Python Interpreters: `3.6`, `3.7`, `3.8`, `3.9`, `3.10`
   b. **Parallel Job** execution, generated from the `matrix`, that runs the `Test Suite`


Prerequisites
=============

You need to have `Python` installed.

Quickstart
==========

Using `pip` is the approved way for installing `my_new_project`.

.. code-block:: sh

    python3 -m pip install my_new_project


TODO Document a use case


License
=======

|gh-lic|

* `BSD 3-Clause License`_


License
=======

* Free software: BSD 3-Clause License



.. LINKS

.. _tox: https://tox.wiki/en/latest/

.. _pytest: https://docs.pytest.org/en/7.1.x/

.. _build: https://github.com/pypa/build

.. _pypi.org: https://pypi.org/

.. _test.pypi.org: https://test.pypi.org/

.. _mypy: https://mypy.readthedocs.io/en/stable/

.. _isort: https://pycqa.github.io/isort/

.. _black: https://black.readthedocs.io/en/stable/

.. _Github Actions: https://github.com/JoyMonteiro/waper/actions

.. _BSD 3-Clause License: https://github.com/JoyMonteiro/waper/blob/master/LICENSE


.. BADGE ALIASES

.. Build Status
.. Github Actions: Test Workflow Status for specific branch <branch>

.. |build| image:: https://img.shields.io/github/workflow/status/JoyMonteiro/waper/Test%20Python%20Package/master?label=build&logo=github-actions&logoColor=%233392FF
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/JoyMonteiro/waper/actions/workflows/test.yaml?query=branch%3Amaster


.. Documentation

.. |docs| image:: https://img.shields.io/readthedocs/waper/master?logo=readthedocs&logoColor=lightblue
    :alt: Read the Docs (version)
    :target: https://waper.readthedocs.io/en/master/

.. Code Coverage

.. |coverage| image:: https://img.shields.io/codecov/c/github/JoyMonteiro/waper/master?logo=codecov
    :alt: Codecov
    :target: https://app.codecov.io/gh/JoyMonteiro/waper

.. PyPI

.. |release_version| image:: https://img.shields.io/pypi/v/my_new_project
    :alt: Production Version
    :target: https://pypi.org/project/my_new_project/

.. |wheel| image:: https://img.shields.io/pypi/wheel/waper?color=green&label=wheel
    :alt: PyPI - Wheel
    :target: https://pypi.org/project/my_new_project

.. |supported_versions| image:: https://img.shields.io/pypi/pyversions/waper?color=blue&label=python&logo=python&logoColor=%23ccccff
    :alt: Supported Python versions
    :target: https://pypi.org/project/my_new_project

.. Github Releases & Tags

.. |commits_since_specific_tag_on_master| image:: https://img.shields.io/github/commits-since/JoyMonteiro/waper/v0.0.1/master?color=blue&logo=github
    :alt: GitHub commits since tagged version (branch)
    :target: https://github.com/JoyMonteiro/waper/compare/v0.0.1..master

.. |commits_since_latest_github_release| image:: https://img.shields.io/github/commits-since/JoyMonteiro/waper/latest?color=blue&logo=semver&sort=semver
    :alt: GitHub commits since latest release (by SemVer)

.. LICENSE (eg AGPL, MIT)
.. Github License

.. |gh-lic| image:: https://img.shields.io/github/license/JoyMonteiro/waper
    :alt: GitHub
    :target: https://github.com/JoyMonteiro/waper/blob/master/LICENSE


.. CODE QUALITY

.. Better Code Hub
.. Software Design Patterns

.. |better_code_hub| image:: https://bettercodehub.com/edge/badge/JoyMonteiro/waper?branch=master
    :alt: Better Code Hub
    :target: https://bettercodehub.com/


.. Code Climate CI
.. Code maintainability & Technical Debt

.. |maintainability| image:: https://img.shields.io/codeclimate/maintainability/JoyMonteiro/waper
    :alt: Code Climate Maintainability
    :target: https://codeclimate.com/github/JoyMonteiro/waper/maintainability

.. |tech-debt| image:: https://img.shields.io/codeclimate/tech-debt/JoyMonteiro/waper
    :alt: Technical Debt
    :target: https://codeclimate.com/github/JoyMonteiro/waper/maintainability
