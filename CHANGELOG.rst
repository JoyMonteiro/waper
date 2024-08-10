=========
Changelog
=========

2022-09-21
=======================================
* Moved most identification code to appropriate files

2022-06-24
=======================================
* Some superficial changes

0.0.1 (2022-06-04)
=======================================

| This is the first ever release of the **my_new_project** Python Package.
| The package is open source and is part of the **My New Project** Project.
| The project is hosted in a public repository on github at https://github.com/john-doe-gh-account-name/my-new-project
| The project was scaffolded using the `Cookiecutter Python Package`_ (cookiecutter) Template at https://github.com/boromir674/cookiecutter-python-package/tree/master/src/cookiecutter_python

| Scaffolding included:

- **CI Pipeline** running on Github Actions at https://github.com/john-doe-gh-account-name/my-new-project/actions
  - `Test Workflow` running a multi-factor **Build Matrix** spanning different `platform`'s and `python version`'s
    1. Platforms: `ubuntu-latest`, `macos-latest`
    2. Python Interpreters: `3.6`, `3.7`, `3.8`, `3.9`, `3.10`

- Automated **Test Suite** with parallel Test execution across multiple cpus.
  - Code Coverage
- **Automation** in a 'make' like fashion, using **tox**
  - Seamless `Lint`, `Type Check`, `Build` and `Deploy` *operations*


.. LINKS

.. _Cookiecutter Python Package: https://python-package-generator.readthedocs.io/en/master/
