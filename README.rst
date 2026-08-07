Waper

A rossby Wave Packet trackER
.. start-badges

| |build| |release_version| |wheel| |gh-lic|

|
| **Code:** https://github.com/JoyMonteiro/waper
| **Docs:** https://joymonteiro.github.io/waper/
| **PyPI:** https://pypi.org/project/waper/
| **CI:** https://github.com/JoyMonteiro/waper/actions/


Features
========

1. **Identification** of Rossby wave packets in a meridional wind field, one time step at a
   time. The method is purely spatial and graph-based: detect alternating maxima and minima,
   cluster them, build an association graph, prune it, and extract RWP paths. No spectral
   envelope is computed at any stage.
2. **Tracking** of the identified packets through time, by the energy-weighted overlap of
   their rasterised footprints between consecutive time steps, so that the association
   follows the energetic cores rather than the full extent of a packet.
3. **Catalogue I/O** (``waper.io``), which serialises the identified and tracked packets and
   supports querying and filtering them.
4. **An interactive explorer** (``waper.interface.explorer.RWPExplorer``), a Panel viewer for
   a catalogue of identified and tracked packets.
5. Implements the method of `Pandey, Monteiro & Natarajan (2020)`_, *An Integrated Geometric
   and Topological Approach for the Identification and Visual Analysis of Rossby Wave
   Packets*, Monthly Weather Review 148(8), 3139–3157.
6. Tested on Python 3.11 and 3.12


Development
-----------
Here are some useful notes related to doing development on this project.

1. **Test Suite**, using `pytest`_, located in `tests` dir
2. **Documentation Pages**, a `Quarto`_ site in the `docs` dir, published to `GitHub Pages`
3. **CI Pipeline**, running on `Github Actions`_, defined in `.github/`

   a. **Job Matrix**, spanning the supported `python version`'s

      1. Platforms: `ubuntu-latest`
      2. Python Interpreters: `3.11`, `3.12`
   b. **Parallel Job** execution, generated from the `matrix`, that runs the `Test Suite`
   c. A separate **lint job** running `ruff` and `mypy`


Prerequisites
=============

You need to have `Python` installed. `waper` requires `Python >= 3.11`; CI tests it on
`Python 3.11` and `3.12`, on Linux.

You will also need the following packages, all of which must be installed from the `conda-forge channel` (in a fresh environment preferably)

* `geovista` (also installs `pyvista`)
* `vtk`
* `xarray`
* `networkx`
* `rasterio`
* `scikit-learn`
* `tqdm`

Preferably install them all in a single command so that `mamba/conda` can figure out the optimal way to resolve dependencies.

Quickstart
==========

`waper` is installed from a source checkout, into the environment where you installed the
prerequisites above.

.. code-block:: sh

    git clone https://github.com/JoyMonteiro/waper.git
    cd waper
    python3 -m pip install -e .

Add the `dev` extra (``pip install -e ".[dev]"``) for the test and lint tooling, or the
`docs` extra to build the documentation site.

Usage
=====

Point `Waper` at a dataset holding a meridional wind field, then identify and track.

.. code-block:: python

    import xarray as xr
    from waper import Waper

    ds = xr.open_dataset("v_winds_300mb.nc")

    w = Waper(
        data_array=ds,
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
    )
    w.identify_rwps()
    w.track_rwps()

`Waper` takes further parameters controlling clipping, extrema detection, the latitude band
searched, and how aggressively the graphs and tracks are pruned. They are listed in the
`API reference`_.


License
=======

|gh-lic|

* Free software: `BSD 3-Clause License`_



.. LINKS

.. _pytest: https://docs.pytest.org/en/7.1.x/

.. _Github Actions: https://github.com/JoyMonteiro/waper/actions

.. _Quarto: https://quarto.org/

.. _API reference: https://joymonteiro.github.io/waper/api/

.. _Pandey, Monteiro & Natarajan (2020): https://doi.org/10.1175/MWR-D-20-0014.1

.. _BSD 3-Clause License: https://github.com/JoyMonteiro/waper/blob/main/LICENSE


.. BADGE ALIASES
..
.. Only badges backed by a service this project actually uses are kept. The
.. Read the Docs, Codecov, Code Climate, `pypi/pyversions` and `commits-since`
.. badges were removed: RTD has no finished builds, CI uploads coverage as a
.. workflow artifact rather than to Codecov, Code Climate is retired at shields,
.. and there are no git tags or GitHub releases to count commits since. The
.. `pyversions` badge reports the interpreters of the last PyPI release (0.0.1),
.. which predates the `requires-python = ">= 3.11"` floor and so misstated it.

.. Build Status
.. Github Actions: Test Workflow Status on the default branch

.. |build| image:: https://img.shields.io/github/actions/workflow/status/JoyMonteiro/waper/test.yaml?branch=main&label=build&logo=github-actions&logoColor=%233392FF
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/JoyMonteiro/waper/actions/workflows/test.yaml?query=branch%3Amain

.. PyPI

.. |release_version| image:: https://img.shields.io/pypi/v/waper
    :alt: Production Version
    :target: https://pypi.org/project/waper/

.. |wheel| image:: https://img.shields.io/pypi/wheel/waper?color=green&label=wheel
    :alt: PyPI - Wheel
    :target: https://pypi.org/project/waper

.. LICENSE (eg AGPL, MIT)
.. Github License

.. |gh-lic| image:: https://img.shields.io/github/license/JoyMonteiro/waper
    :alt: GitHub
    :target: https://github.com/JoyMonteiro/waper/blob/main/LICENSE
