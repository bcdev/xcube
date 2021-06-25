===============
``xcube prune``
===============

Delete empty chunks.

.. attention:: This tool will likely be integrated into ``xcube optimize`` in the near future.


::

    $ xcube prune --help

::

    Usage: xcube prune [OPTIONS] DATASET

      Delete empty chunks. Deletes all data files associated with empty (NaN-
      only) chunks in given DATASET, which must have ZARR format.

    Options:
      -v, --verbose  Verbose mode. Multiple may be given, for example "-vvv".
      --dry-run      Just read and process input, but don't produce any outputs.
      --help         Show this message and exit.


A related Python API function is :py:func:`xcube.core.optimize.get_empty_dataset_chunks`.
