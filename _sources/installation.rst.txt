Installation
============

We recommend either ``uv`` or ``pip`` with ``venv`` for installation. First, create a
virtual environment:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv venv

        .. note::
           ``uv`` will handle environment activation implicitly when you use ``uv run``.

    .. tab-item:: ``venv``
        :sync: pip

        .. code-block:: console

            $ python3 -m venv my-env

        Then activate the environment:

        .. code-block:: console

            $ source my-env/bin/activate

Then install ``irdl`` into that environment:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv pip install irdl

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ pip install -U irdl

Check the installation by asking the CLI for help:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv run irdl --help

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ irdl --help

.. tip::

    If you want the ``irdl`` command to be available globally, install it as a ``uv tool``:

    .. code-block:: console

       $ uv tool install irdl

    Upgrade the global tool later with:

    .. code-block:: console

       $ uv tool upgrade irdl
