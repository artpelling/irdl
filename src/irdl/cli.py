"""Automatic generation of a Typer script for all datasets that can be used for download."""

from inspect import signature
from typing import Annotated

import typer
from numpydoc.docscrape import FunctionDoc

import irdl

# Typer app that can be invoked by calling ``irdl`` from the CLI.
app = typer.Typer(no_args_is_help=True)

# Automatically register all supported datasets as subcommands to the app.
for get_dataset in [getattr(irdl, d) for d in irdl.__all__]:
    # get function docstring and signature
    doc = FunctionDoc(get_dataset)
    sig = signature(get_dataset)
    typer_parameters = [
        p.replace(annotation=Annotated[p.annotation, typer.Option(help=" ".join(d.desc).replace("`", ""))])
        for p, d in zip(sig.parameters.values(), doc["Parameters"], strict=True)
    ]
    get_dataset.__signature__ = sig.replace(parameters=typer_parameters)
    # register a subcommand to the main app and add --help information based on the docstring.
    app.command(
        name=get_dataset.__name__.removeprefix("get_"),
        help=doc["Summary"][0] + "\n\n" + " ".join(doc["Extended Summary"]),
    )(get_dataset)

# expose click object for sphinx_click autodoc feature.
typer_click_object = typer.main.get_command(app)
