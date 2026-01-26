from inspect import getmodule, signature
from typing import Annotated

import typer
from numpydoc.docscrape import FunctionDoc

from irdl import get_fabian, get_miracle

app = typer.Typer(no_args_is_help=True)

for get_dataset in (get_fabian, get_miracle):
    doc = FunctionDoc(get_dataset)
    sig = signature(get_dataset)
    typer_parameters = [
        p.replace(annotation=Annotated[p.annotation, typer.Option(help=" ".join(d.desc))])
        for p, d in zip(sig.parameters.values(), doc["Parameters"], strict=True)
    ]
    get_dataset.__signature__ = sig.replace(parameters=typer_parameters)
    app.command(
        name=getmodule(get_dataset).__name__.split(".")[1],
        help=doc["Summary"][0] + "\n\n" + " ".join(doc["Extended Summary"]),
    )(get_dataset)
