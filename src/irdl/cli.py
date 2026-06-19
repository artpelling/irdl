"""Automagic generation of a Typer script for all dataset downloads."""

import pathlib
import types
from importlib.metadata import version
from inspect import signature
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin

import numpy as np
import pyfar as pf
import typer
from numpydoc.docscrape import FunctionDoc

import irdl
from irdl.base import DatasetCategory, _get_dataset_classes
from irdl.cache import cache_dir as resolve_cache_dir
from irdl.cache import cache_size, clean_cache, format_bytes, prune_cache
from irdl.logging import configure_cli_logging

# Configure CLI logging
configure_cli_logging()


def _resolve_union_type(annotation: type) -> type:
    """Resolve Union types to Typer-compatible types."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle types.UnionType (Python 3.10+) and typing.Union
    if origin is types.UnionType or origin is Union:
        # Filter out NoneType and Path types, keep str
        non_none_args = [arg for arg in args if arg is not type(None)]
        # If we have str and/or Path, use str (paths can be passed as strings in CLI)
        if any(arg is str or arg is pathlib.Path or arg == pathlib.Path for arg in non_none_args):
            # Check if None was in the original args
            if type(None) in args:
                return str | None
            return str
        # For other unions, just take the first type
        if type(None) in args:
            return non_none_args[0] | None
        return non_none_args[0]

    return annotation


def _format_cli_value(value: Any) -> str:
    """Format CLI return values for readable terminal output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        sections = []
        for key, item in value.items():
            formatted = _format_cli_item(item)
            indented = "\n".join(f"  {line}" for line in formatted.splitlines())
            sections.append(f"{key}:\n{indented}")
        return "\n\n".join(sections)
    return str(value)


def _format_cli_item(value: Any) -> str:
    """Format one item inside a CLI result mapping."""
    if isinstance(value, np.ndarray):
        summary = f"ndarray shape={value.shape} dtype={value.dtype}"
        if value.ndim == 0:
            return f"{summary}\n{value.item()}"
        return f"{summary}\n{np.array2string(value, threshold=12, edgeitems=2)}"
    if isinstance(value, pf.Signal | pf.Coordinates):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _make_wrapper(cls, method, params, help_text, dataset_name, param_docs):
    def wrapper(**kwargs) -> Any:
        result = method.__func__(cls, **kwargs)
        if result is not None:
            typer.echo(typer.style(_format_cli_value(result), fg=typer.colors.BRIGHT_CYAN))
        return result

    # Build the signature for the wrapper
    new_params = []
    for name, p in params.items():
        if name == "cls":
            continue
        resolved_type = _resolve_union_type(p.annotation)
        # Don't pass default to typer.Option - it's already in the parameter
        new_params.append(p.replace(annotation=Annotated[resolved_type, typer.Option(help=param_docs.get(name, ""))]))

    wrapper.__signature__ = signature(wrapper).replace(parameters=new_params)
    wrapper.__doc__ = help_text
    wrapper.__name__ = f"{dataset_name}_wrapper"
    return wrapper


#: Typer app that can be invoked by calling ``irdl`` from the CLI.
app = typer.Typer(no_args_is_help=True)
cache_app = typer.Typer(no_args_is_help=True, help="Manage cache directory.")
get_app = typer.Typer(no_args_is_help=True, help="Download datasets.")
app.add_typer(cache_app, name="cache")
app.add_typer(get_app, name="get")


def _get_version() -> str:
    """Get the irdl package version."""
    return version("irdl")


@app.callback(invoke_without_command=True)
def main_callback(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    """Handle version flag."""
    if version:
        version_str = _get_version()
        typer.echo(f"irdl {version_str}")
        raise typer.Exit


@cache_app.callback()
def cache_callback(
    ctx: typer.Context,
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Custom cache directory path."),
) -> None:
    """Cache command callback to handle common options."""
    ctx.obj = {"cache_dir": cache_dir}


def _active_dataset_names() -> set[str]:
    return {dataset_class.name for dataset_class in _get_dataset_classes(irdl)}


@cache_app.command(name="size", help="Show cache directory size.")
def cache_size_command(
    ctx: typer.Context,
    human_readable: bool = typer.Option(False, "-H", "--human-readable", help="Format size with binary units."),
) -> int:
    """Show cache directory size."""
    cache_dir = ctx.obj.get("cache_dir") if ctx.obj else None
    size = cache_size(cache_dir)
    typer.echo(typer.style(format_bytes(size, human_readable=human_readable), fg=typer.colors.BRIGHT_MAGENTA))
    return size


@cache_app.command(name="dir", help="Show cache directory path.")
def cache_dir_command(
    ctx: typer.Context,
) -> Path:
    """Show cache directory path."""
    cache_dir = ctx.obj.get("cache_dir") if ctx.obj else None
    path = resolve_cache_dir(cache_dir)
    typer.echo(typer.style(str(path), fg=typer.colors.BRIGHT_BLUE))
    return path


@cache_app.command(name="clean", help="Wipe cache directory.")
def cache_clean_command(
    ctx: typer.Context,
) -> int:
    """Wipe cache directory."""
    cache_dir = ctx.obj.get("cache_dir") if ctx.obj else None
    root = resolve_cache_dir(cache_dir)
    freed = clean_cache(cache_dir, active_dataset_names=_active_dataset_names())
    typer.echo(f"cleaning cache at: {typer.style(str(root), fg=typer.colors.BRIGHT_BLUE)}")
    typer.echo(
        f"freed disk space: {typer.style(format_bytes(freed, human_readable=True), fg=typer.colors.BRIGHT_MAGENTA)}"
    )
    return freed


@cache_app.command(name="prune", help="Remove unreachable cache items for known datasets.")
def cache_prune_command(
    ctx: typer.Context,
) -> int:
    """Remove unreachable cache items for known datasets only.

    Only processes datasets registered in the current IRDL version.
    Unknown directories in the cache are left untouched.
    """
    cache_dir = ctx.obj.get("cache_dir") if ctx.obj else None
    root = resolve_cache_dir(cache_dir)
    freed = prune_cache(cache_dir, active_dataset_names=_active_dataset_names())
    typer.echo(f"pruning cache at: {typer.style(str(root), fg=typer.colors.BRIGHT_BLUE)}")
    typer.echo(
        f"freed disk space: {typer.style(format_bytes(freed, human_readable=True), fg=typer.colors.BRIGHT_MAGENTA)}"
    )
    return freed


def _get_dataset_description(dataset_class: type) -> str:
    """Get the first line of a dataset class's docstring."""
    docstring = dataset_class.__doc__ or ""
    return docstring.strip().split("\n")[0] if docstring.strip() else ""


def _display_dataset(dataset_class: type) -> None:
    """Display a single dataset with its description, DOI, and providers."""
    typer.echo(f"  {typer.style(dataset_class.name, fg=typer.colors.BRIGHT_CYAN)}")
    description = _get_dataset_description(dataset_class)
    if description:
        typer.echo(f"    {description}")
    doi = getattr(dataset_class, "doi", None)
    if doi:
        typer.echo(f"    DOI: https://doi.org/{doi}")
    providers = getattr(dataset_class, "providers", ())
    canonical_provider = getattr(dataset_class, "canonical_provider", None)
    if providers:
        provider_line = ", ".join(f"{name} (canonical)" if name == canonical_provider else name for name in providers)
        typer.echo(f"    Providers: {provider_line}")


@app.command(name="list", help="List all available datasets.")
def list_datasets() -> None:
    """List all available datasets grouped by category."""
    dataset_classes = _get_dataset_classes(irdl)
    if not dataset_classes:
        typer.echo("No datasets available.")
        return

    # Group datasets by category
    datasets_by_category: dict[DatasetCategory, list[type]] = {}
    uncategorized: list[type] = []

    for dataset_class in dataset_classes:
        category = getattr(dataset_class, "_category", None)
        if category is None:
            uncategorized.append(dataset_class)
        elif category in datasets_by_category:
            datasets_by_category[category].append(dataset_class)
        else:
            datasets_by_category[category] = [dataset_class]

    # Display categorized datasets (in DatasetCategory definition order)
    for category in DatasetCategory:
        if category in datasets_by_category:
            datasets = datasets_by_category[category]
            typer.echo(f"\n{typer.style(category.value.replace('_', ' ').title(), fg=typer.colors.BRIGHT_MAGENTA)}:")
            for dataset_class in sorted(datasets, key=lambda x: x.name):
                _display_dataset(dataset_class)

    # Display uncategorized datasets
    if uncategorized:
        typer.echo(f"\n{typer.style('Uncategorized', fg=typer.colors.BRIGHT_YELLOW)}:")
        for dataset_class in sorted(uncategorized, key=lambda x: x.name):
            _display_dataset(dataset_class)


# Automatically register all supported datasets as subcommands to the get app.
for dataset_class in _get_dataset_classes(irdl):
    get_method = dataset_class.get

    # Get docstring and signature from the classmethod
    doc = FunctionDoc(get_method)
    sig = signature(get_method.__func__)

    # Build Typer parameters with help text from docstring
    param_docs = {p.name: " ".join(p.desc).replace("`", "") for p in doc["Parameters"]}

    # Build help text from docstring
    help_text = doc["Summary"][0] + "\n\n" + " ".join(doc["Extended Summary"])

    wrapper = _make_wrapper(dataset_class, get_method, sig.parameters, help_text, dataset_class.name, param_docs)

    # Register subcommand using dataset_class.name for the command name
    get_app.command(
        name=dataset_class.name,
        help=help_text,
    )(wrapper)
