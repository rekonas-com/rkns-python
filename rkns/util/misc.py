from __future__ import annotations

import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T", bound=type)


def cached_import(module_path: str, class_name: str) -> Any:
    """
    Import a module and return the specified class or attribute.

    From Django: https://docs.djangoproject.com/en/5.1/ref/utils/#module-django.utils.module_loading

    Parameters
    ----------
    module_path : str
        The path of the module to import.
    class_name : str
        The name of the class or attribute to retrieve from the module.

    Returns
    -------
    Any
        The specified class or attribute from the module.
    """
    # Check whether module is loaded and fully initialized.
    if not (
        (module := sys.modules.get(module_path))
        and (spec := getattr(module, "__spec__", None))
        and getattr(spec, "_initializing", False) is False
    ):
        module = import_module(module_path)
    return getattr(module, class_name)


def import_from_string(dotted_path: str) -> Any:
    """
    Import a dotted module path and return the designated class.
    From Django: https://docs.djangoproject.com/en/5.1/ref/utils/#module-django.utils.module_loading

    Parameters
    ----------
    dotted_path : str
        The dotted path of the module and class to import.

    Returns
    -------
    Any
        The designated class from the module.

    Raises
    ------
    ImportError
        If the dotted path is not a valid module path or if the module does not
        define the specified attribute or class.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute/class'
            % (module_path, class_name)
        ) from err


def check_open(method: Callable):
    """
    Decorator to check if the RKNS object is closed before executing a method

    Parameters
    ----------
    method
        _description_
    """

    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_is_closed") and self._is_closed:
            raise RuntimeError(
                f"Cannot execute {method.__name__}: RKNS object has been closed"
            )
        return method(self, *args, **kwargs)

    return wrapper


def apply_check_open_to_all_methods(cls: T) -> T:
    """
    Apply the @check_open decorator to all methods of a class.
    That is, it checks if the instance is closed before executing a method.
    """
    for name, method in cls.__dict__.items():
        if (
            callable(method)
            and not name.startswith("__")
            and not isinstance(method, (staticmethod, classmethod))
        ):
            # Skip staticmethods and classmethods
            setattr(cls, name, check_open(method))
    return cls
