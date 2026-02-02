"""Expression utilities for model evaluation."""

from collections.abc import Mapping
from typing import Any, cast

from sympy import Symbol, lambdify, parse_expr


class Expression:
    """
    Parse and evaluate symbolic expressions.

    Parameters
    ----------
    string
        Expression string to parse.
    symbol_dict
        Optional symbol table used during parsing.
    modules
        Backend modules passed to `sympy.lambdify`.
    """

    def __init__(
        self,
        string: str,
        symbol_dict: dict[str, Symbol] | None = None,
        modules: str | list[str] | None = "numpy",
    ) -> None:
        """
        Initialize an expression evaluator.

        Parameters
        ----------
        string
            Expression string to parse.
        symbol_dict
            Optional symbol table used during parsing.
        modules
            Backend modules passed to `sympy.lambdify`.

        Raises
        ------
        ValueError
            If the expression string cannot be parsed.
        """
        try:
            self._expr = parse_expr(string, local_dict=symbol_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to parse expression string '{string}': {e}"
            ) from e
        self._symbols = tuple(
            # Sort symbols by name to ensure deterministic order for lambdify
            sorted(
                (cast(Symbol, s) for s in self._expr.free_symbols),
                key=lambda s: s.name,
            )
        )
        self._symbol_names = [s.name for s in self._symbols]
        self._func = lambdify(self._symbols, self._expr, modules=modules)

    def _repr_latex_(self) -> str:
        """Return LaTeX representation for rich display."""
        return self._expr._repr_latex_()

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """Symbols appearing in the expression, sorted by name."""
        return self._symbols

    def resolve(self, params: Mapping[str, Any]) -> Any:
        """
        Evaluate the expression with provided parameter values.

        Parameters
        ----------
        params : Mapping[str, Any]
            Mapping from symbol names to values.

        Returns
        -------
        Any
            Evaluated value.

        Raises
        ------
        ValueError
            If any required symbol is missing from `params`.
        """
        try:
            values = [params[name] for name in self._symbol_names]
        except KeyError as e:
            raise ValueError(
                f"Value for symbol '{e.args[0]}' not provided in params."
            ) from None
        return self._func(*values)
