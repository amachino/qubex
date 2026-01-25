from typing import Any, cast

from sympy import Expr, Symbol, lambdify, parse_expr


class Variable:
    def __init__(
        self,
        expr: str,
        symbols: dict[str, Symbol] | None = None,
        modules: str | list[str] | None = "numpy",
    ) -> None:
        try:
            self._expr = parse_expr(expr, local_dict=symbols)
        except Exception as e:
            raise ValueError(f"Failed to parse expression string '{expr}': {e}") from e
        self._symbols = tuple(
            # Sort symbols by name to ensure deterministic order for lambdify
            sorted(
                (cast(Symbol, s) for s in self._expr.free_symbols),
                key=lambda s: s.name,
            )
        )
        self._symbol_names = [s.name for s in self._symbols]
        self._func = lambdify(self._symbols, self._expr, modules=modules)

    def __repr__(self) -> str:
        return f"Variable(expr='{self._expr}')"

    def _repr_latex_(self):
        return self._expr._repr_latex_()

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return self._symbols

    def resolve(self, value_map: dict[str, Any]) -> Any:
        try:
            values = [value_map[name] for name in self._symbol_names]
        except KeyError as e:
            raise ValueError(
                f"Value for symbol '{e.args[0]}' not provided in value_map."
            ) from None
        return self._func(*values)
