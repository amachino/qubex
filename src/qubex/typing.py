from typing import Mapping, TypeVar

T_co = TypeVar("T_co", covariant=True)
TargetMap = Mapping[str, T_co]
