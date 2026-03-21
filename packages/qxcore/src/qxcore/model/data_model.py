"""NetCDF-capable base model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typing_extensions import Self

from qxcore.serialization import load_netcdf_file, save_netcdf_file

from .model import Model


class DataModel(Model):
    """Base model that serializes array data to NetCDF variables."""

    def save_netcdf(self, path: str | Path) -> Path:
        """
        Save this model as a NetCDF file.

        Parameters
        ----------
        path : str | Path
            Output `.nc` path.

        Returns
        -------
        Path
            Saved path.
        """
        return save_netcdf_file(
            model=self,
            data=self._raw_field_data(),
            path=path,
        )

    @classmethod
    def load_netcdf(cls, path: str | Path) -> Self:
        """
        Load this model from a NetCDF file.

        Parameters
        ----------
        path : str | Path
            Input `.nc` path.

        Returns
        -------
        Self
            Restored model instance.
        """
        data = load_netcdf_file(model_cls=cls, path=path)
        return cls.model_validate(data)

    def _raw_field_data(self) -> dict[str, Any]:
        """
        Return raw model fields without custom serializer transformation.

        Returns
        -------
        dict[str, Any]
            Field values mapped by field name.
        """
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__class__.model_fields
        }
