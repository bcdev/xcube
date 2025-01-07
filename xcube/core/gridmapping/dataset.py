# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional, Union, Tuple
import warnings

import pyproj
import xarray as xr

from .base import DEFAULT_TOLERANCE
from .base import GridMapping
from .cfconv import get_dataset_grid_mapping_proxies
from .coords import new_grid_mapping_from_coords
from .helpers import _normalize_crs


def new_grid_mapping_from_dataset(
    dataset: xr.Dataset,
    *,
    crs: Union[str, pyproj.crs.CRS] = None,
    tile_size: Union[int, tuple[str, str]] = None,
    prefer_crs: Union[str, pyproj.crs.CRS] = None,
    prefer_is_regular: bool = None,
    emit_warnings: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Optional[GridMapping]:
    # Note `crs` is used if CRS is known in advance,
    # so the code forces its use. `prefer_crs` is used if
    # multiple CRSes are found, and a preference exists.
    # If it is not given, but `crs` is given,
    # then they are the same.
    if crs is not None:
        crs = _normalize_crs(crs)
    if prefer_crs is not None:
        prefer_crs = _normalize_crs(prefer_crs)
    else:
        prefer_crs = crs

    grid_mapping_proxies = get_dataset_grid_mapping_proxies(
        dataset,
        emit_warnings=emit_warnings,
        missing_projected_crs=crs,
        missing_rotated_latitude_longitude_crs=crs,
        missing_latitude_longitude_crs=crs,
    ).values()

    grid_mappings = [
        new_grid_mapping_from_coords(
            x_coords=gmp.coords.x,
            y_coords=gmp.coords.y,
            crs=gmp.crs,
            tile_size=tile_size or gmp.tile_size,
            tolerance=tolerance,
        )
        for gmp in grid_mapping_proxies
    ]

    if len(grid_mappings) > 1:
        if prefer_crs is not None and prefer_is_regular is not None:
            for gm in grid_mappings:
                if gm.crs == prefer_crs and bool(gm.is_regular) == prefer_is_regular:
                    return gm
            for gm in grid_mappings:
                if (
                    gm.crs.is_geographic
                    and prefer_crs.is_geographic
                    and bool(gm.is_regular) == prefer_is_regular
                ):
                    return gm

        if prefer_crs is not None:
            for gm in grid_mappings:
                if gm.crs == prefer_crs:
                    return gm
            for gm in grid_mappings:
                if gm.crs.is_geographic and prefer_crs.is_geographic:
                    return gm

        if prefer_is_regular is not None:
            for gm in grid_mappings:
                if bool(gm.is_regular) == prefer_is_regular:
                    return gm

    # Get arbitrary one (here: first)
    if grid_mappings:
        return grid_mappings[0]

    raise ValueError("cannot find any grid mapping in dataset")
