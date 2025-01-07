# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Tuple, Union

import dask.array as da
import pyproj
import xarray as xr

from xcube.util.assertions import assert_true
from .base import GridMapping
from .helpers import _default_xy_dim_names
from .helpers import _default_xy_var_names
from .helpers import _normalize_crs
from .helpers import _normalize_int_pair
from .helpers import _normalize_number_pair
from .helpers import _to_int_or_float


class RegularGridMapping(GridMapping):
    def __init__(self, **kwargs):
        kwargs.pop("is_regular", None)
        super().__init__(is_regular=True, **kwargs)
        self._xy_coords = None

    def _new_x_coords(self) -> xr.DataArray:
        self._assert_regular()
        x_res = self.x_res
        x1, x2 = self.x_min + x_res / 2, self.x_max - x_res / 2
        x_name, _ = self.xy_dim_names
        return xr.DataArray(
            da.linspace(x1, x2, self.width, chunks=self.tile_width),
            dims=self.xy_dim_names[0],
        )

    def _new_y_coords(self) -> xr.DataArray:
        self._assert_regular()
        y_res = self.y_res
        y1, y2 = self.y_min + y_res / 2, self.y_max - y_res / 2
        if not self.is_j_axis_up:
            y1, y2 = y2, y1
        return xr.DataArray(
            da.linspace(y1, y2, self.height, chunks=self.tile_height),
            dims=self.xy_dim_names[1],
        )

    def _new_xy_coords(self) -> xr.DataArray:
        self._assert_regular()
        x_coords_1d = da.asarray(self.x_coords.data).rechunk(self.tile_width)
        y_coords_1d = da.expand_dims(
            da.asarray(self.y_coords.data).rechunk(self.tile_height), 1
        )
        y_coords_2d, x_coords_2d = da.broadcast_arrays(y_coords_1d, x_coords_1d)
        xy_coords = da.concatenate(
            [da.expand_dims(x_coords_2d, 0), da.expand_dims(y_coords_2d, 0)]
        )
        xy_coords = da.rechunk(
            xy_coords, chunks=(2, xy_coords.chunksize[1], xy_coords.chunksize[2])
        )
        xy_coords = xr.DataArray(
            xy_coords,
            dims=("coord", self.y_coords.dims[0], self.x_coords.dims[0]),
            name="xy_coords",
        )
        xy_coords.name = "xy_coords"
        return xy_coords


def new_regular_grid_mapping(
    size: Union[int, tuple[int, int]],
    xy_min: tuple[float, float],
    xy_res: Union[float, tuple[float, float]],
    crs: Union[str, pyproj.crs.CRS],
    *,
    tile_size: Union[int, tuple[int, int]] = None,
    is_j_axis_up: bool = False,
) -> GridMapping:
    width, height = _normalize_int_pair(size, name="size")
    assert_true(width > 1 and height > 1, "invalid size")

    x_min, y_min = _normalize_number_pair(xy_min, name="xy_min")

    x_res, y_res = _normalize_number_pair(xy_res, name="xy_res")
    assert_true(x_res > 0 and y_res > 0, "invalid xy_res")

    crs = _normalize_crs(crs)

    x_min = _to_int_or_float(x_min)
    y_min = _to_int_or_float(y_min)
    x_max = _to_int_or_float(x_min + x_res * width)
    y_max = _to_int_or_float(y_min + y_res * height)

    if crs.is_geographic:
        # TODO: don't do that.
        #  Instead set NaN in coord vars returned by to_coords()
        if y_min < -90:
            raise ValueError("invalid y_min")
        if y_max > 90:
            raise ValueError("invalid size, y_min combination")

    return RegularGridMapping(
        crs=crs,
        size=(width, height),
        tile_size=tile_size or (width, height),
        xy_bbox=(x_min, y_min, x_max, y_max),
        xy_res=(x_res, y_res),
        xy_var_names=_default_xy_var_names(crs),
        xy_dim_names=_default_xy_dim_names(crs),
        is_lon_360=(x_max > 180) and crs.is_geographic,
        is_j_axis_up=is_j_axis_up,
    )


def to_regular_grid_mapping(
    grid_mapping: GridMapping,
    *,
    tile_size: Union[int, tuple[int, int]] = None,
    is_j_axis_up: bool = False,
) -> GridMapping:
    if grid_mapping.is_regular:
        if tile_size is not None or is_j_axis_up != grid_mapping.is_j_axis_up:
            return grid_mapping.derive(tile_size=tile_size, is_j_axis_up=is_j_axis_up)
        return grid_mapping

    x_min, y_min, x_max, y_max = grid_mapping.xy_bbox
    x_res, y_res = grid_mapping.xy_res
    # x_digits = 2 + abs(round(math.log10(x_res)))
    # y_digits = 2 + abs(round(math.log10(y_res)))
    # x_min = floor_to_fraction(x_min, x_digits)
    # y_min = floor_to_fraction(y_min, y_digits)
    # x_max = ceil_to_fraction(x_max, x_digits)
    # y_max = ceil_to_fraction(y_max, y_digits)
    xy_res = min(x_res, y_res) or max(x_res, y_res)
    width = round((x_max - x_min + xy_res) / xy_res)
    height = round((y_max - y_min + xy_res) / xy_res)
    width = width if width >= 2 else 2
    height = height if height >= 2 else 2

    if tile_size is None:
        tile_size = grid_mapping.tile_size
    return new_regular_grid_mapping(
        size=(width, height),
        xy_min=(x_min, y_min),
        xy_res=xy_res,
        crs=grid_mapping.crs,
        tile_size=tile_size,
        is_j_axis_up=is_j_axis_up,
    )
