# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import abc
import math
from typing import Union

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from .base import DEFAULT_TOLERANCE
from .base import GridMapping
from .helpers import _assert_valid_xy_names
from .helpers import _default_xy_var_names
from .helpers import _normalize_crs
from .helpers import _normalize_int_pair
from .helpers import _normalize_number_pair
from .helpers import Number
from .helpers import _to_int_or_float
from .helpers import from_lon_360
from .helpers import round_to_fraction
from .helpers import to_lon_360

_ER = 6371000


class CoordsGridMapping(GridMapping, abc.ABC):
    """
    Grid mapping constructed from 1D/2D coordinate
    variables and a CRS.
    """

    @property
    def x_coords(self):
        assert isinstance(self._x_coords, xr.DataArray)
        return self._x_coords

    @property
    def y_coords(self):
        assert isinstance(self._y_coords, xr.DataArray)
        return self._y_coords

    def _new_x_coords(self) -> xr.DataArray:
        # Should never come here
        return self._x_coords

    def _new_y_coords(self) -> xr.DataArray:
        # Should never come here
        return self._y_coords


class Coords1DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from
    1D coordinate variables and a CRS.
    """

    def _new_xy_coords(self) -> xr.DataArray:
        y, x = xr.broadcast(self._y_coords, self._x_coords)
        tmp = xr.concat([x, y], dim="coord")
        return tmp.chunk(
            {dim: size for (dim, size) in zip(tmp.dims, self.xy_coords_chunks)}
        )


class Coords2DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from
    2D coordinate variables and a CRS.
    """

    def _new_xy_coords(self) -> xr.DataArray:
        tmp = xr.concat([self._x_coords, self._y_coords], dim="coord")
        return tmp.chunk(
            {dim: size for (dim, size) in zip(tmp.dims, self.xy_coords_chunks)}
        )


def new_grid_mapping_from_coords(
    x_coords: xr.DataArray,
    y_coords: xr.DataArray,
    crs: Union[str, pyproj.crs.CRS],
    *,
    xy_res: Union[Number, tuple[Number, Number]] = None,
    xy_bbox: tuple[Number, Number, Number, Number] = None,
    tile_size: Union[int, tuple[int, int]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GridMapping:
    crs = _normalize_crs(crs)
    assert_instance(x_coords, xr.DataArray, name="x_coords")
    assert_instance(y_coords, xr.DataArray, name="y_coords")
    assert_true(
        x_coords.ndim in (1, 2), "x_coords and y_coords must be either 1D or 2D arrays"
    )
    assert_instance(tolerance, float, name="tolerance")
    assert_true(tolerance > 0.0, "tolerance must be greater zero")

    if x_coords.name and y_coords.name:
        xy_var_names = str(x_coords.name), str(y_coords.name)
    else:
        xy_var_names = _default_xy_var_names(crs)

    tile_size = _normalize_int_pair(tile_size, default=None)
    is_lon_360 = None  # None means "not yet known"
    if crs.is_geographic:
        is_lon_360 = bool(da.any(x_coords > 180))

    if x_coords.ndim == 1:
        # We have 1D x,y coordinates
        cls = Coords1DGridMapping

        assert_true(
            x_coords.size >= 2 and y_coords.size >= 2,
            "sizes of x_coords and y_coords 1D arrays must be >= 2",
        )

        size = x_coords.size, y_coords.size

        x_dim, y_dim = x_coords.dims[0], y_coords.dims[0]

        x_diff = _abs_no_zero(x_coords.diff(dim=x_dim).values)
        y_diff = _abs_no_zero(y_coords.diff(dim=y_dim).values)

        if not is_lon_360 and crs.is_geographic:
            is_anti_meridian_crossed = np.any(np.nanmax(x_diff) > 180)
            if is_anti_meridian_crossed:
                x_coords = to_lon_360(x_coords)
                x_diff = _abs_no_zero(x_coords.diff(dim=x_dim))
                is_lon_360 = True

        if xy_res is not None:
            x_res, y_res = _normalize_number_pair(xy_res)
        else:
            x_res = x_diff[0]
            y_res = y_diff[0]
            is_regular = da.allclose(x_diff, x_res, atol=tolerance) and da.allclose(
                y_diff, y_res, atol=tolerance
            )
            if is_regular:
                x_res = round_to_fraction(float(x_res), 5, 0.25)
                y_res = round_to_fraction(float(y_res), 5, 0.25)
            else:
                x_res = round_to_fraction(float(np.nanmedian(x_diff, axis=0)), 2, 0.5)
                y_res = round_to_fraction(float(np.nanmedian(y_diff, axis=0)), 2, 0.5)

        if (
            tile_size is None
            and x_coords.chunks is not None
            and y_coords.chunks is not None
        ):
            tile_size = (max(0, *x_coords.chunks[0]), max(0, *y_coords.chunks[0]))

        # Guess j axis direction
        is_j_axis_up = bool(y_coords[0] < y_coords[-1])

    else:
        # We have 2D x,y coordinates
        cls = Coords2DGridMapping

        assert_true(
            x_coords.shape == y_coords.shape,
            "shapes of x_coords and y_coords" " 2D arrays must be equal",
        )
        assert_true(
            x_coords.dims == y_coords.dims,
            "dimensions of x_coords and y_coords" " 2D arrays must be equal",
        )

        y_dim, x_dim = x_coords.dims

        height, width = x_coords.shape
        size = width, height

        x = da.asarray(x_coords)
        y = da.asarray(y_coords)

        x_x_diff = _abs_no_nan(da.diff(x[0, : x.chunksize[1]]))
        x_y_diff = _abs_no_nan(da.diff(x[: x.chunksize[0], 0]))
        y_x_diff = _abs_no_nan(da.diff(y[0, : x.chunksize[0]]))
        y_y_diff = _abs_no_nan(da.diff(y[: x.chunksize[1], 0]))

        if not is_lon_360 and crs.is_geographic:
            is_anti_meridian_crossed = da.any(da.max(x_x_diff) > 180) or da.any(
                da.max(x_y_diff) > 180
            )
            if is_anti_meridian_crossed:
                x_coords = to_lon_360(x_coords)
                x = da.asarray(x_coords)
                x_x_diff = _abs_no_nan(da.diff(x[0, :]))
                x_y_diff = _abs_no_nan(da.diff(x[:, 0]))
                is_lon_360 = True

        if xy_res is not None:
            x_res, y_res = _normalize_number_pair(xy_res)
        else:
            x_res = x_x_diff[0]
            y_res = y_y_diff[0]

        is_regular = (
            da.allclose(x_x_diff, x_res, atol=tolerance)
            and da.allclose(y_y_diff, y_res, atol=tolerance)
            and da.allclose(x_y_diff, 0, atol=tolerance)
            and da.allclose(y_x_diff, 0, atol=tolerance)
        )

        if not is_regular and xy_res is None:
            x_x_diff = _abs_no_nan(da.diff(x, axis=1))
            x_y_diff = _abs_no_nan(da.diff(x, axis=0))
            y_x_diff = _abs_no_nan(da.diff(y, axis=1))
            y_y_diff = _abs_no_nan(da.diff(y, axis=0))

            if not is_lon_360 and crs.is_geographic:
                is_anti_meridian_crossed = da.any(da.max(x_x_diff) > 180) or da.any(
                    da.max(x_y_diff) > 180
                )
                if is_anti_meridian_crossed:
                    x_coords = to_lon_360(x_coords)
                    x = da.asarray(x_coords)
                    y_x_diff = _abs_no_nan(da.diff(y, axis=1))
                    y_y_diff = _abs_no_nan(da.diff(y, axis=0))
                    is_lon_360 = True

            # Let diff arrays have same shape as original by
            # doubling last rows and columns.
            x_x_diff_c = da.concatenate([x_x_diff, x_x_diff[:, -1:]], axis=1)
            y_x_diff_c = da.concatenate([y_x_diff, y_x_diff[:, -1:]], axis=1)
            x_y_diff_c = da.concatenate([x_y_diff, x_y_diff[-1:, :]], axis=0)
            y_y_diff_c = da.concatenate([y_y_diff, y_y_diff[-1:, :]], axis=0)
            # Find resolution via area
            x_abs_diff = da.sqrt(da.square(x_x_diff_c) + da.square(x_y_diff_c))
            y_abs_diff = da.sqrt(da.square(y_x_diff_c) + da.square(y_y_diff_c))
            if crs.is_geographic:
                # Convert degrees into meters
                x_abs_diff_r = da.radians(x_abs_diff)
                y_abs_diff_r = da.radians(y_abs_diff)
                x_abs_diff = _ER * da.cos(x_abs_diff_r) * y_abs_diff_r
                y_abs_diff = _ER * y_abs_diff_r
            xy_areas = (x_abs_diff * y_abs_diff).flatten()
            xy_areas = da.where(xy_areas > 0, xy_areas, np.nan)
            # Get indices of min and max area
            xy_area_index_min = da.nanargmin(xy_areas)
            xy_area_index_max = da.nanargmax(xy_areas)
            # Convert area to edge length
            xy_res_min = math.sqrt(xy_areas[xy_area_index_min])
            xy_res_max = math.sqrt(xy_areas[xy_area_index_max])
            # Empirically weight min more than max
            xy_res = 0.7 * xy_res_min + 0.3 * xy_res_max
            if crs.is_geographic:
                # Convert meters back into degrees
                # print(f'xy_res in meters: {xy_res}')
                xy_res = math.degrees(xy_res / _ER)
                # print(f'xy_res in degrees: {xy_res}')
            # Because this is an estimation, we can round to a nice number
            xy_res = round_to_fraction(xy_res, digits=1, resolution=0.5)
            x_res, y_res = float(xy_res), float(xy_res)

        if tile_size is None and x_coords.chunks is not None:
            j_chunks, i_chunks = x_coords.chunks
            tile_size = max(0, *i_chunks), max(0, *j_chunks)

        if tile_size is not None:
            tile_width, tile_height = tile_size
            x_coords = x_coords.chunk(
                {
                    x_coords.dims[0]: tile_height,
                    x_coords.dims[1]: tile_width,
                }
            )
            y_coords = y_coords.chunk(
                {
                    y_coords.dims[0]: tile_height,
                    y_coords.dims[1]: tile_width,
                }
            )

        # Guess j axis direction
        is_j_axis_up = da.all(
            y_coords[0, : y.chunksize[1]] < y_coords[-1, : y.chunksize[1]]
        )

    assert_true(
        x_res > 0 and y_res > 0,
        "internal error: x_res and y_res could not be determined",
        exception_type=RuntimeError,
    )

    x_res, y_res = _to_int_or_float(x_res), _to_int_or_float(y_res)
    if xy_bbox is None:
        x_res_05, y_res_05 = x_res / 2, y_res / 2
        x_min = _to_int_or_float(x_coords[..., 0].min() - x_res_05)
        x_max = _to_int_or_float(x_coords[..., -1].max() + x_res_05)
        if is_j_axis_up:
            y_min = _to_int_or_float(float(y_coords[0, ...].min()) - y_res_05)
            y_max = _to_int_or_float(float(y_coords[-1, ...].max()) + y_res_05)
        else:
            y_min = _to_int_or_float(float(y_coords[-1, ...].min()) - y_res_05)
            y_max = _to_int_or_float(float(y_coords[0, ...].max()) + y_res_05)
        xy_bbox = (x_min, y_min, x_max, y_max)

    if cls is Coords1DGridMapping and is_regular:
        from .regular import RegularGridMapping

        cls = RegularGridMapping

    return cls(
        x_coords=x_coords,
        y_coords=y_coords,
        crs=crs,
        size=size,
        tile_size=tile_size,
        xy_bbox=xy_bbox,
        xy_res=(x_res, y_res),
        xy_var_names=xy_var_names,
        xy_dim_names=(str(x_dim), str(y_dim)),
        is_regular=is_regular,
        is_lon_360=is_lon_360,
        is_j_axis_up=is_j_axis_up,
    )


def _abs_no_zero(array: Union[xr.DataArray, da.Array, np.ndarray]):
    array = da.fabs(array)
    return da.where(da.isclose(array, 0), np.nan, array)


def _abs_no_nan(array: Union[da.Array, np.ndarray]):
    array = da.fabs(array)
    return da.where(da.logical_or(da.isnan(array), da.isclose(array, 0)), 0, array)


def grid_mapping_to_coords(
    grid_mapping: GridMapping,
    xy_var_names: tuple[str, str] = None,
    xy_dim_names: tuple[str, str] = None,
    reuse_coords: bool = False,
    exclude_bounds: bool = False,
) -> dict[str, xr.DataArray]:
    """Get CF-compliant axis coordinate variables and cell
    boundary coordinate variables.

    Defined only for grid mappings with regular x,y coordinates.

    Args:
        grid_mapping: A regular grid mapping.
        xy_var_names: Optional coordinate variable names (x_var_name,
            y_var_name).
        xy_dim_names: Optional coordinate dimensions names (x_dim_name,
            y_dim_name).
        reuse_coords: Whether to either reuse target coordinate arrays
            from target_gm or to compute new ones.
        exclude_bounds: If True, do not create bounds coordinates.
            Defaults to False. Ignored if *reuse_coords* is True.

    Returns:
        dictionary with coordinate variables
    """

    if xy_var_names:
        _assert_valid_xy_names(xy_var_names, name="xy_var_names")
    if xy_dim_names:
        _assert_valid_xy_names(xy_dim_names, name="xy_dim_names")

    if reuse_coords:
        try:
            # noinspection PyUnresolvedReferences
            x, y = grid_mapping.x_coords, grid_mapping.y_coords
        except AttributeError:
            x, y = None, None
        if (
            isinstance(x, xr.DataArray)
            and isinstance(y, xr.DataArray)
            and x.ndim == 1
            and y.ndim == 1
            and x.size == grid_mapping.width
            and y.size == grid_mapping.height
        ):
            return {
                name: xr.DataArray(coord.values, dims=dim, attrs=coord.attrs)
                for name, dim, coord in zip(xy_var_names, xy_dim_names, (x, y))
            }

    x_name, y_name = xy_var_names or grid_mapping.xy_var_names
    x_dim_name, y_dim_name = xy_dim_names or grid_mapping.xy_dim_names
    w, h = grid_mapping.size
    x1, y1, x2, y2 = grid_mapping.xy_bbox
    x_res, y_res = grid_mapping.xy_res
    x_res_05 = x_res / 2
    y_res_05 = y_res / 2

    dtype = np.float64

    x_data = np.linspace(x1 + x_res_05, x2 - x_res_05, w, dtype=dtype)
    if grid_mapping.is_lon_360:
        x_data = from_lon_360(x_data)

    if grid_mapping.is_j_axis_up:
        y_data = np.linspace(y1 + y_res_05, y2 - y_res_05, h, dtype=dtype)
    else:
        y_data = np.linspace(y2 - y_res_05, y1 + y_res_05, h, dtype=dtype)

    if grid_mapping.crs.is_geographic:
        x_attrs = dict(
            long_name="longitude coordinate",
            standard_name="longitude",
            units="degrees_east",
        )
        y_attrs = dict(
            long_name="latitude coordinate",
            standard_name="latitude",
            units="degrees_north",
        )
    else:
        x_attrs = dict(
            long_name="x coordinate of projection",
            standard_name="projection_x_coordinate",
        )
        y_attrs = dict(
            long_name="y coordinate of projection",
            standard_name="projection_y_coordinate",
        )

    x_coords = xr.DataArray(x_data, dims=x_dim_name, attrs=x_attrs)
    y_coords = xr.DataArray(y_data, dims=y_dim_name, attrs=y_attrs)
    coords = {
        x_name: x_coords,
        y_name: y_coords,
    }
    if not exclude_bounds:
        x_bnds_0_data = np.linspace(x1, x2 - x_res, w, dtype=dtype)
        x_bnds_1_data = np.linspace(x1 + x_res, x2, w, dtype=dtype)

        if grid_mapping.is_lon_360:
            x_bnds_0_data = from_lon_360(x_bnds_0_data)
            x_bnds_1_data = from_lon_360(x_bnds_1_data)

        if grid_mapping.is_j_axis_up:
            y_bnds_0_data = np.linspace(y1, y2 - y_res, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y1 + y_res, y2, h, dtype=dtype)
        else:
            y_bnds_0_data = np.linspace(y2, y1 + y_res, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y2 - y_res, y1, h, dtype=dtype)

        bnds_dim_name = "bnds"
        x_bnds_name = f"{x_name}_{bnds_dim_name}"
        y_bnds_name = f"{y_name}_{bnds_dim_name}"
        # Note, according to CF, bounds variables are not required to have
        # any attributes, so we don't pass any.
        x_bnds_coords = xr.DataArray(
            list(zip(x_bnds_0_data, x_bnds_1_data)), dims=[x_dim_name, bnds_dim_name]
        )
        y_bnds_coords = xr.DataArray(
            list(zip(y_bnds_0_data, y_bnds_1_data)), dims=[y_dim_name, bnds_dim_name]
        )
        x_coords.attrs.update(bounds=x_bnds_name)
        y_coords.attrs.update(bounds=y_bnds_name)
        coords.update(
            {
                x_bnds_name: x_bnds_coords,
                y_bnds_name: y_bnds_coords,
            }
        )

    return coords
