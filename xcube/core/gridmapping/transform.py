# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Union

import numpy as np
import pyproj
import pyproj.transformer as pt
import xarray as xr

from .base import DEFAULT_TOLERANCE
from .base import GridMapping
from .coords import new_grid_mapping_from_coords
from .helpers import _assert_valid_xy_names
from .helpers import _normalize_number_pair
from .helpers import Number
from .helpers import _normalize_crs


# Cannot be used, but should, see TODO in transform_grid_mapping()
#
# class TransformedGridMapping(GridMapping, abc.ABC):
#     """
#     Grid mapping constructed from 1D/2D coordinate variables and a CRS.
#     """
#
#     def __init__(self,
#                  /,
#                  xy_coords: xr.DataArray,
#                  **kwargs):
#         self._xy_coords = xy_coords
#         super().__init__(**kwargs)
#
#     @property
#     def xy_coords(self) -> xr.DataArray:
#         return self._xy_coords
#


def transform_grid_mapping(
    grid_mapping: GridMapping,
    crs: Union[str, pyproj.crs.CRS],
    *,
    xy_res: Union[Number, tuple[Number, Number]] = None,
    tile_size: Union[int, tuple[int, int]] = None,
    xy_var_names: tuple[str, str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GridMapping:
    target_crs = _normalize_crs(crs)

    if xy_var_names:
        _assert_valid_xy_names(xy_var_names, name="xy_var_names")

    source_crs = grid_mapping.crs
    if source_crs == target_crs:
        if tile_size is not None or xy_var_names is not None:
            return grid_mapping.derive(tile_size=tile_size, xy_var_names=xy_var_names)
        return grid_mapping

    transformer = pt.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def _transform(block: np.ndarray) -> np.ndarray:
        x1, y1 = block
        x2, y2 = transformer.transform(x1, y1)
        return np.stack([x2, y2])

    xy_coords = xr.apply_ufunc(
        _transform,
        grid_mapping.xy_coords,
        output_dtypes=[np.float64],
        dask="parallelized",
    )
    if xy_res is not None:
        if grid_mapping.is_j_axis_up:
            gm_ymin = grid_mapping.y_coords[0].values
            gm_ymax = grid_mapping.y_coords[-1].values
        else:
            gm_ymin = grid_mapping.y_coords[-1].values
            gm_ymax = grid_mapping.y_coords[0].values
        y_min = np.min(
            transformer.transform(
                grid_mapping.x_coords.values,
                np.repeat(gm_ymin, grid_mapping.size[0]),
            )[1]
        )
        y_max = np.max(
            transformer.transform(
                grid_mapping.x_coords.values,
                np.repeat(gm_ymax, grid_mapping.size[0]),
            )[1]
        )
        x_min = np.min(
            transformer.transform(
                np.repeat(grid_mapping.x_coords[0].values, grid_mapping.size[1]),
                grid_mapping.y_coords.values,
            )[0]
        )
        x_max = np.max(
            transformer.transform(
                np.repeat(grid_mapping.x_coords[-1].values, grid_mapping.size[1]),
                grid_mapping.y_coords.values,
            )[0]
        )
        x_res, y_res = _normalize_number_pair(xy_res)
        x_res_05, y_res_05 = x_res / 2, y_res / 2
        xy_bbox = (
            x_min - x_res_05,
            y_min - y_res_05,
            x_max + x_res_05,
            y_max + y_res_05,
        )
    else:
        xy_bbox = None

    xy_var_names = xy_var_names or ("transformed_x", "transformed_y")

    # TODO: Use a specialized grid mapping here that can store the
    #   *xy_coords* directly. Splitting the xy_coords dask array into
    #   x,y components as done here may be very inefficient for larger
    #   arrays, because x cannot be computed independently from y.
    #   This means, any access of x chunks will cause y chunks to be
    #   computed too and vice versa. As same operations are performed
    #   on x and y arrays, this will take twice as long as if operation
    #   would be performed on the xy_coords dask array directly.

    if tile_size is None:
        tile_size = grid_mapping.tile_size

    return new_grid_mapping_from_coords(
        x_coords=xy_coords[0].rename(xy_var_names[0]),
        y_coords=xy_coords[1].rename(xy_var_names[1]),
        crs=target_crs,
        xy_res=xy_res,
        xy_bbox=xy_bbox,
        tile_size=tile_size,
        tolerance=tolerance,
    )
