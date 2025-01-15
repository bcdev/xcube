# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import warnings
from collections.abc import MutableMapping
from typing import Optional, Dict, Any, Union, Set, List, Tuple
from collections.abc import Hashable

import numpy as np
import pyproj
import xarray as xr
import zarr
import zarr.convenience

from xcube.core.schema import get_dataset_chunks
from xcube.util.assertions import assert_instance
from .base import CRS_WGS84


class GridCoords:
    """Grid coordinates comprising x and y of
    type xarray.DataArray.
    """

    def __init__(self):
        self.x: Optional[xr.DataArray] = None
        self.y: Optional[xr.DataArray] = None


class GridMappingProxy:
    """Grid mapping comprising *crs* of type pyproj.crs.CRS,
    grid coordinates, an optional name, coordinates, and a
    tile size (= spatial chunk sizes).
    """

    def __init__(
        self,
        crs: Optional[pyproj.crs.CRS] = None,
        name: Optional[str] = None,
        coords: Optional[GridCoords] = None,
        tile_size: Optional[tuple[int, int]] = None,
    ):
        self.crs = crs
        self.name = name
        self.coords = coords
        self.tile_size = tile_size


def get_dataset_grid_mapping_proxies(
    dataset: xr.Dataset,
    *,
    missing_latitude_longitude_crs: pyproj.crs.CRS = None,
    missing_rotated_latitude_longitude_crs: pyproj.crs.CRS = None,
    missing_projected_crs: pyproj.crs.CRS = None,
    emit_warnings: bool = False,
) -> dict[Union[Hashable, None], GridMappingProxy]:
    """Find grid mappings encoded as described in the CF conventions
    [Horizontal Coordinate Reference Systems, Grid Mappings, and Projections]
    (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections).

    Args:
        dataset
        missing_latitude_longitude_crs
        missing_rotated_latitude_longitude_crs
        missing_projected_crs
        emit_warnings

    Returns:

    """
    grid_mapping_proxies: dict[Union[Hashable, None], GridMappingProxy] = dict()

    # Find any grid mapping variables by CF 'grid_mapping' attribute
    for var_name, var in dataset.variables.items():
        grid_mapping_var_name = var.attrs.get("grid_mapping")
        if (
            grid_mapping_var_name
            and grid_mapping_var_name not in grid_mapping_proxies
            and grid_mapping_var_name in dataset
        ):
            grid_mapping_var = dataset[grid_mapping_var_name]
            gmp = _parse_crs_from_attrs(grid_mapping_var.attrs)
            grid_mapping_proxies[grid_mapping_var_name] = gmp

    # If no grid mapping variables found,
    # try if CRS is encoded in some variable's attributes
    if not grid_mapping_proxies:
        for var_name, var in dataset.variables.items():
            gmp = _parse_crs_from_attrs(var.attrs)
            if gmp is not None:
                grid_mapping_proxies[var_name] = gmp
                break

    # If no grid mapping variables found,
    # try if CRS is encoded in dataset attributes
    if not grid_mapping_proxies:
        gmp = _parse_crs_from_attrs(dataset.attrs)
        if gmp is not None:
            grid_mapping_proxies[None] = gmp

    # Find coordinate variables.
    latitude_longitude_coords = GridCoords()
    rotated_latitude_longitude_coords = GridCoords()
    projected_coords = GridCoords()

    potential_coord_vars = _find_potential_coord_vars(dataset)

    # Find coordinate variables that use a CF standard_name.
    coords_standard_names = (
        (latitude_longitude_coords, "longitude", "latitude"),
        (rotated_latitude_longitude_coords, "grid_longitude", "grid_latitude"),
        (projected_coords, "projection_x_coordinate", "projection_y_coordinate"),
    )
    for var_name in potential_coord_vars:
        var = dataset[var_name]
        if var.ndim not in (1, 2):
            continue
        standard_name = var.attrs.get("standard_name")
        for coords, x_name, y_name in coords_standard_names:
            if coords.x is None and standard_name == x_name:
                coords.x = var
            if coords.y is None and standard_name == y_name:
                coords.y = var

    # Find coordinate variables by common naming convention.
    coords_var_names = (
        (latitude_longitude_coords, ("lon", "longitude"), ("lat", "latitude")),
        (
            rotated_latitude_longitude_coords,
            ("rlon", "rlongitude"),
            ("rlat", "rlatitude"),
        ),
        (projected_coords, ("x", "xc", "transformed_x"), ("y", "yc", "transformed_y")),
    )
    for var_name in potential_coord_vars:
        var = dataset[var_name]
        if var.ndim not in (1, 2):
            continue
        for coords, x_names, y_names in coords_var_names:
            if coords.x is None and var_name in x_names:
                coords.x = var
            if coords.y is None and var_name in y_names:
                coords.y = var

    # Assign found coordinates to grid mappings
    #
    for gmp in grid_mapping_proxies.values():
        if gmp.name == "latitude_longitude":
            gmp.coords = latitude_longitude_coords
        elif gmp.name == "rotated_latitude_longitude":
            gmp.coords = rotated_latitude_longitude_coords
        else:
            gmp.coords = projected_coords

    _complement_grid_mapping_coords(
        latitude_longitude_coords,
        "latitude_longitude",
        missing_latitude_longitude_crs or CRS_WGS84,
        grid_mapping_proxies,
    )
    _complement_grid_mapping_coords(
        rotated_latitude_longitude_coords,
        "rotated_latitude_longitude",
        missing_rotated_latitude_longitude_crs,
        grid_mapping_proxies,
    )
    _complement_grid_mapping_coords(
        projected_coords, None, missing_projected_crs, grid_mapping_proxies
    )

    # Collect complete grid mappings
    complete_grid_mappings = dict()
    for var_name, gmp in grid_mapping_proxies.items():
        if (
            gmp.coords is not None
            and gmp.coords.x is not None
            and gmp.coords.y is not None
            and gmp.coords.x.size >= 2
            and gmp.coords.y.size >= 2
            and gmp.coords.x.ndim == gmp.coords.y.ndim
        ):
            if gmp.coords.x.ndim == 1:
                gmp.tile_size = _find_dataset_tile_size(
                    dataset, gmp.coords.x.dims[0], gmp.coords.y.dims[0]
                )
                complete_grid_mappings[var_name] = gmp
            elif gmp.coords.x.ndim == 2 and gmp.coords.x.dims == gmp.coords.y.dims:
                gmp.tile_size = _find_dataset_tile_size(
                    dataset, gmp.coords.x.dims[1], gmp.coords.x.dims[0]
                )
                complete_grid_mappings[var_name] = gmp
        elif emit_warnings:
            warnings.warn(
                f'CRS "{gmp.name}": '
                f"missing x- and/or y-coordinates "
                f'(grid mapping variable "{var_name}": '
                f'grid_mapping_name="{gmp.name}")'
            )

    return complete_grid_mappings


def _parse_crs_from_attrs(attrs: dict[Hashable, Any]) -> Optional[GridMappingProxy]:
    # noinspection PyBroadException
    try:
        crs = pyproj.crs.CRS.from_cf(attrs)
    except pyproj.crs.CRSError:
        return None
    return GridMappingProxy(crs=crs, name=attrs.get("grid_mapping_name"))


def _complement_grid_mapping_coords(
    coords: GridCoords,
    grid_mapping_name: Optional[str],
    missing_crs: Optional[pyproj.crs.CRS],
    grid_mappings: dict[Optional[str], GridMappingProxy],
):
    if coords.x is not None or coords.y is not None:
        grid_mapping = next(
            (
                grid_mapping
                for grid_mapping in grid_mappings.values()
                if grid_mapping_name is None or grid_mapping_name == grid_mapping.name
            ),
            None,
        )
        if grid_mapping is None and missing_crs is not None:
            grid_mapping = GridMappingProxy(crs=missing_crs, name=grid_mapping_name)
            grid_mappings[None] = grid_mapping

        if grid_mapping is not None:
            if grid_mapping.coords is None:
                grid_mapping.coords = coords
            # Edge case from GeoTIFF with CRS-84 with 1D
            # coordinates named "x" and "y" as read by rioxarray
            if grid_mapping.coords.x is None:
                grid_mapping.coords.x = coords.x
            if grid_mapping.coords.y is None:
                grid_mapping.coords.y = coords.y


def _find_potential_coord_vars(dataset: xr.Dataset) -> list[Hashable]:
    """Find potential coordinate variables.

    We need this function as we can not rely on xarray.Dataset.coords,
    because 2D coordinate arrays are most likely not indicated as such
    in many datasets.
    """

    # Collect bounds variables. We must exclude them.
    bounds_vars = set()
    for k in dataset.variables:
        var = dataset[k]

        # Bounds variable as recommended through CF conventions
        bounds_k = var.attrs.get("bounds")
        if bounds_k is not None and bounds_k in dataset:
            bounds_vars.add(bounds_k)

        # Bounds variable by naming convention,
        # e.g. "lon_bnds" or "lat_bounds"
        k_splits = str(k).rsplit("_", maxsplit=1)
        if len(k_splits) == 2:
            k_base, k_suffix = k_splits
            if k_suffix in ("bnds", "bounds") and k_base in dataset:
                bounds_vars.add(k)

    potential_coord_vars = []

    # First consider any CF global attribute "coordinates"
    coordinates = dataset.attrs.get("coordinates")
    if coordinates is not None:
        for var_name in coordinates.split():
            if _is_potential_coord_var(dataset, bounds_vars, var_name):
                potential_coord_vars.append(var_name)

    # Then consider any other 1D/2D variables
    for var_name in dataset.variables:
        if var_name not in potential_coord_vars and _is_potential_coord_var(
            dataset, bounds_vars, var_name
        ):
            potential_coord_vars.append(var_name)

    return potential_coord_vars


def _is_potential_coord_var(
    dataset: xr.Dataset, bounds_var_names: set[str], var_name: Hashable
) -> bool:
    if var_name in dataset:
        var = dataset[var_name]
        return var.ndim in (1, 2) and var_name not in bounds_var_names
    return False


def _find_dataset_tile_size(
    dataset: xr.Dataset, x_dim_name: Hashable, y_dim_name: Hashable
) -> Optional[tuple[int, int]]:
    """Find the most likely tile size in *dataset*"""
    dataset_chunks = get_dataset_chunks(dataset)
    tile_width = dataset_chunks.get(x_dim_name)
    tile_height = dataset_chunks.get(y_dim_name)
    if tile_width is not None and tile_height is not None:
        return tile_width, tile_height
    return None


def add_spatial_ref(
    dataset_store: zarr.convenience.StoreLike,
    crs: pyproj.CRS,
    crs_var_name: Optional[str] = "spatial_ref",
    xy_dim_names: Optional[tuple[str, str]] = None,
):
    """Helper function that allows adding a spatial reference to an
    existing Zarr dataset.

    Args:
        dataset_store: The dataset's existing Zarr store or path.
        crs: The spatial coordinate reference system.
        crs_var_name: The name of the variable that will hold the
            spatial reference. Defaults to "spatial_ref".
        xy_dim_names: The names of the x and y dimensions. Defaults to
            ("x", "y").
    """
    assert_instance(dataset_store, (MutableMapping, str), name="group_store")
    assert_instance(crs_var_name, str, name="crs_var_name")
    x_dim_name, y_dim_name = xy_dim_names or ("x", "y")

    spatial_attrs = crs.to_cf()
    spatial_attrs["_ARRAY_DIMENSIONS"] = []  # Required by xarray
    group = zarr.open(dataset_store, mode="r+")
    spatial_ref = group.array(crs_var_name, 0, shape=(), dtype=np.uint8, fill_value=0)
    spatial_ref.attrs.update(**spatial_attrs)

    for item_name, item in group.items():
        if item_name != crs_var_name:
            dims = item.attrs.get("_ARRAY_DIMENSIONS")
            if (
                dims
                and len(dims) >= 2
                and dims[-2] == y_dim_name
                and dims[-1] == x_dim_name
            ):
                item.attrs["grid_mapping"] = crs_var_name

    zarr.convenience.consolidate_metadata(dataset_store)
