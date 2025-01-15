# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Union, Callable, Any, Optional
from collections.abc import Mapping, Hashable

import numpy as np
import xarray as xr
from dask import array as da

from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.helpers import scale_xy_res_and_size
from .affine import affine_transform_dataset
from .affine import resample_dataset
from .rectify import rectify_dataset

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]

# If _SCALE_LIMIT is exceeded, we don't need
# to downscale source image before we can
# rectify it.
_SCALE_LIMIT = 0.95


def resample_in_space(
    source_ds: xr.Dataset,
    /,
    source_gm: Optional[GridMapping] = None,
    target_gm: Optional[GridMapping] = None,
    ref_ds: Optional[xr.Dataset] = None,
    var_configs: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    encode_cf: bool = True,
    gm_name: Optional[str] = None,
    rectify_kwargs: Optional[dict] = None,
):
    """
    Resample a dataset *source_ds* in the spatial dimensions.

    If the source grid mapping *source_gm* is not given,
    it is derived from *dataset*:
    ``source_gm = GridMapping.from_dataset(source_ds)``.

    If the target grid mapping *target_gm* is not given,
    it is derived from *source_gm* as
    ``target_gm = source_gm.to_regular()``,
    or if target dataset *ref_ds* is given as
    ``target_gm = GridMapping.from_dataset(ref_ds)``.

    New in 1.6: If *ref_ds* is given, its coordinate
    variables are copied by reference into the returned
    dataset.

    If *source_gm* is almost equal to *target_gm*, this
    function is a no-op and *dataset* is returned unchanged.

    Otherwise, the function computes a spatially
    resampled version of *dataset* and returns it.

    Using *var_configs*, the resampling of individual
    variables can be configured. If given, *var_configs*
    must be a mapping from variable names to configuration
    dictionaries which can have the following properties:

    * ``spline_order`` (int) - The order of spline polynomials
        used for interpolating. It is used for up-sampling only.
        Possible values are 0 to 5.
        Default is 1 (bi-linear) for floating point variables,
        and 0 (= nearest neighbor) for integer and bool variables.
    * ``aggregator`` (str) - An optional aggregating
        function. It is used for down-sampling only.
        Examples are ``numpy.nanmean``, ``numpy.nanmin``,
        ``numpy.nanmax``.
        Default is ``numpy.nanmean`` for floating point variables,
        and None (= nearest neighbor) for integer and bool variables.
    * ``recover_nan`` (bool) - whether a special algorithm
        shall be used that is able to recover values that would
        otherwise yield NaN during resampling.
        Default is False for all variable types since this
        may require considerable CPU resources on top.

    Note that *var_configs* is only used if the resampling involves
    an affine transformation. This is true if the CRS of
    *source_gm* and CRS of *target_gm* are equal and one of two
    cases is given:

    1. *source_gm* is regular.
       In this case the resampling is the affine transformation.
       and the result is returned directly.
    2. *source_gm* is not regular and has a lower resolution
       than *target_cm*.
       In this case *dataset* is down-sampled first using an affine
       transformation. Then the result is rectified.

    In all other cases, no affine transformation is applied and
    the resampling is a direct rectification.

    Args:
        source_ds: The source dataset.
        source_gm: The source grid mapping.
        target_gm: The target grid mapping. Must be regular.
        ref_ds: An optional dataset that provides the
            target grid mapping if *target_gm* is not provided.
            If *ref_ds* is given, its coordinate variables are copied
            by reference into the returned dataset.
        var_configs: Optional resampling configurations
            for individual variables.
        encode_cf: Whether to encode the target grid mapping
            into the resampled dataset in a CF-compliant way.
            Defaults to ``True``.
        gm_name: Name for the grid mapping variable.
            Defaults to "crs". Used only if *encode_cf* is ``True``.
        rectify_kwargs: Keyword arguments passed func:`rectify_dataset`
            should a rectification be required.


    Returns: The spatially resampled dataset, or None if the requested
        output area does not intersect with *dataset*.
    """
    if source_gm is None:
        # No source grid mapping given, so do derive it from dataset.
        source_gm = GridMapping.from_dataset(source_ds)

    if target_gm is None:
        # No target grid mapping given, so do derive it
        # from target dataset or source grid mapping.
        if ref_ds is not None:
            target_gm = GridMapping.from_dataset(ref_ds)
        else:
            target_gm = source_gm.to_regular()

    if source_gm.is_close(target_gm):
        # If source and target grid mappings are almost equal.
        # NOTE: Actually we should only return input here if
        # encode_cf == False and gm_name is None and target_ds is None.
        # Otherwise, create a copy and apply encoding and coords copy.
        return source_ds

    # target_gm must be regular
    GridMapping.assert_regular(target_gm, name="target_gm")

    # Are source and target both geographic grid mappings?
    both_geographic = source_gm.crs.is_geographic and target_gm.crs.is_geographic

    if both_geographic or source_gm.crs == target_gm.crs:
        # If CRSes are both geographic or their CRSes are equal:
        if source_gm.is_regular:
            # If also the source is regular, then resampling reduces
            # to an affine transformation.
            return affine_transform_dataset(
                source_ds,
                source_gm=source_gm,
                ref_ds=ref_ds,
                target_gm=target_gm,
                var_configs=var_configs,
                encode_cf=encode_cf,
                gm_name=gm_name,
            )

        # If the source is not regular, we need to rectify it,
        # so the target is regular. Our rectification implementation
        # works only correctly if source pixel size >= target pixel
        # size. Therefore, check if we must downscale source first.
        x_scale = source_gm.x_res / target_gm.x_res
        y_scale = source_gm.y_res / target_gm.y_res
        if x_scale > _SCALE_LIMIT and y_scale > _SCALE_LIMIT:
            # Source pixel size >= target pixel size.
            # We can rectify.
            return rectify_dataset(
                source_ds,
                source_gm=source_gm,
                ref_ds=ref_ds,
                target_gm=target_gm,
                encode_cf=encode_cf,
                gm_name=gm_name,
                **(rectify_kwargs or {}),
            )
        else:
            # Source has higher resolution than target.
            # Downscale first, then rectify
            _, downscaled_size = scale_xy_res_and_size(
                source_gm.xy_res, source_gm.size, (x_scale, y_scale)
            )
            downscaled_dataset = resample_dataset(
                source_ds,
                ((1 / x_scale, 0, 0), (0, 1 / y_scale, 0)),
                size=downscaled_size,
                tile_size=source_gm.tile_size,
                xy_dim_names=source_gm.xy_dim_names,
                var_configs=var_configs,
            )
            downscaled_gm = GridMapping.from_dataset(
                downscaled_dataset,
                tile_size=source_gm.tile_size,
                crs=source_gm.crs,
            )
            return rectify_dataset(
                downscaled_dataset,
                source_gm=downscaled_gm,
                ref_ds=ref_ds,
                target_gm=target_gm,
                encode_cf=encode_cf,
                gm_name=gm_name,
                **(rectify_kwargs or {}),
            )

    # If CRSes are not both geographic and their CRSes are different
    # transform the source_gm so its CRS matches the target CRS:
    transformed_source_gm = source_gm.transform(target_gm.crs, xy_res=target_gm.xy_res)
    # source_ds = source_ds.drop_vars(source_gm.xy_dim_names)
    # if "crs" in source_ds:
    #     source_ds = source_ds.drop_vars("crs")
    # if "spatial_ref" in source_ds:
    #     source_ds = source_ds.drop_vars("spatial_ref")
    # source_ds = source_ds.copy()
    # for var in source_ds.data_vars:
    #     if "grid_mapping" in source_ds[var].attrs:
    #         attrs = source_ds[var].attrs
    #         del attrs["grid_mapping"]
    #         source_ds[var] = source_ds[var].assign_attrs(attrs)
    # transformed_x, transformed_y = transformed_source_gm.xy_coords
    # attrs = dict(grid_mapping="spatial_ref")
    # transformed_x.attrs = attrs
    # transformed_y.attrs = attrs
    # source_ds = source_ds.assign_coords(
    #     spatial_ref=xr.DataArray(0, attrs=transformed_source_gm.crs.to_cf()),
    #     transformed_x=transformed_x,
    #     transformed_y=transformed_y,
    # )
    transformed_x, transformed_y = transformed_source_gm.xy_coords
    attrs = dict(grid_mapping="transformed_spatial_ref")
    transformed_x.attrs = attrs
    transformed_y.attrs = attrs
    source_ds = source_ds.assign_coords(
        transformed_spatial_ref=xr.DataArray(
            0, attrs=transformed_source_gm.crs.to_cf()
        ),
        transformed_x=transformed_x,
        transformed_y=transformed_y,
    )
    return resample_in_space(
        source_ds,
        source_gm=transformed_source_gm,
        ref_ds=ref_ds,
        target_gm=target_gm,
        var_configs=var_configs,
        encode_cf=encode_cf,
        gm_name=gm_name,
        rectify_kwargs=rectify_kwargs,
    )
