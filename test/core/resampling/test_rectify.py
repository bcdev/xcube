# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
import pytest

# noinspection PyUnresolvedReferences
import xarray as xr

from test.sampledata import SourceDatasetMixin
from test.sampledata import create_s2plus_dataset
from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import rectify_dataset

nan = np.nan
_nan_ = np.nan


# noinspection PyMethodMayBeStatic
class RectifyDatasetTest(SourceDatasetMixin, unittest.TestCase):
    def test_rectify_2x2_to_default(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(4, 4), xy_min=(-1, 49), xy_res=2, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        rad = target_ds.rad
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [nan, nan, nan, nan],
                    [nan, 1.0, 2.0, nan],
                    [3.0, 3.0, 2.0, nan],
                    [nan, 4.0, nan, nan],
                ],
                dtype=rad.dtype,
            ),
        )

    def test_rectify_2x2_to_7x7(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()
        # Add offset to "rad" so its values do not lie on a plane
        source_ds["rad"] = source_ds.rad + xr.DataArray(
            np.array([[0.0, 0.0], [0.0, 1.0]]), dims=("y", "x")
        )

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [nan, 1.0, nan, nan, nan, nan, nan],
                    [nan, 1.0, 1.0, nan, nan, nan, nan],
                    [nan, 1.0, 1.0, 1.0, 2.0, nan, nan],
                    [nan, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 5.0, 2.0, nan, nan],
                    [nan, 3.0, 5.0, 5.0, nan, nan, nan],
                    [nan, nan, 5.0, nan, nan, nan, nan],
                ],
                dtype=rad.dtype,
            ),
        )

    def test_rectify_2x2_to_7x7_with_ref_ds(self):

        source_ds = self.new_2x2_dataset_with_irregular_coords()
        # Add offset to "rad" so its values do not lie on a plane
        source_ds["rad"] = source_ds.rad + xr.DataArray(
            np.array([[0.0, 0.0], [0.0, 1.0]]), dims=("y", "x")
        )

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)
        # target_ds is tested in test_rectify_2x2_to_default() above,
        # so we know it is valid.
        # We now do the same but use target_ds as reference dataset:
        ref_ds = target_ds
        target_ds = rectify_dataset(source_ds, ref_ds=ref_ds)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        # Coordinates must now be SAME, not almost equal.
        np.testing.assert_equal(lon.values, ref_ds.lon.values)
        np.testing.assert_equal(lat.values, ref_ds.lat.values)
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [nan, 1.0, nan, nan, nan, nan, nan],
                    [nan, 1.0, 1.0, nan, nan, nan, nan],
                    [nan, 1.0, 1.0, 1.0, 2.0, nan, nan],
                    [nan, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 5.0, 2.0, nan, nan],
                    [nan, 3.0, 5.0, 5.0, nan, nan, nan],
                    [nan, nan, 5.0, nan, nan, nan, nan],
                ],
                dtype=rad.dtype,
            ),
        )

    def test_rectify_2x2_to_7x7_triangular_interpol(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()
        # Add offset to "rad" so its values do not lie on a plane
        source_ds["rad"] = source_ds.rad + xr.DataArray(
            np.array([[0.0, 0.0], [0.0, 1.0]]), dims=("y", "x")
        )

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(
            source_ds, target_gm=target_gm, interpolation="triangular"
        )

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [_nan_, 1.000, _nan_, _nan_, _nan_, _nan_, _nan_],
                    [_nan_, 1.478, 1.391, _nan_, _nan_, _nan_, _nan_],
                    [_nan_, 1.957, 1.870, 1.784, 1.697, _nan_, _nan_],
                    [_nan_, 2.435, 2.348, 2.261, 2.174, 2.087, 2.000],
                    [3.000, 3.000, 3.000, 3.000, 3.000, _nan_, _nan_],
                    [_nan_, 4.000, 4.000, 4.000, _nan_, _nan_, _nan_],
                    [_nan_, _nan_, 5.000, _nan_, _nan_, _nan_, _nan_],
                ],
                dtype=rad.dtype,
            ),
            decimal=3,
        )

    def test_rectify_2x2_to_7x7_bilinear_interpol(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()
        # Add offset to "rad" so its values do not lie on a plane
        source_ds["rad"] = source_ds.rad + xr.DataArray(
            np.array([[0.0, 0.0], [0.0, 1.0]]), dims=("y", "x")
        )

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(
            source_ds, target_gm=target_gm, interpolation="bilinear"
        )

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [_nan_, 1.000, _nan_, _nan_, _nan_, _nan_, _nan_],
                    [_nan_, 1.488, 1.410, _nan_, _nan_, _nan_, _nan_],
                    [_nan_, 1.994, 1.949, 1.858, 1.722, _nan_, _nan_],
                    [_nan_, 2.520, 2.506, 2.448, 2.344, 2.195, 2.000],
                    [3.000, 3.112, 3.163, 3.153, 3.082, _nan_, _nan_],
                    [_nan_, 4.000, 4.041, 4.020, _nan_, _nan_, _nan_],
                    [_nan_, _nan_, 5.000, _nan_, _nan_, _nan_, _nan_],
                ],
                dtype=rad.dtype,
            ),
            decimal=3,
        )

    def test_rectify_2x2_to_7x7_invalid_interpol(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        with pytest.raises(ValueError, match="invalid interpolation: 'bicubic'"):
            rectify_dataset(source_ds, target_gm=target_gm, interpolation="bicubic")

    def test_rectify_2x2_to_7x7_subset(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(1.5, 50.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)
        lon, lat, rad = self._assert_shape_and_dim(target_ds, (7, 7))
        np.testing.assert_almost_equal(
            lon.values, np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=lon.dtype)
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array([57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0], dtype=lat.dtype),
        )
        np.testing.assert_almost_equal(
            rad.values,
            np.array(
                [
                    [nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan],
                    [1.0, nan, nan, nan, nan, nan, nan],
                    [1.0, 1.0, 2.0, nan, nan, nan, nan],
                    [3.0, 1.0, 2.0, 2.0, 2.0, nan, nan],
                    [3.0, 4.0, 2.0, nan, nan, nan, nan],
                    [4.0, 4.0, nan, nan, nan, nan, nan],
                ],
                dtype=rad.dtype,
            ),
        )

    def test_rectify_2x2_to_7x7_ij_only(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()
        source_ds = source_ds.drop_vars("rad")

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(
            source_ds, target_gm=target_gm, output_ij_names=("source_i", "source_j")
        )

        self.assertEqual({"source_i", "source_j"}, set(target_ds.data_vars.keys()))

    def test_rectify_2x2_to_7x7_deprecations(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(7, 7), xy_min=(-0.5, 49.5), xy_res=1.0, crs=CRS_WGS84
        )

        # Just to cover emitting deprecation warning
        rectify_dataset(source_ds, target_gm=target_gm, xy_var_names=("X", "Y"))

    def test_rectify_2x2_to_13x13(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (13, 13))
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_j_axis_up(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            is_j_axis_up=True,
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(target_ds, (13, 13))
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    50.0,
                    50.5,
                    51.0,
                    51.5,
                    52.0,
                    52.5,
                    53.0,
                    53.5,
                    54.0,
                    54.5,
                    55.0,
                    55.5,
                    56.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            rad.values, self.expected_rad_13x13(rad.dtype)[::-1]
        )

    def test_rectify_2x2_to_13x13_j_axis_up_dask_5x5(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            tile_size=5,
            is_j_axis_up=True,
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((5, 5, 3), (5, 5, 3))
        )
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    50.0,
                    50.5,
                    51.0,
                    51.5,
                    52.0,
                    52.5,
                    53.0,
                    53.5,
                    54.0,
                    54.5,
                    55.0,
                    55.5,
                    56.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            rad.values, self.expected_rad_13x13(rad.dtype)[::-1]
        )

    def test_rectify_2x2_to_13x13_dask_7x7(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84, tile_size=7
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((7, 6), (7, 6))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_5x5(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84, tile_size=5
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((5, 5, 3), (5, 5, 3))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_3x13(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            tile_size=(3, 13),
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((13,), (3, 3, 3, 3, 1))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_13x3(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13),
            xy_min=(-0.25, 49.75),
            xy_res=0.5,
            crs=CRS_WGS84,
            tile_size=(13, 3),
        )

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        lon, lat, rad = self._assert_shape_and_dim(
            target_ds, (13, 13), chunks=((3, 3, 3, 3, 1), (13,))
        )

        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_antimeridian(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords_antimeridian()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(177.75, 49.75), xy_res=0.5, crs=CRS_WGS84
        )

        self.assertEqual(True, target_gm.is_lon_360)

        target_ds = rectify_dataset(source_ds, target_gm=target_gm)

        self.assertIsNotNone(target_ds)
        lon, lat, rad = self._assert_shape_and_dim(target_ds, (13, 13))
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [
                    178.0,
                    178.5,
                    179.0,
                    179.5,
                    180.0,
                    -179.5,
                    -179.0,
                    -178.5,
                    -178.0,
                    -177.5,
                    -177.0,
                    -176.5,
                    -176.0,
                ],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_output_ij_names(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84
        )

        target_ds = rectify_dataset(
            source_ds, target_gm=target_gm, output_ij_names=("source_i", "source_j")
        )

        lon, lat, rad, source_i, source_j = self._assert_shape_and_dim(
            target_ds, (13, 13), var_names=("rad", "source_i", "source_j")
        )
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))
        np.testing.assert_almost_equal(
            np.floor(source_i.values + 0.5), self.expected_i_13x13()
        )
        np.testing.assert_almost_equal(
            np.floor(source_j.values + 0.5), self.expected_j_13x13()
        )

    def test_rectify_2x2_to_13x13_output_ij_names_dask(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-0.25, 49.75), xy_res=0.5, crs=CRS_WGS84, tile_size=5
        )

        target_ds = rectify_dataset(
            source_ds, target_gm=target_gm, output_ij_names=("source_i", "source_j")
        )
        lon, lat, rad, source_i, source_j = self._assert_shape_and_dim(
            target_ds,
            (13, 13),
            chunks=((5, 5, 3), (5, 5, 3)),
            var_names=("rad", "source_i", "source_j"),
        )
        np.testing.assert_almost_equal(
            lon.values,
            np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                dtype=lon.dtype,
            ),
        )
        np.testing.assert_almost_equal(
            lat.values,
            np.array(
                [
                    56.0,
                    55.5,
                    55.0,
                    54.5,
                    54.0,
                    53.5,
                    53.0,
                    52.5,
                    52.0,
                    51.5,
                    51.0,
                    50.5,
                    50.0,
                ],
                dtype=lat.dtype,
            ),
        )
        # print('source_i:', np.floor(source_i.values + 0.5))
        # print('source_j:', np.floor(source_j.values + 0.5))
        np.testing.assert_almost_equal(
            np.floor(source_i.values + 0.5), self.expected_i_13x13()
        )
        np.testing.assert_almost_equal(
            np.floor(source_j.values + 0.5), self.expected_j_13x13()
        )
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_none(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(10.0, 50.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm)
        self.assertIsNone(target_ds)

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(-10.0, 50.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm)
        self.assertIsNone(target_ds)

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(0.0, 58.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm)
        self.assertIsNone(target_ds)

        target_gm = GridMapping.regular(
            size=(13, 13), xy_min=(0.0, 42.0), xy_res=0.5, crs=CRS_WGS84
        )
        target_ds = rectify_dataset(source_ds, target_gm=target_gm)
        self.assertIsNone(target_ds)

    def _assert_shape_and_dim(
        self, target_ds, size, chunks=None, var_names=("rad",)
    ) -> tuple[xr.DataArray, ...]:
        w, h = size

        self.assertIn("lon", target_ds)
        lon = target_ds["lon"]
        self.assertEqual((w,), lon.shape)
        self.assertEqual(("lon",), lon.dims)

        self.assertIn("lat", target_ds)
        lat = target_ds["lat"]
        self.assertEqual((h,), lat.shape)
        self.assertEqual(("lat",), lat.dims)

        # noinspection PyShadowingBuiltins
        vars = []
        for var_name in var_names:
            self.assertIn(var_name, target_ds)
            var = target_ds[var_name]
            self.assertEqual((h, w), var.shape)
            self.assertEqual(("lat", "lon"), var.dims)
            self.assertEqual(chunks, var.chunks)
            vars.append(var)

        return (lon, lat, *vars)

    def test_compute_and_extract_source_pixels(self):
        from xcube.core.resampling.rectify import _compute_ij_images_numpy_parallel
        from xcube.core.resampling.rectify import _compute_var_image_numpy_parallel

        self._assert_compute_and_extract_source_pixels(
            _compute_ij_images_numpy_parallel, _compute_var_image_numpy_parallel, False
        )
        from xcube.core.resampling.rectify import _compute_ij_images_numpy_sequential
        from xcube.core.resampling.rectify import _compute_var_image_numpy_sequential

        self._assert_compute_and_extract_source_pixels(
            _compute_ij_images_numpy_sequential,
            _compute_var_image_numpy_sequential,
            False,
        )

    def test_compute_and_extract_source_pixels_j_axis_up(self):
        from xcube.core.resampling.rectify import _compute_ij_images_numpy_parallel
        from xcube.core.resampling.rectify import _compute_var_image_numpy_parallel

        self._assert_compute_and_extract_source_pixels(
            _compute_ij_images_numpy_parallel, _compute_var_image_numpy_parallel, True
        )
        from xcube.core.resampling.rectify import _compute_ij_images_numpy_sequential
        from xcube.core.resampling.rectify import _compute_var_image_numpy_sequential

        self._assert_compute_and_extract_source_pixels(
            _compute_ij_images_numpy_sequential,
            _compute_var_image_numpy_sequential,
            True,
        )

    def _assert_compute_and_extract_source_pixels(
        self, compute_ij_images, compute_var_image, is_j_axis_up: bool
    ):
        source_ds = self.new_2x2_dataset_with_irregular_coords()

        dst_src_ij = np.full((2, 13, 13), np.nan, dtype=np.float64)
        dst_x_offset = -0.25
        dst_y_offset = 49.75 if is_j_axis_up else 56.25
        dst_x_scale = 0.5
        dst_y_scale = 0.5 if is_j_axis_up else -0.5
        compute_ij_images(
            source_ds.lon.values,
            source_ds.lat.values,
            0,
            0,
            dst_src_ij,
            dst_x_offset,
            dst_y_offset,
            dst_x_scale,
            dst_y_scale,
            1e-5,
        )

        if not is_j_axis_up:
            np.testing.assert_almost_equal(
                np.floor(dst_src_ij[0] + 0.5), self.expected_i_13x13()
            )
            np.testing.assert_almost_equal(
                np.floor(dst_src_ij[1] + 0.5), self.expected_j_13x13()
            )
        else:
            np.testing.assert_almost_equal(
                np.floor(dst_src_ij[0] + 0.5), self.expected_i_13x13()[::-1]
            )
            np.testing.assert_almost_equal(
                np.floor(dst_src_ij[1] + 0.5), self.expected_j_13x13()[::-1]
            )

        target_rad = np.full((13, 13), np.nan, dtype=np.float64)

        src_bbox = [0, 0, source_ds.rad.shape[-1], source_ds.rad.shape[-2]]
        compute_var_image(source_ds.rad.values, dst_src_ij, target_rad, src_bbox, 0)

        if not is_j_axis_up:
            np.testing.assert_almost_equal(
                target_rad, self.expected_rad_13x13(target_rad.dtype)
            )
        else:
            np.testing.assert_almost_equal(
                target_rad, self.expected_rad_13x13(target_rad.dtype)[::-1]
            )

    def expected_i_13x13(self):
        return np.array(
            [
                [nan, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, nan, nan, nan, nan, nan],
                [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, nan, nan, nan, nan],
                [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, nan, nan],
                [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan],
                [nan, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan],
                [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan],
            ],
            dtype=np.float64,
        )

    def expected_j_13x13(self):
        return np.array(
            [
                [nan, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan],
                [nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan],
                [nan, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan],
                [nan, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, nan, nan],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, nan, nan, nan],
                [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan],
                [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan],
            ],
            dtype=np.float64,
        )

    def expected_rad_13x13(self, dtype):
        return np.array(
            [
                [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan],
                [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
                [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [nan, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, nan, nan, nan],
                [nan, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, nan, nan, nan, nan],
                [nan, nan, 3.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, 4.0, nan, nan, nan, nan, nan, nan, nan, nan],
            ],
            dtype=dtype,
        )


class RectifySentinel2DatasetTest(SourceDatasetMixin, unittest.TestCase):
    def test_rectify_dataset(self):
        source_ds = create_s2plus_dataset()

        expected_data = np.array(
            [
                [nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [
                    nan,
                    0.019001,
                    0.019001,
                    0.008999,
                    0.012001,
                    0.012001,
                    0.022999,
                    nan,
                    nan,
                ],
                [nan, 0.021, 0.021, 0.009998, 0.009998, 0.008999, 0.022999, nan, nan],
                [
                    nan,
                    0.022999,
                    0.022999,
                    0.007999,
                    0.007999,
                    0.008999,
                    0.023998,
                    nan,
                    nan,
                ],
                [nan, 0.022999, 0.022999, 0.007, 0.007, 0.009998, 0.021, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, nan, nan],
            ]
        )

        source_gm = GridMapping.from_dataset(
            source_ds, prefer_crs=CRS_WGS84, tolerance=1e-6
        )

        target_ds = rectify_dataset(source_ds, source_gm=source_gm)
        self.assertEqual(((5, 1), (5, 4)), target_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(
            target_ds.rrs_665.values, expected_data, decimal=3
        )

        target_ds = rectify_dataset(source_ds, source_gm=source_gm, tile_size=6)
        self.assertEqual(((6,), (6, 3)), target_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(
            target_ds.rrs_665.values, expected_data, decimal=3
        )

        target_ds = rectify_dataset(
            source_ds, source_gm=source_gm, tile_size=None, is_j_axis_up=True
        )
        self.assertEqual(((5, 1), (5, 4)), target_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(
            target_ds.rrs_665.values, expected_data[::-1], decimal=3
        )

        target_ds = rectify_dataset(
            source_ds, source_gm=source_gm, tile_size=6, is_j_axis_up=True
        )
        self.assertEqual(((6,), (6, 3)), target_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(
            target_ds.rrs_665.values, expected_data[::-1], decimal=3
        )
