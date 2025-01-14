# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
import pyproj

from test.sampledata import SourceDatasetMixin
from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.regular import RegularGridMapping
from xcube.core.new import new_cube
from xcube.core.resampling import resample_in_space

nan = np.nan


class ResampleInSpaceTest(SourceDatasetMixin, unittest.TestCase):
    def test_affine_transform_dataset(self):
        source_ds = new_cube(variables={"CHL": 10.0, "TSM": 8.5})
        source_gm = GridMapping.from_dataset(source_ds)
        target_gm = GridMapping.regular(
            size=(8, 4), xy_min=(0, 0), xy_res=2, crs=CRS_WGS84
        )

        target_ds = resample_in_space(
            source_ds,
            source_gm=source_gm,
            target_gm=target_gm,
            encode_cf=True,
            gm_name="crs",
        )

        self.assertIn("crs", target_ds)
        self.assertEqual(target_gm.crs, pyproj.CRS.from_cf(target_ds.crs.attrs))

        for var_name in ("CHL", "TSM"):
            self.assertIn(var_name, target_ds)
            self.assertEqual("crs", target_ds[var_name].attrs.get("grid_mapping"))

        actual_gm = GridMapping.from_dataset(target_ds)
        self.assertEqual(RegularGridMapping, type(target_gm))
        self.assertEqual(RegularGridMapping, type(actual_gm))
        self.assertEqual(actual_gm.crs, target_gm.crs)
        self.assertEqual(actual_gm.xy_res, target_gm.xy_res)
        self.assertEqual(actual_gm.xy_bbox, target_gm.xy_bbox)
        self.assertEqual(actual_gm.xy_dim_names, target_gm.xy_dim_names)

    # noinspection PyMethodMayBeStatic
    def test_rectify_and_downscale_dataset(self):
        source_ds = self.new_4x4_dataset_with_irregular_coords()
        target_gm = GridMapping.regular(
            size=(2, 2), xy_min=(-1, 51), xy_res=2, crs=CRS_WGS84
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.rad.values,
            np.array(
                [
                    [8.0, 4.0],
                    [13 + 1 / 3, 10 + 1 / 3],
                ],
                dtype=target_ds.rad.dtype,
            ),
        )

    # noinspection PyMethodMayBeStatic
    def test_rectify_and_upscale_dataset(self):
        source_ds = self.new_2x2_dataset_with_irregular_coords()
        target_gm = GridMapping.regular(
            size=(4, 4), xy_min=(-1, 49), xy_res=2, crs=CRS_WGS84
        )
        target_ds = resample_in_space(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.rad.values,
            np.array(
                [
                    [nan, nan, nan, nan],
                    [nan, 1.0, 2.0, nan],
                    [3.0, 3.0, 2.0, nan],
                    [nan, 4.0, nan, nan],
                ],
                dtype=target_ds.rad.dtype,
            ),
        )
