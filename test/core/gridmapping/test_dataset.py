# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os.path
import unittest

import numpy as np
import pyproj
import xarray as xr

import xcube.core.new
from test.sampledata import create_s2plus_dataset
from xcube.core.gridmapping import GridMapping

# noinspection PyProtectedMember

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


# noinspection PyMethodMayBeStatic
class DatasetGridMappingTest(unittest.TestCase):
    def test_from_regular_cube(self):
        dataset = xcube.core.new.new_cube(variables=dict(rad=0.5))
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((360, 180), gm.size)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual((2, 180, 360), gm.xy_coords.shape)
        self.assertEqual(("coord", "lat", "lon"), gm.xy_coords.dims)

    def test_from_regular_cube_with_crs(self):
        dataset = xcube.core.new.new_cube(
            variables=dict(rad=0.5),
            x_start=0,
            y_start=0,
            x_name="x",
            y_name="y",
            crs="epsg:25832",
        )
        gm1 = GridMapping.from_dataset(dataset)
        self.assertEqual(pyproj.CRS.from_string("epsg:25832"), gm1.crs)
        dataset = dataset.drop_vars("crs")
        gm2 = GridMapping.from_dataset(dataset)
        self.assertEqual(GEO_CRS, gm2.crs)
        gm3 = GridMapping.from_dataset(dataset, crs=gm1.crs)
        self.assertEqual(gm1.crs, gm3.crs)
        self.assertEqual(("x", "y"), gm3.xy_var_names)
        self.assertEqual(("x", "y"), gm3.xy_dim_names)

    def test_from_regular_cube_no_chunks_and_chunks(self):
        dataset = xcube.core.new.new_cube(variables=dict(rad=0.5))
        gm1 = GridMapping.from_dataset(dataset)
        self.assertEqual((360, 180), gm1.tile_size)
        dataset = dataset.chunk(dict(lon=10, lat=20))
        gm2 = GridMapping.from_dataset(dataset)
        self.assertEqual((10, 20), gm2.tile_size)

    def test_from_non_regular_cube(self):
        lon = np.array(
            [[8, 9.3, 10.6, 11.9], [8, 9.2, 10.4, 11.6], [8, 9.1, 10.2, 11.3]],
            dtype=np.float32,
        )
        lat = np.array(
            [[56, 56.1, 56.2, 56.3], [55, 55.2, 55.4, 55.6], [54, 54.3, 54.6, 54.9]],
            dtype=np.float32,
        )
        rad = np.random.random(3 * 4).reshape((3, 4))
        dims = ("y", "x")
        dataset = xr.Dataset(
            dict(
                lon=xr.DataArray(lon, dims=dims),
                lat=xr.DataArray(lat, dims=dims),
                rad=xr.DataArray(rad, dims=dims),
            )
        )
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((4, 3), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual((2, 3, 4), gm.xy_coords.shape)
        self.assertEqual(("coord", "y", "x"), gm.xy_coords.dims)
        self.assertEqual((0.8, 0.8), gm.xy_res)

    def test_from_real_olci(self):
        olci_l2_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "examples",
            "notebooks",
            "inputdata",
            "S3-OLCI-L2A.zarr.zip",
        )

        dataset = xr.open_zarr(olci_l2_path, consolidated=False)
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((1189, 1890), gm.size)
        self.assertEqual((512, 512), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((0.0025, 0.0025), gm.xy_res)
        # self.assertAlmostEqual(12.693771178309552, gm.x_min)
        # self.assertAlmostEqual(20.005413821690446, gm.x_max)
        # self.assertAlmostEqual(55.19965017830955, gm.y_min)
        # self.assertAlmostEqual(60.63871982169044, gm.y_max)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(False, gm.is_j_axis_up)
        self.assertEqual((2, 1890, 1189), gm.xy_coords.shape)
        self.assertEqual(("coord", "y", "x"), gm.xy_coords.dims)

        gm = gm.to_regular()
        self.assertEqual((2926, 2177), gm.size)

    def test_from_sentinel_2(self):
        dataset = create_s2plus_dataset()
        tol = 1e-6

        gm = GridMapping.from_dataset(dataset, tolerance=tol)
        # Should pick the projected one which is regular
        self.assertIn("Projected", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=True, tolerance=tol)
        # Should pick the projected one which is regular
        self.assertIn("Projected", gm.crs.type_name)
        self.assertEqual(True, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_is_regular=False, tolerance=tol)
        # Should pick the geographic one which is irregular
        self.assertIn("Geographic", gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

        gm = GridMapping.from_dataset(dataset, prefer_crs=GEO_CRS, tolerance=tol)
        # Should pick the geographic one which is irregular
        self.assertIn("Geographic", gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

        gm = GridMapping.from_dataset(
            dataset, prefer_crs=GEO_CRS, prefer_is_regular=True, tolerance=tol
        )
        # Should pick the geographic one which is irregular
        self.assertIn("Geographic", gm.crs.type_name)
        self.assertEqual(False, gm.is_regular)

    def test_no_grid_mapping_found(self):
        with self.assertRaises(ValueError) as cm:
            GridMapping.from_dataset(xr.Dataset())
        self.assertEqual("cannot find any grid mapping in dataset", f"{cm.exception}")
