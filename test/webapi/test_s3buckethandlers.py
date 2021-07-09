import unittest
import urllib.request

import numpy as np
import xarray as xr

from xcube.core.dsio import open_cube

SKIP_HELP = ('Skipped, because server is not running:'
             ' $ xcube serve --verbose -c examples/serve/demo/config.yml')
SERVER_URL = 'http://localhost:8080'
ENDPOINT_URL = SERVER_URL + '/s3bucket'


def is_server_running() -> bool:
    # noinspection PyBroadException
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2.0) as response:
            response.read()
    except Exception:
        return False
    return 200 <= response.code < 400


XCUBE_SERVER_IS_RUNNING = is_server_running()


class S3BucketHandlersTest(unittest.TestCase):

    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_open_cube_from_xube_server_rel_path(self):
        ds = open_cube('s3bucket/local',
                       format_name='zarr',
                       s3_client_kwargs=dict(endpoint_url=SERVER_URL))
        self.assertCubeOk(ds)

    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_open_cube_from_xube_server_abs_path(self):
        ds = open_cube('http://localhost:8080/s3bucket/local',
                       format_name='zarr')
        self.assertCubeOk(ds)

    def assertCubeOk(self, ds):
        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual((5, 1000, 2000), ds.conc_chl.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.conc_chl.dims)
        conc_chl_values = ds.conc_chl.values
        self.assertEqual((5, 1000, 2000), conc_chl_values.shape)
        self.assertAlmostEqual(0.00005656,
                               float(np.nanmin(conc_chl_values)),
                               delta=1e-6)
        self.assertAlmostEqual(22.4421215,
                               float(np.nanmax(conc_chl_values)),
                               delta=1e-6)
