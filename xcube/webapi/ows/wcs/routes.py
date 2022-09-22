# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import tempfile

import dask.array
from tornado import iostream
from xarray import Dataset

from xcube.server.api import ApiHandler, ApiError
from .api import api
from .context import WcsContext
from .controllers import get_wcs_capabilities_xml, get_describe_coverage_xml, \
    get_coverage, CoverageRequest

WCS_VERSION = '1.0.0'


@api.route('/wcs/1.0.0/WCSCapabilities.xml')
class WcsCapabilitiesXmlHandler(ApiHandler[WcsContext]):
    @api.operation(operationId='getWcsCapabilities',
                   summary='Gets the WCS capabilities as XML document')
    async def get(self):
        self.request.make_query_lower_case()
        capabilities = await self.ctx.run_in_executor(
            None,
            get_wcs_capabilities_xml,
            self.ctx,
            self.request.base_url
        )
        self.response.set_header('Content-Type', 'application/xml')
        await self.response.finish(capabilities)


@api.route('/wcs/kvp')
class WcsKvpHandler(ApiHandler[WcsContext]):
    @api.operation(operationId='invokeWcsMethodFromKvp',
                   summary='Invokes the WCS by key-value pairs')
    async def get(self):
        self.request.make_query_lower_case()
        service = self.request.get_query_arg('service', default='WCS')
        if service != 'WCS':
            raise ApiError.BadRequest(
                'value for "service" parameter must be "WCS"'
            )
        request = self.request.get_query_arg('request')
        if request == "GetCapabilities":
            wcs_version = self.request.get_query_arg(
                "version", default=WCS_VERSION
            )
            if wcs_version != WCS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WCS_VERSION}"'
                )
            capabilities_xml = await self.ctx.run_in_executor(
                None,
                get_wcs_capabilities_xml,
                self.ctx,
                self.request.base_url
            )
            self.response.set_header('Content-Type', 'application/xml')
            await self.response.finish(capabilities_xml)
        elif request == 'DescribeCoverage':
            wcs_version = self.request.get_query_arg('version',
                                                     default=WCS_VERSION)
            if wcs_version != WCS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WCS_VERSION}"'
                )
            coverages = self.request.get_query_arg("coverage")
            if coverages:
                coverages = coverages.split(',')

            describe_coverage_xml = await self.ctx.run_in_executor(
                None,
                get_describe_coverage_xml,
                self.ctx,
                coverages
            )
            self.response.set_header('Content-Type', 'application/xml')
            await self.response.finish(describe_coverage_xml)
        elif request == "GetCoverage":
            wcs_version = self.request.get_query_arg('version',
                                                     default=WCS_VERSION)
            if wcs_version != WCS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WCS_VERSION}"'
                )
            coverage = self.request.get_query_arg('coverage')
            if not coverage:
                raise ApiError.BadRequest(
                    f'missing query argument "coverage"'
                )
            request_crs = self.request.get_query_arg('crs')
            if not request_crs:
                raise ApiError.BadRequest(
                    f'missing query argument "crs"'
                )
            response_crs = self.request.get_query_arg('response_crs',
                                                      default=request_crs)
            time = self.request.get_query_arg('time')
            # QGIS specific hack!
            time = time.replace('Z', '')
            file_format = self.request.get_query_arg('format',
                                                     default='geotiff')
            bbox = self.request.get_query_arg('bbox',
                                              default='-180,90,180,-90')
            width = self.request.get_query_arg('width')
            height = self.request.get_query_arg('height')
            resx = self.request.get_query_arg('resx')
            resy = self.request.get_query_arg('resy')

            cov_req = CoverageRequest({
                'COVERAGE': coverage,
                'CRS': response_crs,
                'TIME': time,
                'BBOX': bbox,
                'FORMAT': file_format,
                'WIDTH': width,
                'HEIGHT': height,
                'RESX': resx,
                'RESY': resy
            })
            cube = await self.ctx.run_in_executor(
                None,
                get_coverage,
                self.ctx,
                cov_req
            )

            cube = await self.ctx.run_in_executor(
                None,
                self._clean_cube,
                cube,
                time
            )

            try:
                if file_format.lower() == 'netcdf4':
                    temp_file_name = await self.ctx.run_in_executor(
                        None, self._write_netcdf, cube
                    )
                elif file_format.lower() == 'geotiff':
                    temp_file_name = await self.ctx.run_in_executor(
                        None, self._write_geotiff, cube
                    )
                else:
                    raise ValueError('Unsupported format: ' + file_format)

                self.response.set_header('Content-Type',
                                         'application/octet-stream')

                chunk_size = 1024 * 1024
                with open(temp_file_name, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if chunk is None or len(chunk) == 0:
                            break
                        try:
                            self.response.write(chunk)
                        except iostream.StreamClosedError:
                            break
            finally:
                if 'temp_file_name' in locals():
                    os.remove(temp_file_name)
            await self.response.finish()
        else:
            raise ApiError.BadRequest(
                f'invalid request type "{request}"'
            )

    @staticmethod
    def _write_geotiff(cube: Dataset) -> str:
        temp_file_handle, temp_file_name = tempfile.mkstemp('.tif')
        cube.rio.to_raster(temp_file_name)
        os.close(temp_file_handle)
        return temp_file_name

    @staticmethod
    def _write_netcdf(cube: Dataset) -> str:
        temp_file_handle, temp_file_name = tempfile.mkstemp('.nc')
        cube.to_netcdf(temp_file_name, 'w')
        os.close(temp_file_handle)
        return temp_file_name

    @staticmethod
    def _clean_cube(cube: Dataset, time: str) -> Dataset:
        del cube.attrs['history']
        cube = cube.drop_vars(['lat_bnds', 'lon_bnds', 'time_bnds'])
        cube = cube.sel(time=time)
        cube = cube.reduce(dask.array.squeeze, dim='time')
        return cube


def _query_to_dict(request):
    return {k: v[0] for k, v in request.query.items()}