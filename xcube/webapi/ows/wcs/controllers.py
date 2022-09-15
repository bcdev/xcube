# The MIT License (MIT)
# Copyright (c) 2021/2022 by the xcube team and contributors
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
import re
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
from xarray import Dataset

import xcube.core.store.storepool as sp
from xcube.constants import EXTENSION_POINT_DATASET_IOS
from xcube.core.gen2 import CubeGenerator, OutputConfig
from xcube.core.gen2 import CubeGeneratorRequest
from xcube.core.gen2.local.writer import CubeWriter
from xcube.core.gridmapping import GridMapping
from xcube.util.plugin import get_extension_registry
from xcube.webapi.ows.wcs.context import WcsContext
from xcube.webapi.ows.wmts.controllers import get_crs84_bbox
from xcube.webapi.xml import Document
from xcube.webapi.xml import Element

WCS_VERSION = '1.0.0'
VALID_CRS_LIST = ['EPSG:4326', 'EPSG:3857']


class CoverageRequest:
    coverage = None
    crs = None
    bbox = None
    time = None
    width = None
    height = None
    format = None
    resx = None
    resy = None
    interpolation = None
    parameter = None
    exceptions = None

    def __init__(self, req: Dict[str, Any]):
        if 'COVERAGE' in req:
            self.coverage = req['COVERAGE']
        if 'CRS' in req:
            self.crs = req['CRS']
        if 'BBOX' in req:
            self.bbox = req['BBOX']
        if 'TIME' in req:
            self.time = req['TIME']
        if 'WIDTH' in req:
            self.width = req['WIDTH']
        if 'HEIGHT' in req:
            self.height = req['HEIGHT']
        if 'FORMAT' in req:
            self.format = req['FORMAT']
        if 'RESX' in req:
            self.resx = req['RESX']
        if 'RESY' in req:
            self.resy = req['RESY']
        if 'INTERPOLATION' in req:
            self.interpolation = req['INTERPOLATION']
        if 'PARAMETER' in req:
            self.parameter = req['PARAMETER']
        if 'EXCEPTIONS' in req:
            self.exceptions = req['EXCEPTIONS']


def get_wcs_capabilities_xml(ctx: WcsContext, base_url: str) -> str:
    """
    Get WCSCapabilities.xml according to
    https://schemas.opengis.net/wcs/1.0.0/.

    :param ctx: server context
    :param base_url: the request base URL
    :return: XML plain text in UTF-8 encoding
    """
    element = _get_capabilities_element(ctx, base_url)
    document = Document(element)
    return document.to_xml(indent=4)


def get_describe_coverage_xml(ctx: WcsContext, coverages: List[str] = None) -> str:
    element = _get_describe_element(ctx, coverages)
    document = Document(element)
    return document.to_xml(indent=4)


def translate_to_generator_request(ctx: WcsContext, req: CoverageRequest) \
        -> CubeGeneratorRequest:
    data_id = _get_input_data_id(ctx, req)
    bbox = []
    for v in req.bbox.split(' '):
        bbox.append(float(v))

    return CubeGeneratorRequest.from_dict(
        {
            # todo - put generic data store here
            'input_config': {
                'store_id': 'file',
                'store_params': {
                    'root': '../../../../examples/serve/demo'
                },
                'data_id': f'{data_id}'
            },
            'cube_config': {
                'variable_names': [f'{req.coverage}'.split('.')[-1]],
                'crs': f'{req.crs}',
                'bbox': tuple(bbox)
            },
            'output_config': {
                'store_id': 'memory',
                'replace': True,
                'data_id': f'{req.coverage}.zarr',
            }
        }
    )


def get_coverage(ctx: WcsContext, req: CoverageRequest) -> Dataset:
    _validate_coverage_req(ctx, req)
    gen_req = translate_to_generator_request(ctx, req)

    gen = CubeGenerator.new()

    result = gen.generate_cube(request=gen_req)
    if not result.status == 'ok':
        raise ValueError(f'Failed to generate cube: {result.message}')

    memory_store = sp.get_data_store_instance('memory')
    cube_id = list(memory_store.store.get_data_ids())[0]
    cube = memory_store.store.open_data(cube_id)

    # _write_debug_output(cube)

    return cube


def _write_debug_output(cube):
    history = str(cube.history[0])
    del cube.attrs['history']
    cube['history'] = history
    cw = CubeWriter(OutputConfig('file',
                                 writer_id='dataset:netcdf:file',
                                 data_id='/../../../test_cube.nc'))
    cw.write_cube(cube, GridMapping.from_dataset(cube))


def _get_output_region(req: CoverageRequest) -> Optional[tuple[float, ...]]:
    if not req.bbox:
        return None

    output_region = []
    for v in req.bbox.split(' '):
        output_region.append(float(v))
    return tuple(output_region)


def _get_input_data_id(ctx: WcsContext, req: CoverageRequest) -> str:
    for dataset_config in ctx.datasets_ctx.get_dataset_configs():
        ds_name = dataset_config['Identifier']
        ds = ctx.datasets_ctx.get_dataset(ds_name)

        var_names = sorted(ds.data_vars)
        for var_name in var_names:
            qualified_var_name = f'{ds_name}.{var_name}'
            if req.coverage == qualified_var_name:
                return dataset_config['Path']
    raise RuntimeError('Should never come here. Contact the developers.')


def _get_input_path(ctx: WcsContext, req: CoverageRequest) -> str:
    for dataset_config in ctx.datasets_ctx.get_dataset_configs():
        ds_name = dataset_config['Identifier']
        ds = ctx.datasets_ctx.get_dataset(ds_name)

        var_names = sorted(ds.data_vars)
        for var_name in var_names:
            qualified_var_name = f'{ds_name}.{var_name}'
            if req.coverage == qualified_var_name:
                path = dataset_config['Path']
                break
        store_instance_id = dataset_config['StoreInstanceId']
        store = ctx.datasets_ctx.get_data_store_pool(). \
            get_store(store_instance_id)
        return store.root + '/' + path
    raise RuntimeError('Should never come here. Contact the developers.')


def _get_input_store_id(ctx: WcsContext, req: CoverageRequest) -> str:
    for dataset_config in ctx.datasets_ctx.get_dataset_configs():
        ds_name = dataset_config['Identifier']
        ds = ctx.datasets_ctx.get_dataset(ds_name)

        var_names = sorted(ds.data_vars)
        for var_name in var_names:
            qualified_var_name = f'{ds_name}.{var_name}'
            if req.coverage == qualified_var_name:
                return dataset_config['StoreInstanceId']
    raise RuntimeError('Should never come here. Contact the developers.')


def _validate_coverage_req(ctx: WcsContext, req: CoverageRequest):
    def _has_no_invalid_bbox() -> bool:
        if req.bbox:
            return _is_valid_bbox(req.bbox)
        else:
            return True

    def _has_no_invalid_time() -> bool:
        if req.time:
            return _is_valid_time(req.time)
        else:
            return True

    if req.coverage and _is_valid_coverage(ctx, req.coverage) \
            and req.crs and _is_valid_crs(req.crs) \
            and ((req.bbox and _is_valid_bbox(req.bbox))
                 or (req.time and _is_valid_time(req.time))) \
            and _has_no_invalid_bbox \
            and _has_no_invalid_time() \
            and ((req.width and req.height)
                 or (req.resx and req.resy)) \
            and ((req.width and not req.resx)
                 or (req.width and not req.resy)
                 or (req.height and not req.resx)
                 or (req.height and not req.resy)
                 or (req.resx and not req.width)
                 or (req.resx and not req.height)
                 or (req.resy and not req.width)
                 or (req.resy and not req.height)) \
            and req.format and _is_valid_format(req.format) \
            and not req.parameter \
            and not req.interpolation \
            and not req.exceptions:
        return
    elif not req.coverage or not _is_valid_coverage(ctx, req.coverage):
        raise ValueError('No valid value for parameter COVERAGE provided. '
                         'COVERAGE must be a variable name prefixed with '
                         'its dataset name. Example: my_dataset.my_var')
    elif req.parameter:
        raise ValueError('PARAMETER not yet supported')
    elif req.interpolation:
        raise ValueError('INTERPOLATION not yet supported')
    elif req.exceptions:
        raise ValueError('EXCEPTIONS not yet supported')
    elif ((req.width and not req.height)
          or (req.height and not req.width)
          or (req.resx and not req.resy)
          or (req.resy and not req.resx)
          or (req.width and req.resx or req.resy)
          or (req.height and req.resx or req.resy)):
        raise ValueError('Either both WIDTH and HEIGHT, or both RESX and RESY '
                         'must be provided.')
    elif not req.format or not _is_valid_format(req.format):
        raise ValueError('FORMAT wrong or missing. Must be one of ' +
                         ', '.join(_get_formats_list()))
    elif True:
        raise ValueError('Reason unclear, fix me')


def _is_valid_coverage(ctx: WcsContext, coverage: str) -> bool:
    band_infos = _extract_band_infos(ctx, [coverage])
    if band_infos:
        return True
    return False


def _is_valid_crs(crs: str) -> bool:
    return crs in VALID_CRS_LIST


def _is_valid_bbox(bbox: str) -> bool:
    bbox_regex = re.compile(r'-?\d{1,3} -?\d{1,2} -?\d{1,3} -?\d{1,2}')
    if not bbox_regex.match(bbox):
        raise ValueError('BBOX must be given as `minx miny maxx maxy`')
    return True


def _is_valid_format(format_req: str) -> bool:
    return format_req in _get_formats_list()


def _is_valid_time(time: str) -> bool:
    try:
        datetime.fromisoformat(time)
    except ValueError:
        raise ValueError('TIME value must be given in the format'
                         '\'YYYY-MM-DD[*HH[:MM[:SS[.mmm[mmm]]]]'
                         '[+HH:MM[:SS[.ffffff]]]]\'')
    return True


# noinspection HttpUrlsUsage
def _get_capabilities_element(ctx: WcsContext,
                              base_url: str) -> Element:
    service_element = _get_service_element(ctx)
    capability_element = _get_capability_element(base_url)
    content_element = Element('ContentMetadata')

    band_infos = _extract_band_infos(ctx)
    for var_name in band_infos.keys():
        content_element.add(Element('CoverageOfferingBrief', elements=[
            Element('name', text=var_name),
            Element('label', text=band_infos[var_name].label),
            Element('lonLatEnvelope', elements=[
                Element('gml:pos', text=f'{band_infos[var_name].bbox[0]}'
                                        f' {band_infos[var_name].bbox[1]}'),
                Element('gml:pos', text=f'{band_infos[var_name].bbox[2]}'
                                        f' {band_infos[var_name].bbox[3]}')
            ])
        ]))

    return Element(
        'WCS_Capabilities',
        attrs={
            'xmlns': "http://www.opengis.net/wcs",
            'xmlns:gml': "http://www.opengis.net/gml",
            'xmlns:xlink': "http://www.w3.org/1999/xlink",
            'version': WCS_VERSION,
        },
        elements=[
            service_element,
            capability_element,
            content_element
        ]
    )


def _get_service_element(ctx: WcsContext) -> Element:
    service_provider = ctx.config.get('ServiceProvider')

    def _get_value(path):
        v = None
        node = service_provider
        for k in path:
            if not isinstance(node, dict) or k not in node:
                return ''
            v = node[k]
            node = v
        return str(v) if v is not None else ''

    def _get_individual_name():
        individual_name = _get_value(['ServiceContact', 'IndividualName'])
        individual_name = tuple(individual_name.split(' ').__reversed__())
        return '{}, {}'.format(*individual_name)

    element = Element('Service', elements=[
        Element('description',
                text=_get_value(['WCS-description'])),
        Element('name',
                text=_get_value(['WCS-name'])),
        Element('label',
                text=_get_value(['WCS-label'])),
        Element('keywords', elements=[
            Element('keyword', text=k) for k in service_provider['keywords']
        ]),
        Element('responsibleParty', elements=[
            Element('individualName',
                    text=_get_individual_name()),
            Element('organisationName',
                    text=_get_value(['ProviderName'])),
            Element('positionName',
                    text=_get_value(['ServiceContact',
                                     'PositionName'])),
            Element('contactInfo', elements=[
                Element('phone', elements=[
                    Element('voice',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Phone',
                                             'Voice'])),
                    Element('facsimile',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Phone',
                                             'Facsimile'])),
                ]),
                Element('address', elements=[
                    Element('deliveryPoint',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'DeliveryPoint'])),
                    Element('city',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'City'])),
                    Element('administrativeArea',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'AdministrativeArea'])),
                    Element('postalCode',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'PostalCode'])),
                    Element('country',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'Country'])),
                    Element('electronicMailAddress',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'ElectronicMailAddress'])),
                ]),
                Element('onlineResource', attrs={
                    'xlink:href': _get_value(['ProviderSite'])})
            ]),
        ]),
        Element('fees', text='NONE'),
        Element('accessConstraints', text='NONE')
    ])
    return element


def _get_capability_element(base_url: str) -> Element:
    get_capabilities_url = f'{base_url}?service=WCS&amp;version=1.0.0&amp;' \
                           f'request=GetCapabilities'
    describe_url = f'{base_url}?service=WCS&amp;version=1.0.0&amp;' \
                   f'request=DescribeCoverage'
    get_url = f'{base_url}?service=WCS&amp;version=1.0.0&amp;' \
              f'request=GetCoverage'
    return Element('Capability', elements=[
        Element('Request', elements=[
            Element('GetCapabilities', elements=[
                Element('DCPType', elements=[
                    Element('HTTP', elements=[
                        Element('Get', elements=[
                            Element('OnlineResource',
                                    attrs={'xlink:href': get_capabilities_url})
                        ])
                    ])
                ])
            ]),
            Element('DescribeCoverage', elements=[
                Element('DCPType', elements=[
                    Element('HTTP', elements=[
                        Element('Get', elements=[
                            Element('OnlineResource',
                                    attrs={'xlink:href': describe_url})
                        ])
                    ])
                ])
            ]),
            Element('GetCoverage', elements=[
                Element('DCPType', elements=[
                    Element('HTTP', elements=[
                        Element('Get', elements=[
                            Element('OnlineResource',
                                    attrs={'xlink:href': get_url})
                        ])
                    ])
                ])
            ]),
        ]),
        Element('Exception', elements=[
            Element('Format', text='application/x-ogc-wcs')
        ])
    ])


def _get_describe_element(ctx: WcsContext, coverages: List[str] = None) \
        -> Element:
    coverage_elements = []

    band_infos = _extract_band_infos(ctx, coverages, True)
    for var_name in band_infos.keys():
        coverage_elements.append(Element('CoverageOffering', elements=[
            Element('description', text=band_infos[var_name].label),
            Element('name', text=var_name),
            Element('label', text=band_infos[var_name].label),
            Element('lonLatEnvelope', elements=[
                Element('gml:pos', text=f'{band_infos[var_name].bbox[0]} '
                                        f'{band_infos[var_name].bbox[1]}'),
                Element('gml:pos', text=f'{band_infos[var_name].bbox[2]} '
                                        f'{band_infos[var_name].bbox[3]}')
            ]),
            Element('keywords', elements=[
                Element('keyword', text='grid')
            ]),
            Element('domainSet', elements=[
                Element('spatialDomain', elements=[
                    Element('gml:Envelope', elements=[
                        Element('gml:pos',
                                text=f'{band_infos[var_name].bbox[0]} '
                                     f'{band_infos[var_name].bbox[1]}'),
                        Element('gml:pos',
                                text=f'{band_infos[var_name].bbox[2]} '
                                     f'{band_infos[var_name].bbox[3]}')
                    ])
                ]),
                Element('temporalDomain', elements=[
                    Element('gml:timePosition', text=time_step)
                    for time_step in band_infos[var_name].time_steps])
            ]),
            Element('rangeSet', elements=[
                Element('RangeSet', elements=[
                    Element('name', text=var_name),
                    Element('label', text=band_infos[var_name].label),
                    Element('axisDescription', elements=[
                        Element('AxisDescription', elements=[
                            Element('name', text='Band'),
                            Element('label', text='Band'),
                            Element('values', elements=[
                                Element('interval', elements=[
                                    Element('min', text=
                                    f'{band_infos[var_name].min:0.4f}'),
                                    Element('max', text=
                                    f'{band_infos[var_name].max:0.4f}')
                                ])
                            ]),
                        ])
                    ])
                ])
            ]),
            Element('supportedCRSs', elements=[
                Element('requestResponseCRSs', text=' '.join(VALID_CRS_LIST))
            ]),
            Element('supportedFormats', elements=[
                Element('formats', text=f) for f in _get_formats_list()
            ])
        ]))

    return Element(
        'CoverageDescription',
        attrs={
            'xmlns': "http://www.opengis.net/wcs",
            'xmlns:gml': "http://www.opengis.net/gml",
            'version': WCS_VERSION,
        },
        elements=coverage_elements
    )


def _get_formats_list() -> List[str]:
    formats = get_extension_registry().find_extensions(
        EXTENSION_POINT_DATASET_IOS,
        lambda e: 'w' in e.metadata.get('modes', set())
    )
    return [ext.name for ext in formats if not ext.name == 'mem']


class BandInfo:

    def __init__(self, var_name: str, label: str,
                 bbox: tuple[float, float, float, float],
                 time_steps: list[str]):
        self.var_name = var_name
        self.label = label
        self.bbox = bbox
        self.min = np.nan
        self.max = np.nan
        self.time_steps = time_steps


def _extract_band_infos(ctx: WcsContext, coverages: List[str] = None,
                        full: bool = False) -> Dict[str, BandInfo]:
    band_infos = {}
    for dataset_config in ctx.datasets_ctx.get_dataset_configs():
        ds_name = dataset_config['Identifier']
        ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_name)
        grid_mapping = ml_dataset.grid_mapping
        ds = ml_dataset.base_dataset

        try:
            bbox = get_crs84_bbox(grid_mapping)
        except ValueError:
            warnings.warn(f'cannot compute geographical'
                          f' bounds for dataset {ds_name}, ignoring it')
            continue

        x_name, y_name = grid_mapping.xy_dim_names

        var_names = sorted(ds.data_vars)
        for var_name in var_names:
            qualified_var_name = f'{ds_name}.{var_name}'
            if coverages and qualified_var_name not in coverages:
                continue
            var = ds[var_name]

            label = var.long_name if hasattr(var, 'long_name') else var_name
            is_spatial_var = var.ndim >= 2 \
                             and var.dims[-1] == x_name \
                             and var.dims[-2] == y_name
            if not is_spatial_var:
                continue

            is_temporal_var = var.ndim >= 3
            time_steps = None
            if is_temporal_var:
                time_steps = [f'{str(d)[:19]}Z' for d in var.time.values]

            band_info = BandInfo(qualified_var_name, label, bbox, time_steps)
            if full:
                nn_values = var.values[~np.isnan(var.values)]
                band_info.min = nn_values.min()
                band_info.max = nn_values.max()

            band_infos[f'{ds_name}.{var_name}'] = band_info

    return band_infos