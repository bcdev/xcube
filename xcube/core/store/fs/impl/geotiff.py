# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional, Tuple, Dict, Any

import fsspec
import rasterio
import xarray as xr

from xcube.core.mldataset import LazyMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataType
from xcube.core.store import MULTI_LEVEL_DATASET_TYPE
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from .dataset import DatasetGeoTiffFsDataAccessor


class GeoTIFFMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset for GeoTIFF format.

    Args:
        fs: fsspec.AbstractFileSystem object.
        root: Optional root path identifier.
        data_id: dataset identifier.
        open_params: keyword arguments.
    """

    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        root: Optional[str],
        data_id: str,
        **open_params: dict[str, Any],
    ):
        super().__init__(ds_id=data_id)
        self._fs = fs
        self._root = root
        self._path = data_id
        self._open_params = open_params
        self._file_url = None

    def _get_overview_count(self):
        with rasterio.open(self._file_url) as rio_dataset:
            overviews = rio_dataset.overviews(1)
        return overviews

    def _get_num_levels_lazily(self) -> int:
        self._file_url = self._get_file_url()
        if isinstance(self._fs, fsspec.AbstractFileSystem):
            with DatasetGeoTiffFsDataAccessor.create_env_session(self._fs):
                overviews = self._get_overview_count()
        else:
            assert_true(self._fs is None, message="invalid type for fs")
        return len(overviews) + 1

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        tile_size = self._open_params.get("tile_size", (512, 512))
        self._file_url = self._get_file_url()
        return DatasetGeoTiffFsDataAccessor.open_dataset(
            self._fs,
            self._file_url,
            tile_size,
            overview_level=index - 1 if index > 0 else None,
        )

    def _get_file_url(self):
        if isinstance(self._fs.protocol, str):
            protocol = self._fs.protocol
        else:
            protocol = self._fs.protocol[0]
        return protocol + "://" + self._path


MULTI_LEVEL_GEOTIFF_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        tile_size=JsonArraySchema(
            items=(
                JsonNumberSchema(minimum=256, default=512),
                JsonNumberSchema(minimum=256, default=512),
            ),
            default=[512, 512],
        ),
    ),
    additional_properties=False,
)


# noinspection PyAbstractClass
class MultiLevelDatasetGeoTiffFsDataAccessor(DatasetGeoTiffFsDataAccessor):
    """
    Opener/writer extension name: "mldataset:geotiff:<protocol>"
    """

    @classmethod
    def get_data_type(cls) -> DataType:
        return MULTI_LEVEL_DATASET_TYPE

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return MULTI_LEVEL_GEOTIFF_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> MultiLevelDataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)
        return GeoTIFFMultiLevelDataset(fs, root, data_id, **open_params)
