# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from abc import abstractmethod, ABC
from typing import Any, Optional

import xarray as xr

from xcube.constants import EXTENSION_POINT_DATA_OPENERS
from xcube.constants import EXTENSION_POINT_DATA_WRITERS
from xcube.util.assertions import assert_given
from xcube.util.extension import Extension
from xcube.util.extension import ExtensionPredicate
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.plugin import get_extension_registry
from .preload import PreloadHandle
from .datatype import DataType
from .datatype import DataTypeLike
from .error import DataStoreError


#######################################################
# Data accessor instantiation and registry query
#######################################################


def new_data_opener(
    opener_id: str,
    extension_registry: Optional[ExtensionRegistry] = None,
    **opener_params,
) -> "DataOpener":
    """Get an instance of the data opener identified by *opener_id*.

    The optional, extra opener parameters *opener_params* may
    be used by data store (``xcube.core.store.DataStore``)
    implementations so they can share their internal state with the opener.

    Args:
        opener_id: The data opener identifier.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.
        **opener_params: Extra opener parameters.

    Returns:
        A data opener instance.
    """
    assert_given(opener_id, "opener_id")
    extension_registry = extension_registry or get_extension_registry()
    if not extension_registry.has_extension(EXTENSION_POINT_DATA_OPENERS, opener_id):
        raise DataStoreError(f"A data opener named" f" {opener_id!r} is not registered")
    return extension_registry.get_component(EXTENSION_POINT_DATA_OPENERS, opener_id)(
        **opener_params
    )


def new_data_writer(
    writer_id: str,
    extension_registry: Optional[ExtensionRegistry] = None,
    **writer_params,
) -> "DataWriter":
    """Get an instance of the data writer identified by *writer_id*.

    The optional, extra writer parameters *writer_params* may be used by
    data store (``xcube.core.store.DataStore``) implementations so they
    can share their internal state with the writer.

    Args:
        writer_id: The data writer identifier.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.
        **writer_params: Extra writer parameters.

    Returns:
        A data writer instance.
    """
    assert_given(writer_id, "writer_id")
    extension_registry = extension_registry or get_extension_registry()
    if not extension_registry.has_extension(EXTENSION_POINT_DATA_WRITERS, writer_id):
        raise DataStoreError(f"A data writer named" f" {writer_id!r} is not registered")
    return extension_registry.get_component(EXTENSION_POINT_DATA_WRITERS, writer_id)(
        **writer_params
    )


def find_data_opener_extensions(
    predicate: ExtensionPredicate = None,
    extension_registry: Optional[ExtensionRegistry] = None,
) -> list[Extension]:
    """Get registered data opener extensions using the optional
    filter function *predicate*.

    Args:
        predicate: An optional filter function.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.

    Returns:
        List of matching extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(
        EXTENSION_POINT_DATA_OPENERS, predicate=predicate
    )


def find_data_writer_extensions(
    predicate: ExtensionPredicate = None,
    extension_registry: Optional[ExtensionRegistry] = None,
) -> list[Extension]:
    """Get registered data writer extensions using the optional filter
    function *predicate*.

    Args:
        predicate: An optional filter function.
        extension_registry: Optional extension registry. If not given,
            the global extension registry will be used.

    Returns:
        List of matching extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(
        EXTENSION_POINT_DATA_WRITERS, predicate=predicate
    )


def get_data_accessor_predicate(
    data_type: DataTypeLike = None, format_id: str = None, storage_id: str = None
) -> ExtensionPredicate:
    """Get a predicate that checks if the name of a data accessor extension is
    compliant with *data_type*, *format_id*, *storage_id*.

    Args:
        data_type: Optional data data type to be supported. May be given
            as type alias name, as a type, or as a DataType instance.
        format_id: Optional data format identifier to be supported.
        storage_id: Optional data storage identifier to be supported.

    Returns:
        A filter function.

    Raises:
        DataStoreError: If an error occurs.
    """
    if any((data_type, format_id, storage_id)):
        data_type = DataType.normalize(data_type) if data_type is not None else None

        def _predicate(extension: Extension) -> bool:
            extension_parts = extension.name.split(":", maxsplit=4)
            if storage_id is not None:
                ext_storage_id = extension_parts[2]
                if ext_storage_id != "*" and ext_storage_id != storage_id:
                    return False
            if format_id is not None:
                ext_format_id = extension_parts[1]
                if ext_format_id != "*" and ext_format_id != format_id:
                    return False
            if data_type is not None:
                ext_data_type = DataType.normalize(extension_parts[0])
                if not data_type.is_super_type_of(ext_data_type):
                    return False
            return True

    else:
        # noinspection PyUnusedLocal
        def _predicate(extension: Extension) -> bool:
            return True

    return _predicate


#######################################################
# Interface Classes
#######################################################


class DataOpener(ABC):
    """An interface that specifies a parameterized `open_data()` operation.

    Possible open parameters are implementation-specific and
    are described by a JSON Schema.

    Note this interface uses the term "opener" to underline the expected
    laziness of the operation. For example, when a xarray.Dataset is
    returned from a Zarr directory, the actual data is represented by
    Dask arrays and will be loaded only on-demand.
    """

    @abstractmethod
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *open_params* to
        :meth:`open_data`.
        If *data_id* is given, the returned schema will be tailored
        to the constraints implied by the identified data resource.
        Some openers might not support this, therefore *data_id*
        is optional, and if it is omitted, the returned schema will be
        less restrictive.

        Args:
            data_id: An optional data resource identifier.

        Returns:
            The schema for the parameters in *open_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def open_data(self, data_id: str, **open_params) -> Any:
        """Open the data resource given by the data resource identifier
        *data_id* using the supplied *open_params*.

        Raises if *data_id* does not exist.

        Args:
            data_id: The data resource identifier.
            **open_params: Opener-specific parameters.

        Returns:
            An xarray.Dataset instance.

        Raises:
            DataStoreError: If an error occurs.
        """

    def close(self):
        """Closes this data opener.

        Should be called if the data opener is no longer needed.

        This method may close local files, remote connections, or release
        allocated resources.

        The default implementation does nothing.
        """

    def __del__(self):
        """Overridden to call ``close()``."""
        self.close()


class DataDeleter(ABC):
    """An interface that specifies a parameterized `delete_data()` operation.

    Possible delete parameters are implementation-specific and
    are described by a JSON Schema.
    """

    @abstractmethod
    def get_delete_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *delete_params*
        to :meth:`delete_data`.
        If *data_id* is given, the returned schema will be tailored to
        the constraints implied by the identified data resource.
        Some deleters might not support this, therefore *data_id*
        is optional, and if it is omitted, the returned schema will
        be less restrictive.

        Args:
            data_id: An optional data resource identifier.

        Returns:
            The schema for the parameters in *delete_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def delete_data(self, data_id: str, **delete_params):
        """Delete a data resource. Raises if *data_id* does not exist.

        Args:
            data_id: A data resource identifier known to exist.
            **delete_params: Deleter-specific parameters.

        Raises:
            DataStoreError: If an error occurs.
        """


class DataWriter(DataDeleter, ABC):
    """An interface that specifies a parameterized `write_data()` operation.

    Possible write parameters are implementation-specific and
    are described by a JSON Schema.
    """

    @abstractmethod
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        """Get the schema for the parameters passed as *write_params* to
        :meth:`write_data`.

        Returns:
            The schema for the parameters in *write_params*.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def write_data(
        self, data: Any, data_id: str, replace: bool = False, **write_params
    ) -> str:
        """Write a data resource using the supplied *data_id* and *write_params*.

        Args:
            data: The data resource's in-memory representation to be
                written.
            data_id: A unique data resource identifier.
            replace: Whether to replace an existing data resource.
            **write_params: Writer-specific parameters.

        Returns:
            The data resource identifier used to write the data
            resource.

        Raises:
            DataStoreError: If an error occurs.
        """


class DataPreloader(ABC):
    """An interface that specifies a parameterized `preload_data()` operation.

    Warning: This is an experimental and potentially unstable API
    introduced in xcube 1.8.

    Many data store implementations rely on remote data APIs.
    Such API may provide only limited data access performance.
    Hence, the approach taken by ``DataStore.open_data(data_id, ...)`` alone
    is suboptimal for a user's perspective. This is because the method is
    blocking as it is not asynchronous, it may take long time before it
    returns, and it cannot report any progress while doing so.
    The reasons for slow and unresponsive data APIs are manifold: intended
    access is by file download, access is bandwidth limited, or not allowing
    for sub-setting.

Data stores may implement the ``preload_data()`` method differently—or not at all.
In most cases, if preloading is required, the data will be downloaded and stored
temporarily in a cache for access.
    """

    @abstractmethod
    def preload_data(
        self,
        *data_ids: str,
        **preload_params: Any,
    ) -> PreloadHandle:
        """Preload the given data items for faster access.

        Warning: This is an experimental and potentially unstable API
        introduced in xcube 1.8.

        The method may be blocking or non-blocking.
        Implementations may offer the following keyword arguments
        in *preload_params*:

        - ``blocking``: whether the preload process is blocking.
          Should be `True` by default if supported.
        - ``monitor``: a callback function that serves as a progress monitor.
          It receives the preload handle and the recent partial state update.

        Args:
            data_ids: Data identifiers to be preloaded.
            preload_params: data store specific preload parameters.
              See method ``get_preload_data_params_schema()`` for information
              on the possible options.

        Returns:
            A handle for the preload process. The default implementation
            returns an empty preload handle.
        """

    # noinspection PyMethodMayBeStatic
    @abstractmethod
    def get_preload_data_params_schema(self) -> JsonObjectSchema:
        """Get the JSON schema that describes the keyword
        arguments that can be passed to ``preload_data()``.

        Returns:
            A ``JsonObjectSchema`` object whose properties describe
            the parameters of ``preload_data()``.
        """


class DataTimeSliceUpdater(DataWriter, ABC):
    """An interface that specifies writing of time slice data."""

    @abstractmethod
    def append_data_time_slice(self, data_id: str, time_slice: xr.Dataset):
        """Append a time slice to the identified data resource.

        Args:
            data_id: The data resource identifier.
            time_slice: The time slice data to be inserted. Must be
                compatible with the data resource.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def insert_data_time_slice(self, data_id: str, time_slice: Any, time_index: int):
        """Insert a time slice into the identified data resource at given index.

        Args:
            data_id: The data resource identifier.
            time_slice: The time slice data to be inserted. Must be
                compatible with the data resource.
            time_index: The time index.

        Raises:
            DataStoreError: If an error occurs.
        """

    @abstractmethod
    def replace_data_time_slice(self, data_id: str, time_slice: Any, time_index: int):
        """Replace a time slice in the identified data resource at given index.

        Args:
            data_id: The data resource identifier.
            time_slice: The time slice data to be inserted. Must be
                compatible with the data resource.
            time_index: The time index.

        Raises:
            DataStoreError: If an error occurs.
        """
