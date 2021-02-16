# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Sequence

from xcube.core.gen2.config import CubeConfig
from xcube.core.gen2.config import InputConfig
from xcube.core.normalize import normalize_dataset
from xcube.core.store import DataStoreError
from xcube.core.store import DataStorePool
from xcube.core.store import TYPE_SPECIFIER_CUBE
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import get_data_store_instance
from xcube.core.store import new_data_opener
from xcube.util.progress import observe_progress


def open_cubes(input_configs: Sequence[InputConfig],
               cube_config: CubeConfig,
               store_pool: DataStorePool = None):
    cubes = []
    all_cube_params = cube_config.to_dict()
    with observe_progress('Opening input(s)', len(input_configs)) as progress:
        for input_config in input_configs:
            opener_id = input_config.opener_id
            store_params = input_config.store_params or {}
            open_params = input_config.open_params or {}
            normalisation_required = False
            if input_config.store_id:
                store_instance = get_data_store_instance(input_config.store_id,
                                                         store_params=store_params,
                                                         store_pool=store_pool)
                store = store_instance.store
                if opener_id is None:
                    opener_ids = None
                    type_specifiers = store.get_type_specifiers_for_data(input_config.data_id)
                    for type_specifier in type_specifiers:
                        if TYPE_SPECIFIER_CUBE.is_satisfied_by(type_specifier):
                            opener_ids = \
                                store.get_data_opener_ids(data_id=input_config.data_id,
                                                          type_specifier=type_specifier)
                            break
                    if not opener_ids:
                        for type_specifier in type_specifiers:
                            if TYPE_SPECIFIER_DATASET.is_satisfied_by(type_specifier):
                                opener_ids = \
                                    store.get_data_opener_ids(data_id=input_config.data_id,
                                                              type_specifier=type_specifier)
                                normalisation_required = True
                                break
                    if not opener_ids:
                        raise DataStoreError(f'Data store "{input_config.store_id}" '
                                             f'does not support datasets')
                    opener_id = opener_ids[0]
                opener = store
                open_params.update(opener_id=opener_id, **open_params)
            else:
                opener = new_data_opener(opener_id)
                open_params.update(**store_params, **open_params)
            open_params_schema = opener.get_open_data_params_schema(input_config.data_id)
            cube_params = {k: v for k, v in all_cube_params.items() if k in open_params_schema.properties}
            cube = opener.open_data(input_config.data_id, **open_params, **cube_params)
            if normalisation_required:
                cube = normalize_dataset(cube)
            cubes.append(cube)
            progress.worked(1)

    return cubes
