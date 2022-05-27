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

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

DATASET_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        data_id=JsonStringSchema(),
        open_params=JsonObjectSchema(additional_properties=True),
        style_ref=JsonStringSchema(),
    ),
    required=['data_id'],
    additional_properties=False
)

DATA_STORE_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        store_id=JsonStringSchema(),
        store_params=JsonObjectSchema(additional_properties=True),
        datasets=JsonArraySchema(
            items=DATASET_CONFIG_SCHEMA)
    ),
    required=['store_id'],
    additional_properties=False
)

DATASETS_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        default_cache_size=JsonStringSchema(),
        data_stores=JsonArraySchema(
            items=DATA_STORE_CONFIG_SCHEMA
        )
    ),
    additional_properties=False
)