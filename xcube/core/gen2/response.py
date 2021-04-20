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

from typing import Dict, Any

from xcube.core.store import DatasetDescriptor
from xcube.util.jsonschema import JsonObjectSchema
from .config import JsonObject


class CubeInfo(JsonObject):
    def __init__(self,
                 dataset_descriptor: DatasetDescriptor,
                 size_estimation: Dict[str, Any]):
        self.dataset_descriptor: DatasetDescriptor = dataset_descriptor
        self.size_estimation: dict = size_estimation

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(dataset_descriptor=DatasetDescriptor.get_schema(),
                                                size_estimation=JsonObjectSchema(additional_properties=True)),
                                required=['dataset_descriptor', 'size_estimation'],
                                additional_properties=False,
                                factory=cls)
