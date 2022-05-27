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

import abc
from typing import Any, Optional

from xcube.server.config import Config


class Context(abc.ABC):
    """An abstract context."""

    @property
    @abc.abstractmethod
    def config(self) -> Config:
        """The current server configuration."""

    @abc.abstractmethod
    def get_api_ctx(self, api_name: str) -> Optional["Context"]:
        """
        Get the API context for *api_name*.
        Can be used to access context objects of other APIs.

        :param api_name: The name of a registered API.
        :return: The API context for *api_name*, or None if no such exists.
        """

    @abc.abstractmethod
    def update(self, prev_ctx: Optional["Context"]):
        """
        Update this context with respect to the current configuration.
        Called when the server configuration changed.

        :param prev_ctx: The previous context instance, or None
            if update() is called for the first time (= initialisation).
        """

    @abc.abstractmethod
    def dispose(self):
        """
        Dispose all allocated resources.
        Called if this context will never be used again.
        """