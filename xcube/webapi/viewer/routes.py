# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import importlib.resources
import json
import os
import pathlib
import random
from typing import Union, Optional

from chartlets import Response as ExtResponse
from chartlets.controllers import get_callback_results
from chartlets.controllers import get_contributions
from chartlets.controllers import get_layout

from xcube.constants import LOG
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from xcube.util.jsonencoder import NumpyJSONEncoder
from .api import api
from .context import ViewerContext

ENV_VAR_XCUBE_VIEWER_PATH = "XCUBE_VIEWER_PATH"

_viewer_module = "xcube.webapi.viewer"
_data_dir = "dist"
_default_filename = "index.html"

_responses = {
    200: {
        "description": "OK",
        "content": {
            "text/html": {
                "schema": {"type": "string"},
            },
        },
    }
}


@api.static_route(
    "/viewer",
    default_filename=_default_filename,
    summary="Brings up the xcube Viewer webpage",
    responses=_responses,
)
def get_local_viewer_path() -> Union[None, str, pathlib.Path]:
    local_viewer_path = os.environ.get(ENV_VAR_XCUBE_VIEWER_PATH)
    if local_viewer_path:
        return local_viewer_path
    try:
        with importlib.resources.files(_viewer_module) as path:
            local_viewer_path = path / _data_dir
            if (local_viewer_path / _default_filename).is_file():
                return local_viewer_path
    except ImportError:
        pass
    LOG.warning(
        f"Cannot find {_data_dir}/{_default_filename}"
        f" in {_viewer_module}, consider setting environment variable"
        f" {ENV_VAR_XCUBE_VIEWER_PATH}",
        exc_info=True,
    )
    return None


@api.route("/viewer/config/{*path}")
class ViewerConfigHandler(ApiHandler[ViewerContext]):
    @api.operation(
        operationId="getViewerConfigurationItem",
        summary="Get a configuration item for the xcube viewer.",
    )
    def get(self, path: Optional[str]):
        config_items = self.ctx.config_items
        if config_items is None:
            raise ApiError.NotFound(f"xcube viewer has" f" not been been configured")
        try:
            data = config_items[path]
        except KeyError:
            raise ApiError.NotFound(
                f"The item {path!r} was not found" f" in viewer configuration"
            )
        content_type = self.get_content_type(path)
        self.response.set_header("Content-Length", str(len(data)))
        if content_type is not None:
            self.response.set_header("Content-Type", content_type)
        self.response.write(data)

    @staticmethod
    def get_content_type(path: str) -> Optional[str]:
        filename_ext = path.split("/")[-1].split(".")[-1]
        if filename_ext in ("json", "xml"):
            return f"application/{filename_ext}; charset=UTF-8"
        elif filename_ext in ("csv", "html", "css", "js"):
            return f"text/{filename_ext}; charset=UTF-8"
        elif filename_ext in ("geojson",):
            return "application/geo+json; charset=UTF-8"
        elif filename_ext in ("gif", "png"):
            return f"image/{filename_ext}"
        elif filename_ext in ("jpeg", "jpg"):
            return "image/jpeg"
        return None


@api.route("/viewer/state")
class ViewerStateHandler(ApiHandler[ViewerContext]):

    # noinspection SpellCheckingInspection
    _id_chars = "abcdefghijklmnopqrstuvwxyz0123456789"

    @api.operation(
        operationId="getViewerState",
        summary="Get previously stored viewer state keys or a specific state.",
        parameters=[
            {
                "name": "key",
                "in": "query",
                "description": "The key of a previously stored state.",
                "required": False,
                "schema": {"type": "string"},
            }
        ],
    )
    def get(self):
        if self.ctx.persistence is None:
            # 501: Not Implemented
            self.response.set_status(501, "Persistence not supported")
            return
        key = self.request.get_query_arg("key", type=str, default="")
        if key:
            if key not in self.ctx.persistence:
                self.response.set_status(404, "State not found")
                return
            state = self.ctx.persistence[key]
            LOG.info(f"Restored state ({len(state)} bytes) for key {key!r}")
            self.response.write(state)
        else:
            keys = list(self.ctx.persistence.keys())
            self.response.write({"keys": keys})

    @api.operation(
        operationId="putViewerState",
        summary="Store a viewer state and return a state key.",
    )
    def put(self):
        if self.ctx.persistence is None:
            # 501: Not Implemented
            self.response.set_status(501, "Persistence not supported")
            return
        state = self.request.body
        key = self.new_key()
        self.ctx.persistence[key] = state
        LOG.info(f"Stored state ({len(state)} bytes) using key {key!r}")
        self.response.write({"key": key})

    def new_key(self, length: int = 8) -> str:
        while True:
            key = "".join(random.choice(self._id_chars) for _ in range(length))
            if key not in self.ctx.persistence:
                return key


class ViewerExtHandler(ApiHandler[ViewerContext]):
    def do_get_contributions(self):
        self._write_response(get_contributions(self.ctx.ext_ctx))

    def do_get_layout(self, contrib_point: str, contrib_index: str):
        self._write_response(
            get_layout(
                self.ctx.ext_ctx, contrib_point, int(contrib_index), self.request.json
            )
        )

    def do_get_callback_results(self):
        self._write_response(get_callback_results(self.ctx.ext_ctx, self.request.json))

    def _write_response(self, response: ExtResponse):
        self.response.set_header("Content-Type", "text/json")
        if response.ok:
            self.response.write(
                json.dumps({"result": response.data}, cls=NumpyJSONEncoder)
            )
        else:
            self.response.set_status(response.status, response.reason)
            self.response.write(
                {"error": {"status": response.status, "message": response.reason}}
            )


@api.route("/viewer/ext/contributions")
class ViewerExtContributionsHandler(ViewerExtHandler):

    # GET /dashi/contributions
    @api.operation(
        operationId="getViewerContributions",
        summary="Get viewer extensions and all their contributions.",
    )
    def get(self):
        self.do_get_contributions()


# noinspection PyPep8Naming
@api.route("/viewer/ext/layout/{contribPoint}/{contribIndex}")
class ViewerExtLayoutHandler(ViewerExtHandler):
    def get(self, contribPoint: str, contribIndex: str):
        """This endpoint is for testing only."""
        self.do_get_layout(contribPoint, contribIndex)

    # POST /dashi/layout/{contrib_point_name}/{contrib_index}
    @api.operation(
        operationId="getViewerContributionLayout",
        summary="Get the initial layout for the given viewer contribution.",
        parameters=[
            {
                "name": "contribPoint",
                "in": "path",
                "description": 'Contribution point name, e.g., "panels"',
                "schema": {"type": "string"},
            },
            {
                "name": "contribIndex",
                "in": "path",
                "description": "Contribution index",
                "schema": {"type": "integer"},
            },
        ],
    )
    def post(self, contribPoint: str, contribIndex: str):
        self.do_get_layout(contribPoint, contribIndex)


@api.route("/viewer/ext/callback")
class ViewerExtCallbackHandler(ViewerExtHandler):

    @api.operation(
        operationId="getViewerContributionCallbackResults",
        summary=(
            "Process the viewer contribution callback requests"
            " and return state change requests."
        ),
    )
    def post(self):
        self.do_get_callback_results()
