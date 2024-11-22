# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import (
    Optional,
    Callable,
    Any,
    Union,
)
from collections.abc import Sequence, Awaitable

import pytest
import tornado.httputil
import tornado.web
from tornado import concurrent

from test.server.mocks import mock_server
from xcube.server.api import Api
from xcube.server.api import ApiContextT
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from xcube.server.api import ApiRoute
from xcube.server.api import Context
from xcube.server.api import ServerConfig
from xcube.server.asyncexec import ReturnT
from xcube.server.webservers.tornado import SERVER_CTX_ATTR_NAME
from xcube.server.webservers.tornado import TornadoApiRequest
from xcube.server.webservers.tornado import TornadoFramework
from xcube.server.webservers.tornado import TornadoRequestHandler
from xcube.util.jsonschema import JsonObjectSchema


class TornadoFrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.application = MockApplication()
        self.io_loop = MockIOLoop()
        # noinspection PyTypeChecker
        self.framework = TornadoFramework(
            application=self.application, io_loop=self.io_loop
        )

    def test_config_schema(self):
        self.assertIsInstance(self.framework.config_schema, JsonObjectSchema)

    def test_add_routes(self):
        class Handler(ApiHandler):
            pass

        self.framework.add_routes(
            [
                route0 := ApiRoute("foo", "foo/bar", Handler),
                route1 := ApiRoute("foo", "foo/baz", Handler, slash=True),
            ],
            "/",
        )
        self.assertEqual(
            [
                ("/foo/bar", TornadoRequestHandler, {"api_route": route0}),
                ("/foo/baz/?", TornadoRequestHandler, {"api_route": route1}),
            ],
            self.application.handlers,
        )

    def test_start_and_update_and_stop(self):
        server = mock_server(framework=self.framework, config={})
        self.assertEqual(0, self.application.listen_count)
        self.assertEqual(0, self.io_loop.start_count)
        self.assertEqual(0, self.io_loop.stop_count)
        self.framework.start(server.ctx)
        self.assertEqual(1, self.application.listen_count)
        self.assertEqual(1, self.io_loop.start_count)
        self.assertEqual(0, self.io_loop.stop_count)
        self.framework.update(server.ctx)
        self.assertEqual(1, self.application.listen_count)
        self.assertEqual(1, self.io_loop.start_count)
        self.assertEqual(0, self.io_loop.stop_count)
        self.framework.stop(server.ctx)
        self.assertEqual(1, self.application.listen_count)
        self.assertEqual(1, self.io_loop.start_count)
        self.assertEqual(1, self.io_loop.stop_count)

    def test_data_logging_disabled_by_default(self):
        server = mock_server(framework=self.framework, config={})
        self.framework.start(server.ctx)
        self.assertNotIn("log_function", self.application.settings)
        self.framework.stop(server.ctx)

    def test_enable_data_logging(self):
        server = mock_server(framework=self.framework, config={"data_logging": True})
        self.framework.start(server.ctx)
        self.assertIn("log_function", self.application.settings)
        log_function = self.application.settings["log_function"]
        self.assertTrue(callable(log_function))

        class DatasetsHandler(tornado.web.RequestHandler):
            def get(self):
                return {}

        # noinspection PyTypeChecker
        handler = DatasetsHandler(
            self.application,
            tornado.httputil.HTTPServerRequest(
                method="GET", uri="/datasets", connection=MockConnection()
            ),
        )

        # For time being, smoke test only, logging is side effect.
        # Check if we can return data record from log_function(),
        # and verify fields are as expected.
        log_function(handler)

        self.framework.stop(server.ctx)

    def test_async_exec(self):
        def my_func(a, b):
            return a + b

        self.assertEqual(0, self.io_loop.call_later_count)
        self.framework.call_later(0.1, my_func, 40, 2)
        self.assertEqual(1, self.io_loop.call_later_count)

        self.assertEqual(0, self.io_loop.run_in_executor_count)
        self.framework.run_in_executor(None, my_func, 40, 2)
        self.assertEqual(1, self.io_loop.run_in_executor_count)

    def test_path_to_pattern_ok(self):
        p2p = TornadoFramework.path_to_pattern

        self.assertEqual("/collections", p2p("/collections"))

        self.assertEqual(
            "/collections/"
            "("
            "?P<collection_id>"
            "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
            ")",
            p2p("/collections/{collection_id}"),
        )

        self.assertEqual(
            "/collections/"
            "("
            "?P<collection_id>"
            "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
            ")"
            "/items",
            p2p("/collections/{collection_id}/items"),
        )

        self.assertEqual(
            "/collections/"
            "("
            "?P<collection_id>"
            "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
            ")"
            "/items/"
            "("
            "?P<item_id>"
            "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
            ")",
            p2p("/collections/{collection_id}/items/{item_id}"),
        )

        self.assertEqual(
            "/s3bucket/"
            "("
            "?P<dataset_id>"
            "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
            ")"
            "\\/?"
            "(?P<path>.*)",
            p2p("/s3bucket/{dataset_id}/{*path}"),
        )

    def test_path_to_pattern_fails(self):
        p2p = TornadoFramework.path_to_pattern

        with self.assertRaises(ValueError) as cm:
            p2p("/datasets/{dataset id}")
        self.assertEqual(
            '"{name}" in path must be a valid identifier, but got "dataset id"',
            f"{cm.exception}",
        )

        with self.assertRaises(ValueError) as cm:
            p2p("/datasets/{dataset_id")
        self.assertEqual(
            'missing closing "}" in "/datasets/{dataset_id"',
            f"{cm.exception}",
        )

        with self.assertRaises(ValueError) as cm:
            p2p("/datasets/dataset_id}/bbox")
        self.assertEqual(
            'missing opening "{" in "/datasets/dataset_id}/bbox"',
            f"{cm.exception}",
        )

        with self.assertRaises(ValueError) as cm:
            p2p("/datasets/{*dataset_id}/places")
        self.assertEqual(
            "wildcard variable must be last in path,"
            ' but path was "/datasets/{*dataset_id}/places"',
            f"{cm.exception}",
        )

        with self.assertRaises(ValueError) as cm:
            p2p("/datasets/{*dataset_id}/{*places}")
        self.assertEqual(
            "only a single wildcard variable is allowed,"
            " but found 2 in path"
            ' "/datasets/{*dataset_id}/{*places}"',
            f"{cm.exception}",
        )


class TornadoApiRequestTest(unittest.TestCase):
    def test_basic_props(self):
        body = b'{"type": "feature", "id": 137}'
        tr = tornado.httputil.HTTPServerRequest(
            method="GET", host="localhost:8080", uri="/datasets", body=body
        )
        request = TornadoApiRequest(tr)
        self.assertEqual("http://localhost:8080/datasets", request.url)
        self.assertEqual(body, request.body)
        self.assertEqual({"type": "feature", "id": 137}, request.json)
        self.assertEqual([], request.get_query_args("details"))

    def test_get_query_args(self):
        tr = tornado.httputil.HTTPServerRequest(
            method="GET",
            host="localhost:8080",
            uri="/datasets?details=1",
        )
        request = TornadoApiRequest(tr)
        self.assertEqual(["1"], request.get_query_args("details"))
        self.assertEqual(["1"], request.get_query_args("details", type=str))
        self.assertEqual([1], request.get_query_args("details", type=int))
        self.assertEqual([True], request.get_query_args("details", type=bool))

    def test_get_query_args_case_insensitive(self):
        tr = tornado.httputil.HTTPServerRequest(
            method="GET",
            host="localhost:8080",
            uri="/datasets?details=1",
        )
        request = TornadoApiRequest(tr)
        request.make_query_lower_case()
        self.assertEqual(["1"], request.get_query_args("details"))
        self.assertEqual(["1"], request.get_query_args("DETAILS"))
        self.assertEqual(["1"], request.get_query_args("Details"))

    def test_get_query_args_invalid_type(self):
        tr = tornado.httputil.HTTPServerRequest(
            method="GET",
            host="localhost:8080",
            uri="/datasets?details=x",
        )
        request = TornadoApiRequest(tr)
        with self.assertRaises(ApiError.BadRequest) as cm:
            request.get_query_args("details", type=int)
        self.assertEqual(
            "HTTP status 400: Query parameter 'details' must have type 'int'.",
            f"{cm.exception}",
        )

    def test_invalid_json(self):
        tr = tornado.httputil.HTTPServerRequest(
            method="GET",
            host="localhost:8080",
            uri="/datasets?details=x",
        )
        request = TornadoApiRequest(tr)
        with self.assertRaises(ApiError.BadRequest) as cm:
            # noinspection PyUnusedLocal
            result = request.json
        self.assertEqual(
            "HTTP status 400:"
            " Body does not contain valid JSON:"
            " Expecting value: line 1 column 1 (char 0)",
            f"{cm.exception}",
        )


class TornadoApiRequestUrlTest(unittest.TestCase):
    tr = tornado.httputil.HTTPServerRequest(
        method="GET",
        host="localhost:8080",
        uri="/datasets?details=1",
    )

    def test_prefixes_not_given(self):
        request = TornadoApiRequest(self.tr)
        self.assertEqual(
            "http://localhost:8080/collections", request.url_for_path("collections")
        )
        self.assertEqual(
            "http://localhost:8080/collections?details=1",
            request.url_for_path("/collections", query="details=1"),
        )
        self.assertEqual("http://localhost:8080", request.base_url)
        self.assertEqual("http://localhost:8080", request.reverse_base_url)

    def test_rel_url_prefix_given(self):
        request = TornadoApiRequest(self.tr, url_prefix="/api/v1")
        self.assertEqual(
            "http://localhost:8080/api/v1/collections?details=1",
            request.url_for_path("/collections", query="details=1"),
        )
        self.assertEqual("http://localhost:8080/api/v1", request.base_url)
        self.assertEqual("http://localhost:8080/api/v1", request.reverse_base_url)

    def test_rel_url_prefix_and_rel_reverse_url_prefix_given(self):
        request = TornadoApiRequest(
            self.tr, url_prefix="/api/v1", reverse_url_prefix="/proxy/9999"
        )
        self.assertEqual(
            "http://localhost:8080/api/v1/collections?details=1",
            request.url_for_path("/collections", query="details=1"),
        )
        self.assertEqual(
            "http://localhost:8080/proxy/9999/collections?details=1",
            request.url_for_path("/collections", query="details=1", reverse=True),
        )
        self.assertEqual("http://localhost:8080/api/v1", request.base_url)
        self.assertEqual("http://localhost:8080/proxy/9999", request.reverse_base_url)

    def test_abs_url_prefix_given(self):
        request = TornadoApiRequest(self.tr, url_prefix="https://test.com")
        self.assertEqual(
            "https://test.com/collections?details=1",
            request.url_for_path("/collections", query="details=1"),
        )
        self.assertEqual(
            "https://test.com/collections?details=1",
            request.url_for_path("/collections", query="details=1", reverse=True),
        )
        self.assertEqual("https://test.com", request.base_url)
        self.assertEqual("https://test.com", request.reverse_base_url)

    def test_abs_url_prefix_and_rel_reverse_url_prefix_given(self):
        request = TornadoApiRequest(
            self.tr,
            url_prefix="https://test.com/api/v1",
            reverse_url_prefix="/proxy/9999",
        )
        self.assertEqual(
            "https://test.com/api/v1/collections?details=1",
            request.url_for_path("/collections", query="details=1"),
        )
        self.assertEqual(
            "http://localhost:8080/proxy/9999/collections?details=1",
            request.url_for_path("/collections", query="details=1", reverse=True),
        )
        self.assertEqual("https://test.com/api/v1", request.base_url)
        self.assertEqual("http://localhost:8080/proxy/9999", request.reverse_base_url)

    def test_abs_reverse_url_prefix_given(self):
        request = TornadoApiRequest(
            self.tr, url_prefix="", reverse_url_prefix="https://test.com/api"
        )
        self.assertEqual(
            "http://localhost:8080/collections?details=1",
            request.url_for_path("/collections", query="details=1"),
        )
        self.assertEqual(
            "https://test.com/api/collections?details=1",
            request.url_for_path("/collections", query="details=1", reverse=True),
        )
        self.assertEqual("http://localhost:8080", request.base_url)
        self.assertEqual("https://test.com/api", request.reverse_base_url)


class TornadoRequestHandlerTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_api_error_converted(self):
        application = tornado.web.Application()

        setattr(application, SERVER_CTX_ATTR_NAME, MockContext())

        # noinspection PyTypeChecker
        request = tornado.httputil.HTTPServerRequest(
            method="GET",
            host="localhost:8080",
            uri="/datasets?details=x",
            connection=MockConnection(),
        )

        class TestHandler(ApiHandler):
            def get(self):
                raise ApiError(550)

            def post(self):
                raise ApiError(551)

            def put(self):
                raise ApiError(552)

            def delete(self):
                raise ApiError(553)

            def options(self):
                raise ApiError(554)

        api_route = ApiRoute("test", "/test", TestHandler)
        handler = TornadoRequestHandler(application, request, api_route=api_route)

        async def test_it():
            with pytest.raises(tornado.web.HTTPError, match=r".*550.*"):
                await handler.get()
            with pytest.raises(tornado.web.HTTPError, match=r".*551.*"):
                await handler.post()
            with pytest.raises(tornado.web.HTTPError, match=r".*552.*"):
                await handler.put()
            with pytest.raises(tornado.web.HTTPError, match=r".*553.*"):
                await handler.delete()
            with pytest.raises(tornado.web.HTTPError, match=r".*554.*"):
                await handler.options()

        import asyncio

        asyncio.run(test_it())


# Helpers


class MockIOLoop:
    def __init__(self):
        self.start_count = 0
        self.stop_count = 0
        self.call_later_count = 0
        self.run_in_executor_count = 0

    def start(self):
        self.start_count += 1

    def stop(self):
        self.stop_count += 1

    # noinspection PyUnusedLocal
    def call_later(self, *args, **kwargs):
        self.call_later_count += 1

    # noinspection PyUnusedLocal
    def run_in_executor(self, *args, **kwargs):
        self.run_in_executor_count += 1


class MockApplication:
    def __init__(self):
        self.handlers = []
        self.settings = {}
        self.ui_modules = {}
        self.ui_methods = {}
        self.listen_count = 0

    # noinspection PyUnusedLocal
    def add_handlers(self, domain: str, handlers: Sequence):
        self.handlers.extend(handlers)

    # noinspection PyUnusedLocal
    def listen(self, *args, **kwargs):
        self.listen_count += 1


class MockConnection:
    def __init__(self):
        self._cb = None

    def set_close_callback(self, cb):
        self._cb = cb


class MockContext(Context):
    def __init__(self):
        self._api = Api("test")
        self._config = {}

    @property
    def apis(self) -> tuple[Api]:
        return (self._api,)

    def get_open_api_doc(self, include_all: bool = False) -> dict[str, Any]:
        return {}

    @property
    def config(self) -> ServerConfig:
        # noinspection PyTypeChecker
        return self._config

    def get_api_ctx(
        self, api_name: str, cls: Optional[type[ApiContextT]] = None
    ) -> Optional[ApiContextT]:
        return self._api if api_name == "test" else None

    def on_update(self, prev_context: Optional["Context"]):
        pass

    def on_dispose(self):
        pass

    def call_later(
        self, delay: Union[int, float], callback: Callable, *args, **kwargs
    ) -> object:
        pass

    def run_in_executor(
        self,
        executor: Optional[concurrent.futures.Executor],
        function: Callable[..., ReturnT],
        *args: Any,
        **kwargs: Any,
    ) -> Awaitable[ReturnT]:
        pass
