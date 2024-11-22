# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os.path
from functools import cache
from typing import Any, Optional
from collections.abc import Mapping

from xcube.constants import DEFAULT_SERVER_ADDRESS
from xcube.constants import DEFAULT_SERVER_PORT
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

BASE_SERVER_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        port=JsonIntegerSchema(title="Server port.", default=DEFAULT_SERVER_PORT),
        address=JsonStringSchema(
            title="Server address.", default=DEFAULT_SERVER_ADDRESS
        ),
        base_dir=JsonStringSchema(
            title="Base directory used to resolve relative local paths."
            " Can be a local filesystem path or an absolute URL.",
        ),
        url_prefix=JsonStringSchema(
            title="Prefix to be prepended to all URL route paths."
            " Can be an absolute URL or a relative URL path.",
        ),
        reverse_url_prefix=JsonStringSchema(
            title="Prefix to be prepended to reverse URL paths"
            " returned by server responses."
            " Can be an absolute URL or a relative URL path.",
        ),
        data_logging=JsonBooleanSchema(
            title="Log JSON data records instead of human-readable message logs."
        ),
        trace_perf=JsonBooleanSchema(
            title="Output performance measures",
        ),
        static_routes=JsonArraySchema(
            title="Static content routes",
            items=JsonObjectSchema(
                properties=dict(
                    path=JsonStringSchema(title="The URL path", min_length=1),
                    dir_path=JsonStringSchema(
                        title="A local directory path", min_length=1
                    ),
                    default_filename=JsonStringSchema(
                        title="Optional default filename",
                        examples=["index.html"],
                        min_length=1,
                    ),
                    openapi_metadata=JsonObjectSchema(
                        title="Optional OpenAPI operation metadata",
                        additional_properties=True,
                    ),
                ),
                required=["path", "dir_path"],
                additional_properties=False,
            ),
        ),
        api_spec=JsonObjectSchema(
            title="API specification",
            description="selected = (includes | ALL) - (excludes | NONE)",
            properties=dict(
                includes=JsonArraySchema(JsonStringSchema(min_length=1)),
                excludes=JsonArraySchema(JsonStringSchema(min_length=1)),
            ),
            additional_properties=False,
        ),
    ),
    # We allow for other configuration settings contributed
    # by APIs. If these APIs are currently not in use,
    # validation would fail if additional_properties=False.
    additional_properties=True,
)


@cache
def normalize_base_dir(base_dir: Optional[str]) -> str:
    """Normalize the given base directory *base_dir*."""
    if base_dir is None:
        base_dir = os.path.abspath("")
    elif not is_absolute_path(base_dir):
        base_dir = os.path.abspath(base_dir)
    while base_dir != "/" and base_dir.endswith("/"):
        base_dir = base_dir[:-1]
    return base_dir


def get_base_dir(config: Mapping[str, Any]) -> str:
    """Get the normalized base directory from configuration *config*."""
    return normalize_base_dir(config.get("base_dir"))


def resolve_config_path(config: Mapping[str, Any], path: str) -> str:
    """Resolve a given relative *path* against the base directory given by
    *config*. Return *path* unchanged, if it is absolute.
    """
    if is_absolute_path(path):
        abs_path = path
    else:
        base_dir = get_base_dir(config)
        abs_path = f"{base_dir}/{path}"
    # Resolve ".." and "." in path
    if "://" in abs_path:
        scheme, host_path = abs_path.split("://", maxsplit=1)
        if "/" in host_path:
            hostname, url_path = host_path.split("/", maxsplit=1)
            url_path = _remove_path_dot_segments(url_path)
            return f"{scheme}://{hostname}/{url_path}"
        else:
            return f"{scheme}://{host_path}"
    else:
        if os.name == "nt":
            # Windows can also live with forward slashes
            abs_path = abs_path.replace("\\", "/")
        return _remove_path_dot_segments(abs_path)


#
# Following code is stolen from urllib3.url
#
def _remove_path_dot_segments(path: str) -> str:
    # See http://tools.ietf.org/html/rfc3986#section-5.2.4 for pseudo-code
    segments = path.split("/")  # Turn the path into a list of segments
    output = []  # Initialize the variable to use to store output

    for segment in segments:
        # '.' is the current directory, so ignore it, it is superfluous
        if segment == ".":
            continue
        # Anything other than '..', should be appended to the output
        elif segment != "..":
            output.append(segment)
        # In this case segment == '..', if we can, we should pop the last
        # element
        elif output:
            output.pop()

    # If the path starts with '/' and the output is empty or the first string
    # is non-empty
    if path.startswith("/") and (not output or output[0]):
        output.insert(0, "")

    # If the path starts with '/.' or '/..' ensure we add one more empty
    # string to add a trailing '/'
    if path.endswith(("/.", "/..")):
        output.append("")

    return "/".join(output)


def is_absolute_path(path: str) -> bool:
    """Test whether *path* is an absolute filesystem path or URL."""
    # This is a rather weak test, may be enhanced if desired
    return "//" in path or ":" in path or path.startswith("/")


def get_url_prefix(config: Mapping[str, Any]) -> str:
    """Get the sanitized URL prefix so, if given, it starts with
    a leading slash and ends without one.

    Args:
        config: Server configuration.

    Returns:
        Sanitized URL prefix, may be an empty string.
    """
    return _sanitize_url_prefix(config.get("url_prefix"))


def get_reverse_url_prefix(config: Mapping[str, Any]) -> str:
    """Get the sanitized reverse URL prefix so, if given, it starts with
    a leading slash and ends without one.

    Args:
        config: Server configuration.

    Returns:
        Sanitized URL prefix, may be an empty string.
    """
    return _sanitize_url_prefix(
        config.get("reverse_url_prefix", config.get("url_prefix"))
    )


def _sanitize_url_prefix(url_prefix: Optional[str]) -> str:
    """Get a sanitized URL prefix so, if given, it starts with
    a leading slash and ends without one.

    Args:
        url_prefix: URL prefix path.

    Returns:
        Sanitized URL prefix path, may be an empty string.
    """
    if not url_prefix:
        return ""

    while url_prefix.startswith("//"):
        url_prefix = url_prefix[1:]
    while url_prefix.endswith("/"):
        url_prefix = url_prefix[:-1]

    if url_prefix == "":
        return ""

    if (
        url_prefix.startswith("/")
        or url_prefix.startswith("http://")
        or url_prefix.startswith("https://")
    ):
        return url_prefix

    return "/" + url_prefix
