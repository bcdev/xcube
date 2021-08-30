# The xcube generator

## Introduction

The *generator* is an xcube feature which allows users to create, manipulate,
and write xcube datasets according to a supplied configuration. The same
configuration can be used to generate a dataset on the user's local computer
or remotely, using an online server.

The generator offers two main user interfaces: A Python API, configured using
Python objects; and a command-line interface, configured using YAML or JSON
files. The Python and file-based configurations have the same structure and
are interconvertible.

The online generator service interfaces with the xcube client via a
well-defined REST API; it is also possible for third-party clients to make use
of this API directly, though it is expected that the Python and command-line
interfaces will be more convenient in most cases.

## Further documentation

This document aims to provide a brief overview of the generation process
and the available configuration options. More details are available in
other documents and in the code itself:

 - Probably the most thorough documentation is available in the [Jupyter demo
   notebooks](https://github.com/dcs4cop/xcube/tree/master/examples/notebooks/generators)
   in the xcube repository. These can be run in any [JupyterLab
   environment](https://jupyterlab.readthedocs.io/en/latest/) containing an
   xcube installation. They combine explanation with interactive worked
   examples to demonstrate practical usage of the generator in typical use
   cases.

 - For the Python API in particular, the [xcube API
   documentation](https://xcube.readthedocs.io/en/latest/api.html#) is
   generated from the docstrings included in the code itself and serves as a
   detailed low-level reference for individual Python classes and methods. The
   docstrings can also be read from a Python environment (e.g. using the `?`
   postfix in IPython or JupyterLab) or, of course, by browsing the source
   code itself.

 - For the YAML/JSON configuration syntax used with the command-line
   interface, there are several examples available in the
   [examples/gen2/configs
   subdirectory](https://github.com/dcs4cop/xcube/tree/master/examples/gen2/configs) of the xcube repository.

## The generation process

The usual cube generation process is as follows:

1. The generator opens the input data store using the store identifier
   and parameters in the *input configuration*.

2. The generator reads from the input store the data specified in the *cube
   configuration* and uses them to create a data cube, often with additional
   manipulation steps such as resampling the data.

3. If an optional *code configuration* has been given, the user-supplied code
   is run on the created data cube, potentially modifying it.

4. The generator writes the generated cube to the data store specified in the
   *output configuration*.

## Invoking the generator from a Python environment

The configurations for the various parts of the generator are used to
initialize a `GeneratorRequest`, which is then passed to
`xcube.core.gen2.generator.CubeGenerator.generate_cube`. The `generate_cube`
method returns a *cube reference* which can be used to open the
cube from the output data store.

The generator can also be directly invoked with a configuration file from a
Python environment, using the
`xcube.core.gen2.generator.CubeGenerator.from_file` method.

## Invoking the generator from the command line

The generator can be invoked from the command line using the `xcube gen2`
subcommand. (Note: the subcommand `xcube gen` invokes an earlier, deprecated
generator feature which is not compatible with the generator framework
described here.)

## Configuration syntax

All Python configuration classes are defined in the `xcube.core.gen2`
package, except for `CodeConfig`, which is in `xcube.core.byoa`.

The types in the parameter tables are given in an ad-hoc, semi-formal notation
whose corresponding Python and JSON representations should be obvious. For the
formal Python type definitions, see the signatures of the `__init__` methods
of the configuration classes; for the formal JSON type definitions, see the
JSON schemata (in [JSON Schema format](https://json-schema.org/)) produced by
the `get_schema` methods of the configuration classes.

### Remote generator service configuration

The command-line interface allows a *service configuration* for the remote
generator service to be provided as a YAML or JSON file. This file defines
the endpoint and access credentials for an online generator service. If it is
provided, the specified remote service will be used to generate the cube.
If it is omitted, the cube will be generated locally. The configuration file
defines three values: `endpoint_url`, `client_id`, and `client_secret`.
A typical service configuration YAML file might look as follows:

```
endpoint_url: "https://xcube-gen.brockmann-consult.de/api/v2/"
client_id: "93da366d7c39517865e4f141ddf1dd81"
client_secret: "d2l0aG91dCByZXN0cmljdGlvbiwgaW5jbHVkaW5nIHd"
```

### Store configuration

When using the command-line interface, a *store configuration* file may
optionally be supplied

In the command-line interface, an additional YAML or JSON file containing one
or more *store configurations* may be supplied. A store configuration consists
of a data store ID and a set of store parameters, which can then be referenced
by an associated *store configuration identifier*. This identifier can be used
in the input configuration, as described below. A typical YAML store
configuration might look as follows:

```
sentinelhub_eu:
  title: SENTINEL Hub (Central Europe)
  description: Datasets from the SENTINEL Hub API deployment in Central Europe
  store_id: sentinelhub
  store_params:
    api_url: https://services.sentinel-hub.com

cds:
  title: C3S Climate Data Store (CDS)
  description: Selected datasets from the Copernicus CDS API
  store_id: cds
  store_params: TODO

my_data_bucket:
  title: TODO
```

### Input configuration

The input configuration defines the data store from which data for the
cube are to be read, and any additional parameters which this data
store requires.

The Python configuration object is `InputConfig`; the corresponding
YAML configuration section is `input_configs`.

| Parameter      | Required? | Type         | Description                    |
|----------------|-----------|--------------|--------------------------------|
| `store_id`     | N         | str          | Identifier for the data store  |
| `opener_id`    | N         | str          | Identifier for the data opener |
| `data_id`      | Y         | str          | Identifier for the dataset     |
| `store_params` | N         | map(str→`*`) | Parameters for the data store  |
| `open_params`  | N         | map(str→`*`) | Parameters for the data opener |
    
`store_id` is a string identifier for a particular xcube data store, defined
by the data store itself. If a store configuration file has been supplied (see
above), a store configuration identifier can also be supplied here in place of
a ‘plain’ store identifier. Store configuration identifiers must be prefixed
by an `@` symbol. If a store configuration identifier is supplied in place of
a store identifier, `store_params` values will be supplied from the predefined
store configuration and can be omitted from the input configuration.

You can list the currently available store
identifiers in a Python environment with the expression `list(map(lambda e:
e.name, xcube.core.store.find_data_store_extensions()))`. (TODO and
via the generator?). 

`data_id` is a string identifier for the dataset within a particular
store. (TODO: describe how to list them.)

The format and content of the `store_params` and `open_params` dictionaries
is defined by the individual store or opener.

### Cube configuration

This configuration element defines the characteristics of the cube that should
be generated. The Python configuration class is called `CubeConfig`, and the
YAML section `cube_config`. All parameters are optional and will be filled
in with defaults if omitted; the default values are dependent on the data
store and dataset.

| Parameter        | Type                         | Units/Description                                                                  |
|------------------|------------------------------|------------------------------------------------------------------------------------|
| `variable_names` | [str, …]                     | Available variables are data store dependent.                                       |
| `crs`            | str                          | PROJ string, JSON string with PROJ parameters, CRS WKT string, or authority string |
| `bbox`           | [float, float, float, float] | Bounding-box (`min_x`, ` min_y`, ` max_x`, ` max_y`) CRS-dependent, usually degrees                                                     |
| `spatial_res`    | float or [float, float]      | CRS-dependent, usually degrees                                                     |
| `tile_size`      | int or [int, int]            | pixels                                                                             |
| `time_range`     | str or [str, str]            | ISO 8601 subset                                                                    |
| `time_period`    | str                          | integer + unit                                                                     |
| `chunks`         | map(str→null/int)          | maps variable names to chunk sizes                                                 |

The `crs` parameter string is interpreted using [`CRS.from_string` in the
pyproj
package](https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_string) and
therefore accepts the same specifiers.

`time_range` specified the start and end of the requested time range. can be
specified either as a date in the format `YYYY-MM-DD` or as a date and time in
the format `YYYY-MM-DD HH:MM:SS`. If the time is omitted, it is taken to be
`00:00:00` (the start of the day) for the start specifier and `24:00:00` (the
end of the day) for the specifier. The end specifier may be omitted; in this
case the current time is used.

`time_period` specified the duration of a single time step in the requested
cube, which determines the temporal resolution. It consists of an integer
denoting the number of time units, followed by single upper-case letter
denoting the time unit. Valid time unit specifiers are D (day), W (week), M
(month), and Y (year). Examples of `time_period` values: `1Y` (one year), `2M`
(two months), `10D` (ten days).

The value of the `chunks` mapping determines how the generated data is chunked
for storage. The chunking has no effect on the data itself, but can have a
dramatic impact on data access speeds in different scenarios. The value of
`chunks` is structured a map from variable names (corresponding to those
specified by the `variable_names` parameter) to chunk sizes.

### Code configuration

Since the code configuration can work directly with instantiated Python
objects (which can't be stored in a YAML file), there are some differences
in code configuration between the Python API and the YAML format. All
parameters are optional (as is the entire code configuration itself).

| Parameter          | Type            | Units/description                                                                       |
|--------------------|-----------------|-----------------------------------------------------------------------------------------|
| `_callable` | Callable | Function to be called to process the datacube. Only available via Python API |
| `callable_ref`     | str (non-empty) | A reference to a Python class or function, in the format `<module>:<function_or_class>` |
| `callable_params`  | map(str→`*`)    | Parameters to be passed to the specified callable                                       |
| `inline_code`      | str (non-empty) | An inline snippet of Python code                                                        |
| `file_set`         | FileSet         | A bundle of Python modules or packages (see details below)                              |
| `install_required` | boolean         | If set, indicates that `file_set` contains modules or packages to be installed.          |

TODO: further details here.

#### The `file_set` parameter

TODO

### Output configuration

This configuration element determines where the generated cube should be
written to. The Python configuration class is called `OutputConfig`, and the
YAML section `output_config`.

| Parameter      | Type         | Units/description                                           |
|----------------|--------------|-------------------------------------------------------------|
| `store_id`     | str          | Identifier of output store                                  |
| `writer_id`    | str          | Identifier of data writer                                   |
| `data_id`      | str          | Identifier under which to write the cube                    |
| `store_params` | map(str→`*`) | Store-dependent parameters for output store                 |
| `write_params` | map(str→`*`) | Writer-dependent parameters for output writer               |
| `replace`      | bool         | If true, replace any existing data with the same identifier. |

