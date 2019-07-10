[![Build Status](https://travis-ci.com/dcs4cop/xcube.svg?branch=master)](https://travis-ci.com/dcs4cop/xcube)
[![codecov](https://codecov.io/gh/dcs4cop/xcube/branch/master/graph/badge.svg)](https://codecov.io/gh/dcs4cop/xcube)


# xcube

Data cubes with [xarray](http://xarray.pydata.org/).

# Table of Contents

- [Installation](#installation)
- [Developer Guide](#developer-guide)
- [User Guide](#user-guide)
- [Docker](#docker)
- [Tools](#tools)
  - [`xcube` Command Line Interface](#xcube-command-line-interface)
  - [`xcube chunk`](#xcube-chunk)
  - [`xcube dump`](#xcube-dump)
  - [`xcube extract`](#xcube-extract)
  - [`xcube gen`](#xcube-gen)
  - [`xcube grid`](#xcube-grid)
  - [`xcube level`](#xcube-level)
  - [`xcube prune`](#xcube-prune)
  - [`xcube resample`](#xcube-resample)
  - [`xcube serve`](#xcube-serve)
  - [`xcube vars2dim`](#xcube-vars2dim)
  - [`xcube verify`](#xcube-verify)


# Installation

First
    
    $ git clone https://github.com/dcs4cop/xcube.git
    $ cd xcube
    $ conda env create
    
Then
    
    $ activate xcube
    $ python setup.py develop

Update
    
    $ activate xcube
    $ git pull --force
    $ python setup.py develop
    
    
Run tests

    $ pytest
    
with coverage

    $ pytest --cov=xcube

with [coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html) in HTML

    $ pytest --cov-report html --cov=xcube

# Developer Guide

...is [here](docs/DEV-GUIDE.md).

# User Guide

A user guide is currently under development, a quickstart on how to generate data cubes can be found [here](docs/quickstart.md).

# Docker

To start a demo using docker use the following commands

    $ docker build -t [your name] .
    $ docker run -d -p [host port]:8000 [your name]
    
**Example:**

    $  docker build -t xcube:0.1.0dev6 .
    $  docker run -d -p 8001:8000 xcube:0.1.0dev6
    $  docker ps

**Docker TODOs:** 

* automatically build images on quay.io 
  and then use a xcube-services ```docker-compose.yml``` configuration.

# Tools

## `xcube` Command Line Interface

    $ xcube --help
    Usage: xcube [OPTIONS] COMMAND [ARGS]...
    
    Xcube Toolkit
    
    Options:
      --version                Show the version and exit.
      --traceback              Enable tracing back errors by dumping the Python
                               call stack. Pass as very first option to also trace
                               back error during command-line validation.

      --scheduler <scheduler>  Enable distributed computing using the Dask
                               scheduler identified by <scheduler>. <scheduler>
                               can have the form <address>?<keyword>=<value>,...
                               where <address> is <host> or <host>:<port> and
                               specifies the scheduler's address in your network.
                               For more information on distributed computing using
                               Dask, refer to http://distributed.dask.org/. Pairs
                               of <keyword>=<value> are passed to the Dask client.
                               Refer to http://distributed.dask.org/en/latest/api.
                               html#distributed.Client
      --help                   Show this message and exit.

    Commands:
      chunk     (Re-)chunk dataset.
      dump      Dump contents of a dataset.
      extract   Extract cube time series.
      gen       Generate data cube.
      grid      Find spatial data cube resolutions and adjust bounding boxes.
      level     Generate multi-resolution levels.
      prune     Delete empty chunks.
      resample  Resample data along the time dimension.
      serve     Serve data cubes via web service.
      vars2dim  Convert cube variables into new dimension.
      verify    Perform cube verification.

    

## `xcube chunk`

(Re-)chunk dataset.

    $ xcube chunk --help
    Usage: xcube chunk [OPTIONS] <input> <output>
    
      (Re-)chunk dataset. Changes the external chunking of all variables of
      <input> according to <chunks> and writes the result to <output>.
    
    Options:
      -f, --format <format>  Format of the output. If not given, guessed from
                             <output>.
      -p, --params <params>  Parameters specific for the output format. Comma-
                             separated list of <key>=<value> pairs.
      -c, --chunks <chunks>  Chunk sizes for each dimension. Comma-separated list
                             of <dim>=<size> pairs, e.g. "time=1,lat=270,lon=270"
      --help                 Show this message and exit.


Example:

    $ xcube chunk input_not_chunked.zarr output_rechunked.zarr --chunks "time=1,lat=270,lon=270"

## `xcube dump`

Dump contents of a dataset.

    $ xcube dump --help
    Usage: xcube dump [OPTIONS] <path>
    
    Dump contents of a dataset.
    
    optional arguments:
      --help                Show this help message and exit
      --variable, -v        Name of a variable (multiple allowed).
      --encoding, -e        Dump also variable encoding information.


Example:

    $ xcube dump xcube_cube.zarr 

## `xcube extract`

Extract cube points.

    $ xcube dump --help
    Usage: xcube extract [OPTIONS] CUBE POINTS

      Extract cube points.

      Extracts data cells from CUBE at coordinates given in each POINTS record
      and writes the resulting values to given output path and format.

      <points> must be a CSV file that provides at least the columns "lon",
      "lat", and "time". The "lon" and "lat" columns provide a point's location
      in decimal degrees. The "time" column provides a point's date or date-
      time. Its format should preferably be ISO, but other formats may work as
      well.

    Options:
      -o, --output TEXT             Output file. If omitted, output is written to
                                    stdout.
      -f, --format [csv|json|xlsx]  Output format. Currently, only 'csv' is
                                    supported.
      -C, --coords                  Include cube cell coordinates in output.
      -B, --bounds                  Include cube cell coordinate boundaries (if
                                    any) in output.
      -I, --indexes                 Include cube cell indexes in output.
      -R, --refs                    Include point values as reference in output.
      --help                        Show this message and exit.


Example:  
    
    $ xcube extract xcube_cube.zarr point_data.csv -CBIR
    
    
## `xcube gen`

Generate data cube.

    $ xcube gen --help
    Usage: xcube gen [OPTIONS] INPUT_FILES

      Generate data cube. Data cubes may be created in one go or successively in
      append mode, input by input. The input may be one or more input files or a
      pattern that may contain wildcards '?', '*', and '**'. The input files can
        be passed as lines of a text file.

    Options:
      -p, --proc INPUT_PROCESSOR      Input processor type name. The choices as
                                      input processor and additional information
                                      about input processors  can be accessed by
                                      calling xcube gen --info . Defaults to "default" - the default input processor 
                                      that can deal with most common datasets conforming with the CF conventions.
      -c, --config CONFIG_FILE        Data cube configuration file in YAML format.
                                      More than one config input file is
                                      allowed.When passing several config files,
                                      they are merged considering the order passed
                                      via command line.
      -d, --dir OUTPUT_DIR            Output directory. Defaults to '.'
      -n, --name OUTPUT_NAME          Output filename pattern. Defaults to
                                      'PROJ_WGS84_{INPUT_FILE}'.
      -f, --format OUTPUT_FORMAT      Output writer type name. Defaults to 'zarr'.
                                      The choices for the output format are:
                                      ['csv', 'mem', 'netcdf4', 'zarr'].
                                      Additional information about output formats
                                      can be accessed by calling xcube
                                      generate_cube --info
      -s, --size OUTPUT_SIZE          Output size in pixels using format
                                      "<width>,<height>".
      -r, --region OUTPUT_REGION      Output region using format "<lon-min>,<lat-
                                      min>,<lon-max>,<lat-max>"
      -v, --variables OUTPUT_VARIABLES
                                      Variables to be included in output. Comma-
                                      separated list of names which may contain
                                      wildcard characters "*" and "?".
      --resampling OUTPUT_RESAMPLING  Fallback spatial resampling algorithm to be
                                      used for all variables. Defaults to
                                      'Nearest'. The choices for the resampling
                                      algorithm are: dict_keys(['Nearest',
                                      'Bilinear', 'Cubic', 'CubicSpline',
                                      'Lanczos', 'Average', 'Min', 'Max',
                                      'Median', 'Mode', 'Q1', 'Q3'])
      -a, --append                    Append successive outputs.
      --dry_run                       Just read and process inputs, but don't
                                      produce any outputs.
      -i, --info                      Displays additional information about format
                                      options or about input processors.
      --sort                          The input file list will be sorted before
                                      creating the data cube. If --sort parameter
                                      is not passed, order of input list will be
                                      kept.
      --help                          Show this message and exit.

Below is the `xcube gen --info` call with 5 input processors installed via plugins.

    $ xcube gen --info
    input processors to be used with option --proc:
      default                           Single-scene NetCDF/CF inputs in xcube standard format
      rbins-seviri-highroc-scene-l2     RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs 
      rbins-seviri-highroc-daily-l2     RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs
      snap-olci-highroc-l2              SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs
      snap-olci-cyanoalert-l2           SNAP Sentinel-3 OLCI CyanoAlert Level-2 NetCDF inputs
      vito-s2plus-l2                    VITO Sentinel-2 Plus Level 2 NetCDF inputs
      
    For more input processors use existing "xcube-gen-..." plugins from the github organisation DCS4COP or write own plugin.
    
    
    output formats to be used with option --format:
      csv                     (*.csv)       CSV file format
      mem                     (*.mem)       In-memory dataset I/O
      netcdf4                 (*.nc)        NetCDF-4 file format
      zarr                    (*.zarr)      Zarr file format (http://zarr.readthedocs.io)
    

Example:

    $ xcube gen -a -s 2000,1000 -r 0,50,5,52.5 -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -n hiroc-cube -t -p default D:\OneDrive\BC\EOData\HIGHROC\2017\01\*.nc

Available xcube input processors within xcube's organisation:
* [xcube-gen-rbins](https://github.com/dcs4cop/xcube-gen-rbins)
* [xcube-gen-bc](https://github.com/dcs4cop/xcube-gen-bc)
* [xcube-gen-vito](https://github.com/dcs4cop/xcube-gen-vito)


## `xcube grid`

Find spatial data cube resolutions and adjust bounding boxes.

    $ xcube grid --help
    Usage: xcube grid [OPTIONS] COMMAND [ARGS]...
    
      Find spatial data cube resolutions and adjust bounding boxes.
    
      We find suitable resolutions with respect to a possibly regional fixed
      Earth grid and adjust regional spatial bounding boxes to that grid. We
      also try to select the resolutions such that they are taken from a certain
      level of a multi-resolution pyramid whose level resolutions increase by a
      factor of two.
    
      The graticule at a given resolution level L within the grid is given by
    
          RES(L) = COVERAGE * HEIGHT(L)
          HEIGHT(L) = HEIGHT_0 * 2 ^ L
          LON(L, I) = LON_MIN + I * HEIGHT_0 * RES(L)
          LAT(L, J) = LAT_MIN + J * HEIGHT_0 * RES(L)
    
      With
    
          RES:      Grid resolution in degrees.
          HEIGHT:   Number of vertical grid cells for given level
          HEIGHT_0: Number of vertical grid cells at lowest resolution level.
    
      Let WIDTH and HEIGHT be the number of horizontal and vertical grid cells
      of a global grid at a certain LEVEL with WIDTH * RES = 360 and HEIGHT *
      RES = 180, then we also force HEIGHT = TILE * 2 ^ LEVEL.
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      abox    Adjust a bounding box to a fixed Earth grid.
      levels  List levels for a resolution or a tile size.
      res     List resolutions close to a target resolution.

    
Example: Find suitable target resolution for a ~300m (Sentinel 3 OLCI FR resolution) 
fixed Earth grid within a deviation of 5%.

    $ xcube grid res 300m -d 5%
    
    TILE    LEVEL   HEIGHT  INV_RES RES (deg)       RES (m), DELTA_RES (%)
    540     7       69120   384     0.0026041666666666665   289.9   -3.4
    4140    4       66240   368     0.002717391304347826    302.5   0.8
    8100    3       64800   360     0.002777777777777778    309.2   3.1
    ...
    
289.9m is close enough and provides 7 resolution levels, which is good. Its inverse resolution is 384,
which is the fixed Earth grid identifier.

We want to see if the resolution pyramid also supports a resolution close to 10m 
(Sentinel 2 MSI resolution).

    $ xcube grid levels 384 -m 6
    LEVEL   HEIGHT  INV_RES RES (deg)       RES (m)
    0       540     3       0.3333333333333333      37106.5
    1       1080    6       0.16666666666666666     18553.2
    2       2160    12      0.08333333333333333     9276.6
    ...
    11      1105920 6144    0.00016276041666666666  18.1
    12      2211840 12288   8.138020833333333e-05   9.1
    13      4423680 24576   4.0690104166666664e-05  4.5

This indicates we have a resolution of 9.1m at level 12.

Lets assume we have data cube region with longitude from 0 to 5 degrees
and latitudes from 50 to 52.5 degrees. What is the adjusted bounding box 
on a fixed Earth grid with the inverse resolution 384?

    $ xcube grid abox  0,50,5,52.5  384
     
    Orig. box coord. = 0.0,50.0,5.0,52.5
    Adj. box coord.  = 0.0,49.21875,5.625,53.4375
    Orig. box WKT    = POLYGON ((0.0 50.0, 5.0 50.0, 5.0 52.5, 0.0 52.5, 0.0 50.0))
    Adj. box WKT     = POLYGON ((0.0 49.21875, 5.625 49.21875, 5.625 53.4375, 0.0 53.4375, 0.0 49.21875))
    Grid size  = 2160 x 1620 cells
    with
      TILE      = 540
      LEVEL     = 7
      INV_RES   = 384
      RES (deg) = 0.0026041666666666665
      RES (m)   = 289.89450727414993

    
Note, to check bounding box WKTs, you can use the 
handy tool [Wicket](https://arthur-e.github.io/Wicket/sandbox-gmaps3.html).
     
## `xcube gen`

    $ xcube gen --help
    Usage: xcube gen [OPTIONS] [INPUTS]...
    
      Generate data cube. Data cubes may be created in one go or successively in
      append mode, input by input. The input paths may be one or more input
      files or a pattern that may contain wildcards '?', '*', and '**'. The
      input paths can also be passed as lines of a text file. To do so, provide
      exactly one input file with ".txt" extension which contains the actual
      input paths to be used.
    
    Options:
      -p, --proc []                   Input processor type name. The choices as
                                      input processor and additional information
                                      about input processors  can be accessed by
                                      calling xcube gen --info . Defaults to
                                      "default" - the default input processor that
                                      can deal with most common datasets
                                      conforming with the CF conventions.
      -c, --config TEXT               Data cube configuration file in YAML format.
                                      More than one config input file is
                                      allowed.When passing several config files,
                                      they are merged considering the order passed
                                      via command line.
      -o, --output TEXT               Output path. Defaults to 'out.zarr'
      -f, --format [csv|mem|netcdf4|zarr]
                                      Output format. The choices for the output
                                      format are: ['csv', 'mem', 'netcdf4',
                                      'zarr']. Additional information about output
                                      formats can be accessed by calling xcube gen
                                      --info. If omitted, the format will be
                                      guessed from the given output path.
      -s, --size TEXT                 Output size in pixels using format
                                      "<width>,<height>".
      -r, --region TEXT               Output region using format "<lon-min>,<lat-
                                      min>,<lon-max>,<lat-max>"
      -v, --variables, --vars TEXT    Variables to be included in output. Comma-
                                      separated list of names which may contain
                                      wildcard characters "*" and "?".
      --resampling [Nearest|Bilinear|Cubic|CubicSpline|Lanczos|Average|Min|Max|Median|Mode|Q1|Q3]
                                      Fallback spatial resampling algorithm to be
                                      used for all variables. Defaults to
                                      'Nearest'. The choices for the resampling
                                      algorithm are: dict_keys(['Nearest',
                                      'Bilinear', 'Cubic', 'CubicSpline',
                                      'Lanczos', 'Average', 'Min', 'Max',
                                      'Median', 'Mode', 'Q1', 'Q3'])
      -a, --append                    Append successive outputs.
      --sort                          The input file list will be sorted before
                                      creating the data cube. If --sort parameter
                                      is not passed, order of input list will be
                                      kept.
      -i, --info                      Displays additional information about format
                                      options or about input processors.
      --dry_run                       Just read and process inputs, but don't
                                      produce any outputs.
      --help                          Show this message and exit.


    $ xcube gen --info
    input processors to be used with option --proc:
      default                           Single-scene NetCDF/CF inputs in xcube standard format
      rbins-seviri-highroc-scene-l2     RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs
      rbins-seviri-highroc-daily-l2     RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs
      snap-olci-highroc-l2              SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs
      snap-olci-cyanoalert-l2           SNAP Sentinel-3 OLCI CyanoAlert Level-2 NetCDF inputs
      vito-s2plus-l2                    VITO Sentinel-2 Plus Level 2 NetCDF inputs
    
    
    output formats to be used with option --format:
      csv                     (*.csv)       CSV file format
      mem                     (*.mem)       In-memory dataset I/O
      netcdf4                 (*.nc)        NetCDF-4 file format
      zarr                    (*.zarr)      Zarr file format (http://zarr.readthedocs.io)
    

Example:

    $ xcube gen -p snap-olci-highroc-l2 -a -s 2000,1000 -r 0,50,5,52.5 -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -o hiroc-cube.zarr D:\OneDrive\BC\EOData\HIGHROC\2017\**\*.nc


## `xcube level`

Generate multi-resolution levels.

    $ xcube level --help
    Usage: xcube level [OPTIONS] <input>
    
      Generate multi-resolution levels. Transform the given dataset by <input>
      into the levels of a multi-level pyramid with spatial resolution
      decreasing by a factor of two in both spatial dimensions and write the
      result to directory <output>.
    
    Options:
      -o, --output <output>           Output directory. If omitted,
                                      "<input>.levels" will be used.
      -l, --link                      Link the <input> instead of converting it to
                                      a level zero dataset. Use with care, as the
                                      <input>'s internal spatial chunk sizes may
                                      be inappropriate for imaging purposes.
      -t, --tile-size <tile-size>     Tile size, given as single integer number or
                                      as <tile-width>,<tile-height>. If omitted,
                                      the tile size will be derived from the
                                      <input>'s internal spatial chunk sizes. If
                                      the <input> is not chunked, tile size will
                                      be 512.
      -n, --num-levels-max <num-levels-max>
                                      Maximum number of levels to generate. If not
                                      given, the number of levels will be derived
                                      from spatial dimension and tile sizes.
      --help                          Show this message and exit.

    
Example:

    $ xcube level -l -t 720 data/cubes/test-cube.zarr

## `xcube prune`

Delete empty chunks.

    $ xcube prune --help
    Usage: xcube prune [OPTIONS] INPUT
    
      Delete empty chunks. Deletes all block files associated with empty (NaN-
      only) chunks in given INPUT cube, which must have ZARR format.
    
    Options:
      --dry-run  Just read and process input, but don't produce any outputs.
      --help     Show this message and exit.



## `xcube vars2dim`

Convert cube variables into new dimension.

    $ xcube vars2dim --help
    Usage: xcube vars2dim [OPTIONS] <cube>
    
      Convert cube variables into new dimension. Moves all variables of <cube>
      into into a single new variable <var-name> with a new dimension <dim-name>
      and writes the results to <output>.
    
    Options:
      -d, --dim_name <dim-name>  Name of the new dimension into variables.
                                 Defaults to "var".
      -v, --var_name <var-name>  Name of the new variable that includes all
                                 variables. Defaults to "data".
      -o, --output <output>      Output file.
      -f, --format <format>      Format of the output. If not given, guessed from
                                 <output>.
      --help                     Show this message and exit.


## `xcube resample`

Resample data along the time dimension.

    $ xcube resample --help
    Usage: xcube resample [OPTIONS] INPUT
    
      Resample data along the time dimension.
    
    Options:
      -c, --config TEXT               Data cube configuration file in YAML format.
                                      More than one config input file is
                                      allowed.When passing several config files,
                                      they are merged considering the order passed
                                      via command line.
      -o, --output TEXT               Output path.
      -f, --format [zarr|netcdf4|mem]
                                      Output format. If omitted, format will be
                                      guessed from output path.
      --variables, --vars TEXT        Comma-separated list of names of variables
                                      to be included.
      -M, --method TEXT               Temporal resampling method. Available
                                      downsampling methods are 'count', 'first',
                                      'last', 'min', 'max', 'sum', 'prod', 'mean',
                                      'median', 'std', 'var', the upsampling
                                      methods are 'asfreq', 'ffill', 'bfill',
                                      'pad', 'nearest', 'interpolate'. If the
                                      upsampling method is 'interpolate', the
                                      option '--kind' will be used, if given.
                                      Other upsampling methods that select
                                      existing values honour the '--tolerance'
                                      option. Defaults to 'mean'.
      -F, --frequency TEXT            Temporal aggregation frequency. Use format
                                      "<count><offset>" where <offset> is one of
                                      'H', 'D', 'W', 'M', 'Q', 'Y'. Defaults to
                                      '1D'.
      -O, --offset TEXT               Offset used to adjust the resampled time
                                      labels. Uses same syntax as frequency. Some
                                      Pandas date offset strings are supported as
                                      well.
      -B, --base INTEGER              For frequencies that evenly subdivide 1 day,
                                      the origin of the aggregated intervals. For
                                      example, for '24H' frequency, base could
                                      range from 0 through 23. Defaults to 0.
      -K, --kind TEXT                 Interpolation kind which will be used if
                                      upsampling method is 'interpolation'. May be
                                      one of 'zero', 'slinear', 'quadratic',
                                      'cubic', 'linear', 'nearest', 'previous',
                                      'next' where 'zero', 'slinear', 'quadratic',
                                      'cubic' refer to a spline interpolation of
                                      zeroth, first, second or third order;
                                      'previous' and 'next' simply return the
                                      previous or next value of the point. For
                                      more info refer to
                                      scipy.interpolate.interp1d(). Defaults to
                                      'linear'.
      -T, --tolerance TEXT            Tolerance for selective upsampling methods.
                                      Uses same syntax as frequency. If the time
                                      delta exceeds the tolerance, fill values
                                      (NaN) will be used. Defaults to the given
                                      frequency.
      --dry-run                       Just read and process inputs, but don't
                                      produce any outputs.
      --help                          Show this message and exit.

Upsampling example:

    xcube resample --vars conc_chl,conc_tsm -F 12H -T 6H -M interpolation -K linear xcube\webapi\res\demo\cube.nc

Downsampling example:

    xcube resample --vars conc_chl,conc_tsm -F 3D -M mean -M std -M count xcube\webapi\res\demo\cube.nc

## `xcube serve`

Serve data cubes via web service. 

    $ xcube serve --help
    Usage: xcube serve [OPTIONS] CUBE...
    
      Serve data cubes via web service.
    
      Serves data cubes by a RESTful API and a OGC WMTS 1.0 RESTful and KVP
      interface. The RESTful API documentation can be found at
      https://app.swaggerhub.com/apis/bcdev/xcube-server.
    
    Options:
      -a, --address ADDRESS  Service address. Defaults to 'localhost'.
      -p, --port PORT        Port number where the service will listen on.
                             Defaults to 8080.
      --prefix PREFIX        Service URL prefix. May contain template patterns
                             such as "${version}" or "${name}". For example
                             "${name}/api/${version}".
      -u, --update PERIOD    Service will update after given seconds of
                             inactivity. Zero or a negative value will disable
                             update checks. Defaults to 2.0.
      -S, --styles STYLES    Color mapping styles for variables. Used only, if one
                             or more CUBE arguments are provided and CONFIG is not
                             given. Comma-separated list with elements of the form
                             <var>=(<vmin>,<vmax>) or
                             <var>=(<vmin>,<vmax>,"<cmap>")
      -c, --config CONFIG    Use datasets configuration file CONFIG. Cannot be
                             used if CUBES are provided.
      --tilecache SIZE       In-memory tile cache size in bytes. Unit suffixes
                             'K', 'M', 'G' may be used. Defaults to '512M'. The
                             special value 'OFF' disables tile caching.
      --tilemode MODE        Tile computation mode. This is an internal option
                             used to switch between different tile computation
                             implementations. Defaults to 0.
      -s, --show             Run viewer app. Requires setting the environment
                             variable XCUBE_VIEWER_PATH to a valid xcube-viewer
                             deployment or build directory. Refer to
                             https://github.com/dcs4cop/xcube-viewer for more
                             information.
      -v, --verbose          Delegate logging to the console (stderr).
      --traceperf            Print performance diagnostics (stdout).
      --help                 Show this message and exit.


### Objective

The `xcube serve` is a light-weight web server that provides various services based on 
xcube data cubes:

* Catalogue services to query for datasets and their variables and dimensions, and feature collections. 
* Tile map service, with some OGC WMTS 1.0 compatibility (REST and KVP APIs)
* Dataset services to extract subsets like time-series and profiles for e.g. JS clients 

Find its API description [here](https://app.swaggerhub.com/apis-docs/bcdev/xcube-server). 

xcube datasets are any datasets that 

* that comply to [Unidata's CDM](https://www.unidata.ucar.edu/software/thredds/v4.3/netcdf-java/CDM/) and to the [CF Conventions](http://cfconventions.org/); 
* that can be opened with the [xarray](https://xarray.pydata.org/en/stable/) Python library;
* that have variables that have at least the dimensions and shape (`time`, `lat`, `lon`), in exactly this order; 
* that have 1D-coordinate variables corresponding to the dimensions;
* that have their spatial grid defined in the WGS84 (`EPSG:4326`) coordinate reference system.

The Xcube server supports local NetCDF files or local or remote [Zarr](https://zarr.readthedocs.io/en/stable/) directories.
Remote Zarr directories must be stored in publicly accessible, AWS S3 compatible 
object storage (OBS).

As an example, here is the [configuration of the demo server](https://github.com/bcdev/xcube-server/blob/master/xcube_server/res/demo/config.yml).

### OGC WMTS compatibility

The Xcube server implements the RESTful and KVP architectural styles
of the [OGC WMTS 1.0.0 specification](http://www.opengeospatial.org/standards/wmts).

The following operations are supported:

* **GetCapabilities**: `/xcube/wmts/1.0.0/WMTSCapabilities.xml`
* **GetTile**: `/xcube/wmts/1.0.0/tile/{DatasetName}/{VarName}/{TileMatrix}/{TileCol}/{TileRow}.png`
* **GetFeatureInfo**: *in progress*

### Explore API of existing xcube-servers

To explore the API of existing xcube-servers go to the [SwaggerHub of bcdev](https://app.swaggerhub.com/apis/bcdev/xcube-server/0.1.0.dev6).
The SwaggerHub allows to choose the xcube-server project and therefore the datasets which are used for the exploration. 

### Run the demo

#### Server

To run the server on default port 8080 using the demo configuration:

    $ xcube serve -v -c xcube/webapi/res/demo/config.yml

To run the server using a particular data cube path and styling information for a variable:

    $ xcube serve --styles conc_chl=(0,20,"viridis") /path/to/my/chl-cube.zarr

Test it:

* Datasets (Data Cubes):
    * [Get datasets](http://localhost:8080/datasets)
    * [Get dataset details](http://localhost:8080/datasets/local)
    * [Get dataset coordinates](http://localhost:8080/datasets/local/coords/time)
* Color bars:
    * [Get color bars](http://localhost:8080/colorbars)
    * [Get color bars (HTML)](http://localhost:8080/colorbars.html)
* WMTS:
    * [Get WMTS KVP Capabilities (XML)](http://localhost:8080/wmts/kvp?Service=WMTS&Request=GetCapabilities)
    * [Get WMTS KVP local tile (PNG)](http://localhost:8080/wmts/kvp?Service=WMTS&Request=GetTile&Version=1.0.0&Layer=local.conc_chl&TileMatrix=0&TileRow=0&TileCol=0&Format=image/png)
    * [Get WMTS KVP remote tile (PNG)](http://localhost:8080/wmts/kvp?Service=WMTS&Request=GetTile&Version=1.0.0&Layer=remote.conc_chl&TileMatrix=0&TileRow=0&TileCol=0&Format=image/png)
    * [Get WMTS REST Capabilities (XML)](http://localhost:8080/wmts/1.0.0/WMTSCapabilities.xml)
    * [Get WMTS REST local tile (PNG)](http://localhost:8080/wmts/1.0.0/tile/local/conc_chl/0/0/1.png)
    * [Get WMTS REST remote tile (PNG)](http://localhost:8080/wmts/1.0.0/tile/remote/conc_chl/0/0/1.png)
* Tiles
    * [Get tile (PNG)](http://localhost:8080/datasets/local/vars/conc_chl/tiles/0/1/0.png)
    * [Get tile grid for OpenLayers 4.x](http://localhost:8080/datasets/local/vars/conc_chl/tilegrid?tiles=ol4)
    * [Get tile grid for Cesium 1.x](http://localhost:8080/datasets/local/vars/conc_chl/tilegrid?tiles=cesium)
    * [Get legend for layer (PNG)](http://localhost:8080/datasets/local/vars/conc_chl/legend.png)
* Time series service (preliminary & unstable, will likely change soon)
    * [Get time stamps per dataset](http://localhost:8080/ts)
    * [Get time series for single point](http://localhost:8080/ts/local/conc_chl/point?lat=51.4&lon=2.1&startDate=2017-01-15&endDate=2017-01-29)
* Places service (preliminary & unstable, will likely change soon)
    * [Get all features](http://localhost:8080/places/all)
    * [Get all features of collection "inside-cube"](http://localhost:8080/features/inside-cube)
    * [Get all features for dataset "local"](http://localhost:8080/places/all/local)
    * [Get all features of collection "inside-cube" for dataset "local"](http://localhost:8080/places/inside-cube/local)


#### Clients

There are example HTML pages for some tile server clients. They need to be run in 
a web server. If you don't have one, you can use Node's `httpserver`:

    $ npm install -g httpserver
    
After starting both the xcube server and web server, e.g. on port 9090

    $ httpserver -d -p 9090

you can run the client demos by following their links given below.
    
   
##### OpenLayers

[OpenLayers 4 Demo](http://localhost:9090/xcube/webapi/res/demo/index-ol4.html)
[OpenLayers 4 Demo with WMTS](http://localhost:9090/xcube/webapi/res/demo/index-ol4-wmts.html)

##### Cesium

To run the [Cesium Demo](http://localhost:9090/xcube/webapi/res/demo/index-cesium.html) first
[download Cesium](https://cesiumjs.org/downloads/) and unpack the zip
into the `xcube-server` source directory so that there exists an 
`./Cesium-<version>` sub-directory. You may have to adapt the Cesium version number 
in the [demo's HTML file](https://github.com/dcs4cop/xcube/blob/master/xcube/webapi/res/demo/index-cesium.html).

### Xcube server TODOs:

* Bug/Performance: ServiceContext.dataset_cache uses dataset names as ID, but actually, caching of *open* datasets 
  should be based on *same* dataset sources, namely given the local file path or the remote URL.
  There may be different identifiers that have the same path!
* Bug/Performance: /xcube/wmts/1.0.0/WMTSCapabilities.xml slow for ZARR data cubes in OTC's object storage.
  15 seconds for first call - investigate and e.g. cache.
* Performance: After some period check if datasets haven't been used for a long time - close them and remove from cache.
* Feature: WMTS GetFeatureInfo
* Feature: collect Path entry of any Dataset and observe if the file are modified, if so remove dataset from cache
  to force its reopening.


## `xcube verify`

Perform cube verification.

    Usage: xcube verify [OPTIONS] INPUT
    
      Perform cube verification.
    
      The tool verifies that INPUT
      * defines the dimensions "time", "lat", "lon";
      * has corresponding "time", "lat", "lon" coordinate variables and that they
        are valid, e.g. 1-D, non-empty, using correct units;
      * has valid  bounds variables for "time", "lat", "lon" coordinate
        variables, if any;
      * has any data variables and that they are valid, e.g. min. 3-D, all have
        same dimensions, have at least dimensions "time", "lat", "lon".
    
      If INPUT is a valid data cube, the tool returns exit code 0. Otherwise a
      violation report is written to stdout and the tool returns exit code 3.
    
    Options:
      --help  Show this message and exit.
