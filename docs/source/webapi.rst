.. _`WMTS`: https://en.wikipedia.org/wiki/Web_Map_Tile_Service
.. _`xcube Web API reference`: https://app.swaggerhub.com/apis/bcdev/xcube-server/v0.11.2


==================
Web API and Server
==================

xcube's RESTful web API is used to publish data cubes to clients. Using the API, clients can

* List configured xcube datasets;
* Get xcube dataset details including metadata, coordinate data, and metadata about all included variables;
* Get cube data;
* Extract time-series statistics from any variable given any geometry;
* Get spatial image tiles from any variable;
* Get places (GeoJSON features including vector data) that can be associated with xcube datasets.

Later versions of API will also allow for xcube dataset management including generation, modification, and deletion
of xcube datasets.

The complete description of all available functions is provided in the in the `xcube Web API reference`_.

The web API is provided through the *xcube server* which is started using the :doc:`cli/xcube_serve` CLI command.
