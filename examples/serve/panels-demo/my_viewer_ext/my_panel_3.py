import pandas as pd
from typing import Any
import altair as alt
import pyproj
import shapely
import shapely.ops

from chartlets import Component, Input, State, Output
from chartlets.components import Box, Button, CircularProgress, Plot, Select, Typography

from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.webapi.viewer.contrib import get_datasets_ctx
from xcube.server.api import Context
from xcube.constants import CRS_CRS84
from xcube.core.geom import mask_dataset_by_geometry, normalize_geometry
from xcube.core.gridmapping import GridMapping

panel = Panel(__name__, title="Spectral")


@panel.layout(
    State(
        "@app",
        "selectedDatasetId",
    ),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGeometry"),
    State("@app", "selectedVariableName"),
)
def render_panel(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    place_geometry: dict[str, Any],
    variable_name: str,
) -> Component:

    dataset = get_dataset(ctx, dataset_id)

    plot = Plot(id="plot", chart=None, style={"paddingTop": 6})

    ds_ctx = get_datasets_ctx(ctx)
    ds_configs = ds_ctx.get_dataset_configs()

    text = (
        f"{ds_configs[0]['Title']} "
        f"/ {time_label[0:-1]} / "
        f"{round(place_geometry['coordinates'][0], 3)}, {round(place_geometry['coordinates'][1], 3)} / "
        f"{variable_name}"
        #   f"{dataset}______________ "
        #   f"{dataset.variables}_______________"
        #   f"{dataset.data_vars}"
    )

    place_text = Typography(id="text", children=[text], color="pink")

    wavelengths = {}
    for var_name, var in dataset.items():
        if "wavelength" in var.attrs:
            wavelengths[var_name] = var.attrs["wavelength"]

    button = Button(id="button", text="ADD Spectral View")  # , style={"maxWidth": 100})

    controls = Box(
        children=[button],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": 6,
            "padding": 6,
        },
    )

    return Box(
        children=[place_text, plot, controls],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "width": "100%",
            "height": "100%",
            "gap": 6,
            "padding": 6,
        },
    )


def get_wavelength(
    dataset,
    time_label: float,
    place_geometry: dict[str, Any],
) -> pd.DataFrame:

    grid_mapping = GridMapping.from_dataset(dataset)
    place_geometry = normalize_geometry(place_geometry)
    if place_geometry is not None and not grid_mapping.crs.is_geographic:
        project = pyproj.Transformer.from_crs(
            CRS_CRS84, grid_mapping.crs, always_xy=True
        ).transform
        place_geometry = shapely.ops.transform(project, place_geometry)

    dataset = mask_dataset_by_geometry(dataset, place_geometry)
    if dataset is None:
        # TODO: set error message in panel UI
        print("dataset is None after masking, invalid geometry?")
        return None

    if "time" in dataset.coords:
        if time_label:
            dataset = dataset.sel(time=pd.Timestamp(time_label[0:-1]), method="nearest")
        else:
            dataset = dataset.isel(time=-1)

    variables = []
    wavelengths = []
    for var_name, var in dataset.items():
        if "wavelength" in var.attrs:
            wavelengths.append(var.attrs["wavelength"])
            variables.append(var_name)

    source = []
    for var in variables:
        value = dataset[var].item()
        source.append({"variable": var, "reflectance": value})

    results_df = pd.DataFrame(source)
    results_df["wavelength"] = wavelengths
    return results_df

    # import random
    # #reflectances = range(len(wavelengths))
    # reflectances = [random.uniform(min(wavelengths), max(wavelengths)) for _ in
    #                  range(len(wavelengths))]
    #
    # source = pd.DataFrame(
    #     {'wavelength': wavelengths, 'reflectance': reflectances}
    # )
    # return source


@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGeometry"),
    Input("button", "clicked"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id: str | None = None,
    time_label: float | None = None,
    place_geometry: dict[str, Any] | None = None,
    _clicked: bool | None = None,  # trigger, will always be True
) -> alt.Chart | None:
    dataset = get_dataset(ctx, dataset_id)

    source = get_wavelength(dataset, time_label, place_geometry)
    # source = pd.DataFrame(
    # {'wavelength': [1, 2, 3, 4], 'reflectance': [1, 2, 3, 4]}
    # )

    # TODO - message if source==None

    chart = (
        alt.Chart(source)
        .mark_line(point=True)
        .encode(
            x="wavelength",
            y="reflectance",
            # color=alt.Color("reflectance"),
            tooltip=["wavelength", "reflectance"],  # ,'variable'
        )
    ).properties(title="Spectral Chart", width=360, height=160)

    return chart
