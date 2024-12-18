from chartlets import Component, Input, State, Output
from chartlets.components import Box, Button, CircularProgress, Plot, Select, Typography

from xcube.webapi.viewer.contrib import Panel
from xcube.webapi.viewer.contrib import get_dataset
from xcube.server.api import Context


panel = Panel(__name__, title="Spectral")


@panel.layout(
    State("@app", "selectedDatasetId",),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlace"),
)
def render_panel(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    place_geometry: str,
) -> Component:
    dataset = get_dataset(ctx, dataset_id)
   # plot = Plot(id="plot", chart=None, style={"paddingTop": 6})

    place_text = Typography(id="text", children=[f"selected place_{type(place_geometry)}"], color='pink')
    button = Button(id="button", text="ADD", style={"maxWidth": 100})

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
        children=[place_text, controls],
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

