import holoviews as hv
import xarray as xr

from foreal.core import Node

# from holoviews.operation.datashader import datashade


def concat_coords(da):
    string = ""
    for coord in da.coords:
        if len(da.coords[coord]) == 1:
            string += " " + str(da.coords[coord].values.squeeze())
    if not string:
        string = "name"
    return string


class HoloViewPlot(Node):
    def __init__(self):
        super().__init__()

    def forward(self, data, request):
        components = []
        if not isinstance(data, list):
            data = [data]
        for example in data:
            if isinstance(example, xr.DataArray):
                example.name = concat_coords(example)

                if "frequency" in example.dims and "time" in example.dims:
                    example.coords["time"] = (example.coords["time"]).values.astype(
                        float
                    )
                    # make sure we have enough dimensions such that holoview does not fail
                example = example.expand_dims({"plot": [0]})

                hv_ds = hv.Dataset(example)

                if "frequency" in example.dims and "time" in example.dims:
                    plot = hv_ds.to(
                        hv.Image, kdims=["time", "frequency"], dynamic=False
                    ).opts(colorbar=False, xaxis=None, yaxis=None, cmap="bgy")
                else:
                    kdim = "time"
                    if "frequency" in example.dims:
                        kdim = "frequency"

                    plot = hv_ds.to(hv.Curve, kdims=[kdim])

                components += [plot]

        return components
