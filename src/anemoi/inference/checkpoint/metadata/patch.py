# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


def zarr_fill(metadata):
    metadata["variables"] = metadata["attrs"]["variables"]
    return metadata


def drop_fill(metadata):
    metadata["variables"] = [x for x in metadata["forward"]["variables"] if x not in metadata["drop"]]
    return metadata


def select_fill(metadata):
    metadata["variables"] = [x for x in metadata["forward"]["variables"] if x in metadata["select"]]
    return metadata


def rename_fill(metadata, select):

    rename = metadata["rename"]
    variables = metadata["forward"]["variables"]
    if select is not None:
        variables = [x for x in variables if x in select]

    metadata["variables"] = [rename.get(x, x) for x in variables]

    return metadata


def join_fill(metadata):
    variables = []
    for dataset in metadata["datasets"]:
        for k in dataset["variables"]:
            if k not in variables:
                variables.append(k)

    metadata["variables"] = variables
    return metadata


def patch(a, b):
    if "drop" in a:
        return drop_fill(
            {
                "action": "drop",
                "drop": a["drop"],
                "forward": zarr_fill({"action": "zarr", "attrs": b}),
            }
        )

    if "select" in a:
        return select_fill(
            {
                "action": "select",
                "select": a["select"],
                "forward": zarr_fill({"action": "zarr", "attrs": b}),
            }
        )

    if "rename" in a:
        return rename_fill(
            {
                "action": "rename",
                "rename": a["rename"],
                "forward": zarr_fill({"action": "zarr", "attrs": b}),
            },
            a.get("select"),
        )

    raise NotImplementedError(f"Cannot patch {a}")


def list_to_dict(datasets, config):

    arguments = config["dataloader"]["training"]
    assert "dataset" in arguments
    assert isinstance(arguments["dataset"], list)
    assert len(arguments["dataset"]) == len(datasets)

    patched = [patch(a, b) for a, b in zip(arguments["dataset"], datasets)]

    return {
        "specific": join_fill({"action": "join", "datasets": patched}),
        "version": "0.2.0",
        "arguments": arguments,
    }
