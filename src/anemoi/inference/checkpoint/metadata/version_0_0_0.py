# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from . import Metadata

LOG = logging.getLogger(__name__)


class Version_0_0_0(Metadata):
    """
    Reference for very old checkpoints
    Will not work and need to be updated
    """

    def __init__(self, metadata):
        super().__init__(metadata)

        FORCING_PARAMS = [
            "z",
            "lsm",
            "sdor",
            "slor",
            "cos_latitude",
            "cos_longitude",
            "sin_latitude",
            "sin_longitude",
            "cos_julian_day",
            "cos_local_time",
            "sin_julian_day",
            "sin_local_time",
            "insolation",
        ]

        indices = dict(
            forcing=self._index_of(FORCING_PARAMS),
            full=self._index_of(self.variables),
            diagnostic=[],
            prognostic=self._index_of(self.ordering),
        )

        config = dict(
            data_indices=dict(
                data=dict(
                    input=indices,
                    output=indices,
                ),
                model=dict(
                    input=indices,
                    output=indices,
                ),
            ),
            config=dict(
                data=dict(timestep=6, frequency=6),
                training=dict(
                    multistep_input=2,
                    precision="32",
                ),
            ),
        )

        self._metadata.update(config)

    def _index_of(self, names):
        return [self.variable_to_index[name] for name in names]

    def dump(self, indent=0):
        print("Version_0_0_0: Not implemented")

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = [
        "z",
        "sp",
        "msl",
        "lsm",
        "sst",
        "sdor",
        "slor",
        "10u",
        "10v",
        "2t",
        "2d",
    ]
    param_level_pl = (
        ["q", "t", "u", "v", "w", "z"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    param_level_ml = ([], [])

    ordering = [
        "q_50",
        "q_100",
        "q_150",
        "q_200",
        "q_250",
        "q_300",
        "q_400",
        "q_500",
        "q_600",
        "q_700",
        "q_850",
        "q_925",
        "q_1000",
        "t_50",
        "t_100",
        "t_150",
        "t_200",
        "t_250",
        "t_300",
        "t_400",
        "t_500",
        "t_600",
        "t_700",
        "t_850",
        "t_925",
        "t_1000",
        "u_50",
        "u_100",
        "u_150",
        "u_200",
        "u_250",
        "u_300",
        "u_400",
        "u_500",
        "u_600",
        "u_700",
        "u_850",
        "u_925",
        "u_1000",
        "v_50",
        "v_100",
        "v_150",
        "v_200",
        "v_250",
        "v_300",
        "v_400",
        "v_500",
        "v_600",
        "v_700",
        "v_850",
        "v_925",
        "v_1000",
        "w_50",
        "w_100",
        "w_150",
        "w_200",
        "w_250",
        "w_300",
        "w_400",
        "w_500",
        "w_600",
        "w_700",
        "w_850",
        "w_925",
        "w_1000",
        "z_50",
        "z_100",
        "z_150",
        "z_200",
        "z_250",
        "z_300",
        "z_400",
        "z_500",
        "z_600",
        "z_700",
        "z_850",
        "z_925",
        "z_1000",
        "sp",
        "msl",
        "sst",
        "10u",
        "10v",
        "2t",
        "2d",
        "z",
        "lsm",
        "sdor",
        "slor",
    ]

    param_format = {"param_level": "{param}{levelist}"}

    computed_constants = [
        "cos_latitude",
        "cos_longitude",
        "sin_latitude",
        "sin_longitude",
    ]

    computed_forcing = [
        "cos_julian_day",
        "cos_local_time",
        "sin_julian_day",
        "sin_local_time",
        "insolation",
    ]

    @property
    def variables(self):
        return self.ordering + self.computed_constants + self.computed_forcing

    @property
    def variables_with_nans(self):
        return []

    ###########################################################################
    @property
    def order_by(self):
        return dict(
            valid_datetime="ascending",
            param_level=self.ordering,
            remapping={"param_level": "{param}_{levelist}"},
        )

    @property
    def select(self):
        return dict(
            param_level=self.variables,
            remapping={"param_level": "{param}_{levelist}"},
        )

    def graph_kids(self):
        return []
