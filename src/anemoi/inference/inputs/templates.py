# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import base64
import logging
import zlib

import earthkit.data as ekd

from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)

# 1 - Get a GRIB with mars: retrieve,param=tp,levtype=sfc,type=fc,step=6,target=data.grib,grid=0.25/0.25
# 2 - grib_set -s edition=2,packingType=grid_ccsds data.grib data.grib2
# 3 - grib_set -d 0 data.grib2 out.grib
# 4 - python -c 'import base64, sys, zlib;print(base64.b64encode(zlib.compress(open(sys.argv[1], "rb").read())))' out.grib

# tp in grib2, 0.25/0.25 grid
TEMPLATE_0p25 = """
eJxzD/J0+v+fgYkBAh4AsSgjQxIDgyIjI/sLbnEeoAAjEAsyASlGBk4WRgMDA0OggAczAwP/9QVgXf9RAQMDK1Cc6SJMijXWqw
HENmgFMkQLIwsYmC9MAGMgsGIBEhyMBxn+zwMqBVnFxohkFsgNQhAnMDAygan/ULdKsoIdoAViNwiDSD4FBpBFbGwgJazs5kAA
AKlBO70=
"""

TEMPLATE_N320 = """
eJztlDFLw0AYht87m1ZRCgpFCg4VHEQs1C6KOASpqDjVSRykiIJDBzddhAyKji46SsE/0MHd6uTg4KiTuoiTODh1ie93SfXaSO
gP8B5yd4TjI1y+91lcXZ73fWiY4bxwyihsAqNKpd77hwf4QvEZ1FwU+hKqUChM8eRSD9C74GmFcfjRAWhPKsreWdd5U/6Im8zu
80ZwQLnAELIYY4U8ipiDiyWygjLWUcEWqtgnBzjGKTlHDVeGBrnDo+EVn4YmPy9NcoZJ4hpKasdij3jklJypesit4eWHN/UlaF
6LHvnBtVgL8do4IZchDxGe9EdAD0JyFhOk8ieHHTRiuCfv7SRgMRPDdgxehHoXXNs4+Kcb2m+tq3uO/p24fxnXA1avdPaR6a1G
DJ19+nc3S5/bfR9moZUNpiSanFamJF/tiWvl0M7mb2KZ3yDJTPRvuoO0t7J/ZlzgEXGD7YpSaBBxSWAV8QvQDI3zGhrozvgoMF
ONlhJXHdNa4q4qLVahzcq0mrjNpeWKtN04rZel/YDZBKdedQP/glYU1SaV5VFx8FCgYChtFj9wNbKOCBgTsvekFNI5iHSTSTni
pKY5vgHl2Dri
"""

TEMPLATE_O96 = """
eJxlkjFIW1EUhv//mailoVSNaCFDChlEMjwhoEgGkYAWMohkcAgllEAdgjhkcMiQ4RUcMjg4ONji4ODg4ODg4KDg4ODQwaGDQw
cHB4cODh2E1+/FDoXex7v3cO6555z/u3dp9cNiHCtQfwQNpnHrk/TeHnp4PZHBYf6RgMV6lXIYhjP4rgfwfu0G1pTi/webF0nG
xE6vfoz66b9gjH/eaPwNoFhWk8oprwJZigpV0pzKWlBFy6pqRTWtqU5gU+tqaVNtbamjriJtq6cd7WpP+zrQoY50rBOd6kznVL
7StW70Xbf6oTv91L0e9KhfetJvPSMk5WFn/NZZTzrnvAuectGhS55z2QuueNlVr7jmNdfdcNPrbnnTbW+5464jb7vnHb4eVoSn
w06biBaRTU7UOVkjQ5VMFTKWyVyiQpFKBSrmqJylgwydpCD8TGdPdPhIp/d0fEfntyi4QckVis5RdorCY5Qeongf5bsQ6EEigk
gHMm0ItSDVhFgdcjUIViFZgWgZsiUIFyFdgHgO8lkuYT7FNOxLxd+4leSqB/3PPSZvYPTlCchBf4lf3orepZMHoOnE7o4l85s8
jZBgMAlJD80y/gBx3pOi
"""

TEMPLATES = {(0.25, 0.25): TEMPLATE_0p25, "N320": TEMPLATE_N320, "O96": TEMPLATE_O96}


@input_registry.register("templates")
class TemplatesInput(Input):
    """A dummy input that creates GRIB templates"""

    def __init__(self, context, *args, **kwargs):
        super().__init__(context)

    def __repr__(self):
        return f"TemplatesInput({self.context.checkpoint.grid})"

    def create_input_state(self, *, date):
        raise NotImplementedError("TemplatesInput.create_input_state() not implemented")

    def template(self, variable, date, **kwargs):

        # import eccodes
        typed = self.context.checkpoint.typed_variables[variable]

        if not typed.is_accumulation:
            return None

        grid = self.context.checkpoint.grid
        if isinstance(grid, str):
            grid = grid.upper()

        template = zlib.decompress(base64.b64decode(TEMPLATES[grid]))
        # eccodes.codes_new_from_message(template)

        return ekd.from_source("memory", template)[0]

    def load_forcings(self, variables, dates):
        raise NotImplementedError("TemplatesInput.load_forcings() not implemented")
