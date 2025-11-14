# anemoi/inference/pre_processors/mars_fc_time_to_step.py
from typing import Any, List
from anemoi.inference.processor import Processor
from anemoi.inference.pre_processors import pre_processor_registry
from earthkit.data.utils.dates import to_datetime
from datetime import timedelta
from earthkit.data.utils.dates import to_timedelta
import itertools
import logging


LOG = logging.getLogger(__name__)


@pre_processor_registry.register("mars_fc_time_to_step")
class MarsFcTimeToStep(Processor):
    """
    Patch fc MARS requests by converting multiple times (e.g. 0000/0600)
    into a single base time with steps (e.g. time=0000, step=0/6).

    Config:
      - base_time: str | None  ("0000" to force midnight; if None, uses earliest)
      - align_date: "min" | "first"  (pick earliest date or the first one)
    """

    def __init__(self, context, base_date_time: str | None = None) -> None:
        super().__init__(context)
        self.base_date_time = base_date_time

    def process(self, state):
        return state

    def patch_data_request(self, req: dict[str, Any]) -> dict[str, Any]:
        r = dict(req)  # shallow copy is fine here

        # if r.get("type") != "fc":
        #     return r

        def as_list(x):
            if x is None:
                return []
            if isinstance(x, (list, tuple, set)):
                return list(x)
            return [x]

        times: List[str] = [str(t).zfill(4) for t in as_list(r.get("time"))]
        dates: List[str] = [str(d) for d in as_list(r.get("date"))]
        date_times: List[datetime] = [to_datetime(d + ":" + t) for d, t in itertools.product(dates, times)]

        

        # Handling the Accumulation case - mars request incorrectly moves the date back by the step
        if any( var in self.context.checkpoint.accumulations for var in r['param']):
            assert len(r['date']) == 1 and len(r['time']) == 1 and len(r['step']) == 1, "Accumulation case requires single date, time, and step"
            
            steps_in = [int(s) for s in as_list(r.get("step"))] if r.get("step") is not None else []
            
            curr_date_time = to_datetime(r['date'][0] + ":" + r['time'][0])

            
            # Correct date time - based on forecast start date
            fixed_date_time = to_datetime(self.base_date_time) 


            # Correcting the step - based on the current
            steps_in = r['step'] if isinstance(r['step'], list) else [r['step']]
            correct_step = [ curr_date_time + to_timedelta(step) - fixed_date_time for step in steps_in ]
            correct_step = [ int(step.total_seconds() / 3600) for step in correct_step ]


            r['date'] = fixed_date_time.strftime("%Y-%m-%d")
            r['time'] = fixed_date_time.strftime("%H%M")
            r['step'] = correct_step
            return r


        # Nothing to collapse
        if len(times) <= 1 and len(dates) <= 1:
            return r

        # This is the 

        base_date_time = to_datetime(self.base_date_time)
        base_date = base_date_time.strftime("%Y-%m-%d")
        base_time = base_date_time.strftime("%H%M")

        time_deltas = [date_time - base_date_time for date_time in date_times]
        time_deltas = [ int(delta.total_seconds() / 3600) for delta in time_deltas ] # in hours/steps

        # Build step list from times relative to base_time (hour differences)
        new_steps = sorted(set([s for s in time_deltas if s >= 0]))

        r["date"] = [base_date]
        r["time"] = [base_time]
        r["step"] = new_steps

        return r