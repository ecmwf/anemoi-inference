# anemoi/inference/pre_processors/mars_fc_time_to_step.py
from typing import Any, List
from anemoi.inference.processor import Processor
from anemoi.inference.pre_processors import pre_processor_registry

@pre_processor_registry.register("mars_fc_time_to_step")
class MarsFcTimeToStep(Processor):
    """
    Patch fc MARS requests by converting multiple times (e.g. 0000/0600)
    into a single base time with steps (e.g. time=0000, step=0/6).

    Config:
      - base_time: str | None  ("0000" to force midnight; if None, uses earliest)
      - align_date: "min" | "first"  (pick earliest date or the first one)
    """

    def __init__(self, context, base_time: str | None = None, align_date: str = "min") -> None:
        super().__init__(context)
        self.base_time = base_time
        self.align_date = align_date

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
        steps_in = [int(s) for s in as_list(r.get("step"))] if r.get("step") is not None else []

        # Nothing to collapse
        if len(times) <= 1 and len(dates) <= 1:
            return r

        # Choose base date/time
        base_date = min(dates) if self.align_date == "min" and dates else (dates[0] if dates else None)
        base_time = self.base_time or (min(times) if times else "0000")

        # Build step list from times relative to base_time (hour differences)
        def t2h(t: str) -> int:
            return int(t[:2])

        hour0 = t2h(base_time)
        derived_steps = [t2h(t) - hour0 for t in times]
        new_steps = sorted(set([s for s in steps_in + derived_steps if s >= 0]))

        if base_date is not None:
            r["date"] = base_date
        r["time"] = base_time
        if new_steps:
            r["step"] = new_steps

        return r