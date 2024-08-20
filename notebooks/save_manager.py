import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

TYPE = Optional[Union[str, List[Union[Dict[str, Any], str]]]]


class SaveManager:

    def __init__(
        self,
        categories: Optional[List[str]],
        presidents: Optional[List[str]],
        path_dir="./tmp",
    ):
        if not categories:
            categories = []
        if not presidents:
            presidents = []

        self.dir = path_dir
        self.path_file = os.path.join(self.dir, "data.json")
        all_timer = self.load()
        self.label = (
            "-".join(categories) + "_" + "-".join(f"[{name}]" for name in presidents)
        )

        if not self.label in all_timer:
            all_timer[self.label] = {}
        all_timer[self.label]["categories"] = categories
        all_timer[self.label]["presidents"] = presidents
        self.save(all_timer)

    def timer(self, encapsulte_dicts: TYPE = None) -> "Timer":
        all_timer = self.load()

        d, label_timer = self.find_place(all_timer, encapsulte_dicts)
        return Timer(lambda: self.save(all_timer), d, label_timer)

    def fill(self, encapsulte_dicts: TYPE, value: Any) -> None:
        all_timer = self.load()
        d, label_timer = self.find_place(all_timer, encapsulte_dicts)
        d[label_timer] = value
        self.save(all_timer)

    def find_place(self, all_timer: dict, encapsulte_dicts: TYPE) -> Tuple[dict, str]:

        d = all_timer[self.label]
        label_timer = None

        if encapsulte_dicts is None:
            return d, label_timer

        if isinstance(encapsulte_dicts, str):
            encapsulte_dicts = [encapsulte_dicts]

        for index, e in enumerate(encapsulte_dicts):
            if e is None:
                continue

            label = (
                "_".join(f"[{name}:{str(val)}]" for name, val in e.items())
                if isinstance(e, dict)
                else e
            )

            if index + 1 == len(encapsulte_dicts):
                if isinstance(e, str):
                    label_timer = label
                    break

            if not label in d:
                d[label] = (
                    {name: val for name, val in e.items()}
                    if isinstance(e, dict)
                    else {}
                )

            d = d[label]

        return d, label_timer

    def load(self):
        if not os.path.exists(self.path_file):
            return {}
        with open(self.path_file, "r") as reader:
            return json.load(reader)

    def save(self, all_timer: dict) -> None:
        os.makedirs(name=self.dir, exist_ok=True)
        with open(self.path_file, "w") as writer:
            json.dump(all_timer, writer)


class Timer:
    def __init__(
        self, save: Callable[[], None], to_fill: Dict[str, Any], label: str = None
    ):
        self.to_fill = to_fill
        self.label = label if label else "time"
        self.save = save

    def begin(self):
        self.start = time.time()

    def end(self):
        self.to_fill[self.label] = time.time() - self.start
        self.save()

    def __enter__(self):
        self.begin()

    def __exit__(self, *args):
        self.end()
