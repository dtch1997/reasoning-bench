from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import match
from inspect_ai.solver import generate


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["prompt"],
        target=record["label"]["final"],
    )


TaskName = Literal[
    "task01",
    "task02",
    "task03",
    "task04",
    "task05",
    "task06",
    "task07",
    "task08",
    "task09",
    "task10",
    "task11",
    "task12",
    "task13",
    "task14",
    "task15",
    "task16",
    "task17",
    "task18",
    "task19",
    "task20",
    "task21",
    "task22",
    "task23",
]


@task(name="procbench")
def procbench(
    name: TaskName = "task01",
    split: Literal["train", "validation"] = "train",
):
    dataset = hf_dataset(
        path="ifujisawa/procbench",
        name=name,
        sample_fields=record_to_sample,
        split=split,
        auto_id=True,
        shuffle=True,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(location="any"),
        config=GenerateConfig(
            temperature=0.0,
        ),
    )
