from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import match
from inspect_ai.solver import generate

TEMPLATE = """
Use the following context to answer the question:
{context}

Question: {question}
"""


def record_to_sample(record: dict[str, Any]) -> Sample:
    context_sentences: list[str] = []
    for sentences in record["context"]:
        context_sentences.extend(sentences)
    context = " ".join(context_sentences)

    return Sample(
        input=TEMPLATE.format(context=context, question=record["question"]),
        target=record["answer"],
    )


@task(name="hotpot_qa")
def hotpot_qa(
    name: Literal["fullwiki", "distractor"] = "fullwiki",
    split: Literal["train", "validation"] = "train",
):
    dataset = hf_dataset(
        path="hotpot_qa",
        name=name,
        sample_fields=record_to_sample,
        split="train",
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
