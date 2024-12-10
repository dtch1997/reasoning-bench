from typing import Any

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
    context: str = ""
    for paragraph in record["paragraphs"]:
        context += paragraph["paragraph_text"]
        context += "\n"

    prompt: str = TEMPLATE.format(context=context, question=record["question"])

    return Sample(input=prompt, target=record["answer"])


@task(name="musique")
def musique():
    dataset = hf_dataset(
        path="fladhak/musique",
        sample_fields=record_to_sample,
        split="validation",
        auto_id=True,
        shuffle=True,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(location='any'),
        config=GenerateConfig(
            temperature=0.0,
        ),
    )
