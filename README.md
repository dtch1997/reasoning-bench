# Reasoning-Bench

A collection of existing reasoning tasks for LLMs, implemented using [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai).

## Setup

Install the repo:

```bash
git clone https://github.com/dtch1997/reasoning-bench
cd reasoning-bench
pip install -e .
```

## Usage

We use [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) to define tasks, run evaluations, and view results.

First, create a `.env` file containing environment variables [as described here](https://inspect.ai-safety-institute.org.uk/workflow.html#sec-workflow-configuration).

To run a task, use the `inspect` CLI:

```bash
inspect eval reasoning_bench/tasks/musique [KWARGS]
```

To view results, use the `inspect` dashboard:

```bash
inspect view
```

## Tasks

Currently, the following tasks are implemented:

- [MuSiQue](reasoning_bench/tasks/musique/README.md)
- [HotPotQA](reasoning_bench/tasks/hotpot_qa/README.md)
- [ProcBench](reasoning_bench/tasks/procbench/README.md)

Additional tasks can be added by pattern-matching the existing implementations.
PRs are welcome!
