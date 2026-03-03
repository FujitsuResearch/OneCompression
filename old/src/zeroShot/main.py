import json
import logging

from . import evaluator
from . import tasks
from .utils import pattern_match


def get_result(args, model):

    if args.tasks is None:
        raise ValueError("Please specify a task to run")
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=model,
        args=args,
        tasks_list=task_names,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    return results

    # if args.output_path:
    #     with open(args.output_path, "w") as f:
    #         f.write(dumped)

    # print(f"{args.model}"
    #       f"num_fewshot: {args.num_fewshot},"
    #       f" batch_size: {args.batch_size}")
    # if args.table_results:
    #     print(evaluator.make_table(results))
    # else:
    #     from pprint import pprint
    #     pprint(results)
