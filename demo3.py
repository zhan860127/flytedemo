import typing
from flytekit import task, workflow, Resources, ImageSpec ,conditional


@task
def add_5(a:int)->int:
    return a+5

@task
def simple_wf()->int:
    return 10

@workflow
def my_wf_example(a: int) -> int:
    """example

    Workflows can have inputs and return outputs of simple or complex types.

    :param a: input a
    :return: outputs
    """

    x = add_5(a=a)

    # You can use outputs of a previous task as inputs to other nodes.
    z = add_5(a=x)

    # You can call other workflows from within this workflow
    d = simple_wf()

    # You can add conditions that can run on primitive types and execute different branches
    e = conditional("bool").if_(a == 5).then(add_5(a=d)).else_().then(add_5(a=z))

    # Outputs of the workflow have to be outputs returned by prior nodes.
    # No outputs and single or multiple outputs are supported
    return e


def gt_100(x: int) -> bool:
    return x > 100


@eager
async def eager_workflow_with_conditionals(x: int) -> int:
    out = await add_one(x=x)

    if out < 0:
        return -1
    elif await gt_100(x=out):
        return 100
    else:
        out = await double(x=out)

    assert out >= -1
    return out
