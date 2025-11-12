import multiprocessing
from multiprocessing import Queue
from typing import Any, Callable, Dict, Tuple


def _iteration_wrapper(
    queue: Queue,
    target: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """
    Internal wrapper run inside the forked process.

    It executes the target callable and communicates the outcome back to the
    parent process through a multiprocessing.Queue.  The result is always a
    dictionary with at least a ``status`` field of either ``"ok"`` or
    ``"error"``.
    """
    try:
        result = target(*args, **kwargs)
        queue.put({"status": "ok", "result": result})
    except Exception as exc:  # pragma: no cover - defensive, should rarely happen
        import traceback

        queue.put(
            {
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_iteration_in_subprocess(
    target: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute ``target`` in a forked subprocess and wait for it to finish.

    Args:
        target: Callable to execute inside the subprocess.
        *args: Positional arguments for ``target``.
        **kwargs: Keyword arguments for ``target``.

    Returns:
        Dictionary containing:
            - ``status``: ``"ok"`` when the call succeeded, otherwise ``"error"``.
            - ``result``: Value returned by ``target`` (when successful).
            - ``error`` / ``traceback``: Present when an exception occurred.
            - ``exitcode``: Exit code of the subprocess.
    """
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_iteration_wrapper,
        args=(queue, target, args, kwargs),
    )
    process.start()
    process.join()

    payload: Dict[str, Any]
    if not queue.empty():
        payload = queue.get()
    else:
        # No payload was returned – treat non-zero exit codes as errors.
        payload = {"status": "ok", "result": None}

    payload["exitcode"] = process.exitcode
    if payload["status"] == "ok" and process.exitcode not in (0, None):
        payload = {
            "status": "error",
            "error": f"Child process exited with code {process.exitcode}",
            "result": payload.get("result"),
            "exitcode": process.exitcode,
        }
    return payload


