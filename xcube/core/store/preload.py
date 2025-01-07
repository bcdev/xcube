# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from abc import abstractmethod, ABC
from asyncio import CancelledError
from concurrent.futures import Executor, Future
from concurrent.futures.thread import ThreadPoolExecutor
from enum import Enum
import threading
from typing import Any, Callable

import tabulate

from xcube.util.assertions import assert_given, assert_instance


class PreloadStatus(Enum):
    """Preload process status."""

    waiting = "waiting"
    started = "started"
    stopped = "stopped"
    cancelled = "cancelled"
    failed = "failed"

    def __str__(self):
        return self.name.upper()

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


class PreloadState:
    """Preload state."""

    def __init__(
        self,
        data_id: str,
        status: PreloadStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        exception: BaseException | None = None,
    ):
        assert_given(data_id, name="data_id")
        self.data_id = data_id
        self.status = status
        self.progress = progress
        self.message = message
        self.exception = exception

    def update(self, event: "PreloadState"):
        """Update this state with the given partial state.

        Args:
            event: the partial state.
        """
        assert_instance(event, PreloadState, name="event")
        if self.data_id == event.data_id:
            if event.status is not None:
                self.status = event.status
            if event.progress is not None:
                self.progress = event.progress
            if event.message is not None:
                self.message = event.message
            if event.exception is not None:
                self.exception = event.exception

    def __str__(self):
        return ", ".join(f"{k}={v}" for k, v in _to_dict(self).items())

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in _to_dict(self).items())
        return f"{self.__class__.__name__}({args})"


class PreloadHandle(ABC):
    """A handle for a preload job.

    Instances of this class are returned by the
    ``DataStore.preload_data()`` method.
    """

    @abstractmethod
    def get_state(self, data_id: str) -> PreloadState:
        """Get the preload state for the given *data_id*.

        Args:
            data_id: The data identifier.
        Returns:
            The preload state.
        """

    @property
    @abstractmethod
    def cancelled(self) -> bool:
        """True` if the preload job has been cancelled."""

    @abstractmethod
    def cancel(self):
        """Cancel the preload job."""

    def close(self):
        """Close the preload job.

        Should be called if the preloaded data is no longer needed.

        This method usually cleans the cache associated with
        this preload object.

        The default implementation does nothing.
        """

    @abstractmethod
    def show(self) -> Any:
        """Show the current state of the preload job.

        This method is useful for non-blocking / asynchronous preload
        implementations, especially in a Jupyter Notebook context.
        In this case an implementation might want to display a widget
        suitable for in-place updating, e.g., an ``ipywidgets`` widget.
        """

    def notify(self, event: PreloadState):
        """Notify about a preload state change.

        Updates the preload job using the given partial state
        *event* that refers to state changes of a running
        preload task.

        Args:
            event: A partial state
        """

    def __enter__(self) -> "PreloadHandle":
        """Enter the context.

        Returns:
            This object.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context. Calls ``close()``."""
        self.close()


class NullPreloadHandle(PreloadHandle):
    """Null-pattern implementation of a ``PreloadHandle``."""

    def get_state(self, data_id: str) -> PreloadState:
        return PreloadState(data_id)

    @property
    def cancelled(self) -> bool:
        return False

    def cancel(self):
        pass

    def show(self) -> Any:
        return None


class ExecutorPreloadHandle(PreloadHandle):
    """An implementation of a ``PreloadHandle`` that uses a
    ``concurrent.futures.Executor`` for concurrently preloading data.

    You can either use this class

    - directly, by passing your `preload_data` function to the constructor
    - or by deriving your own class from it, then overriding the
      default implementation of its ``preload_data`` method.

    By default, the preload state updates are displayed. How, depends on the
    context: If executed in Jupyter notebooks (``IPython.display`` is available)
    the handle makes use of ``ipywidgets``, if installed, otherwise it outputs HTML.
    If not executed in a notebook it dumps process information to ``stdout``.
    If you don't want any output of preload state updates or results,
    use the ``silent`` flag.

    Args:
        data_ids: The identifiers of the data resources to be preloaded.
        preload_data: The function that preloads individual datasets.
            Optional. If not provided, you should override the
            ``preload_data`` method to implement the preloading.
        executor: Optional executor such as an instance of
            ``concurrent.futures.thread.ThreadPoolExecutor`` or
            ``concurrent.futures.process.ProcessPoolExecutor``.
            If not provided, a ``ThreadPoolExecutor`` with default settings
            will be used.
        blocking: `True` (the default) if the constructor should wait for
            all preload task to finish before the calling thread
            continues execution.
        silent: ``True`` if you don't want any preload state output.
            Defaults to ``False``.
    """

    def __init__(
        self,
        data_ids: tuple[str, ...],
        preload_data: Callable[[PreloadHandle, str], None] | None = None,
        executor: Executor | None = None,
        blocking: bool = True,
        silent: bool = False,
    ):
        self._preload_data = preload_data
        self._executor = executor or ThreadPoolExecutor()
        self._blocking = blocking

        self._states = {data_id: PreloadState(data_id=data_id) for data_id in data_ids}
        self._cancel_event = threading.Event()
        self._display = PreloadDisplay.create(list(self._states.values()), silent)
        self._silent = silent
        self._lock = threading.Lock()
        self._futures: dict[str, Future[str]] = {}
        for data_id in data_ids:
            future: Future[str] = self._executor.submit(self._run_preload_data, data_id)
            future.add_done_callback(self._handle_preload_data_done)
            self._futures[data_id] = future

        if blocking:
            if not self._silent:
                self._display.show()
            self._executor.shutdown(wait=True)

    def get_state(self, data_id: str) -> PreloadState:
        return self._states[data_id]

    @property
    def cancelled(self) -> bool:
        """Return true if and only if the internal flag is true."""
        return self._cancel_event.is_set()

    def cancel(self):
        self._cancel_event.set()
        for future in self._futures.values():
            future.cancel()
        self._executor.shutdown(wait=False)

    def notify(self, event: PreloadState):
        state = self._states[event.data_id]
        if (
            event.status is not None
            and event.status != state.status
            and state.status
            in (PreloadStatus.stopped, PreloadStatus.cancelled, PreloadStatus.failed)
        ):
            # Status cannot be changed
            return
        with self._lock:
            state.update(event)
            if not self._silent:
                self._display.update()

    def preload_data(self, data_id: str):
        """Preload the data resource given by *data_id*.

        Concurrently executes the *preload_data* passed to the constructor,
        if any. Otherwise, it does nothing.

        Can be overridden by clients to implement the actual preload operation.

        Args:
           data_id: The data identifier of the data resource to be preloaded.
        """
        if self._preload_data is not None:
            self._preload_data(self, data_id)

    def _run_preload_data(self, data_id: str) -> str:
        self.notify(PreloadState(data_id, status=PreloadStatus.started))
        self.preload_data(data_id)
        return data_id

    def _handle_preload_data_done(self, f: Future[str]):
        # Find the data_id that belongs to given feature f
        data_id: str | None = None
        for data_id, future in self._futures.items():
            if f is future:
                break
        assert data_id is not None
        try:
            _value = f.result()
            # No exceptions, notify everything seems ok
            self.notify(PreloadState(data_id, status=PreloadStatus.stopped))
        except CancelledError as e:
            # Raised if future has been cancelled
            # while executing `_run_preload_data`
            self.notify(
                PreloadState(data_id, status=PreloadStatus.cancelled, exception=e)
            )
        except Exception as e:
            # Raised if any exception occurred
            # while executing `_run_preload_data`
            self.notify(PreloadState(data_id, status=PreloadStatus.failed, exception=e))

    def show(self) -> Any:
        return self._display.show()

    def _repr_html_(self):
        return self._display.to_html()

    def __str__(self):
        return self._display.to_text()

    def __repr__(self):
        return self._display.to_text()

    def __enter__(self) -> "PreloadHandle":
        """Enter the context.

        Does nothing but returning this handle.
        Only useful when in blocking mode.

        Returns:
            This object.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context.

        Calls ``close()`` if in blocking mode.
        Otherwise, does nothing.
        """
        if self._blocking:
            self.close()


class PreloadDisplay(ABC):

    @classmethod
    def create(cls, states: list[PreloadState], silent: bool | None = None) -> "PreloadDisplay":
        try:
            # noinspection PyUnresolvedReferences
            from IPython.display import display

            if display is not None:
                try:
                    return IPyWidgetsPreloadDisplay(states)
                except ImportError:
                    return IPyPreloadDisplay(states)
        except ImportError:
            pass
        return PreloadDisplay(states)

    def __init__(self, states: list[PreloadState]):
        self.states = states

    def _repr_html_(self) -> str:
        return self.to_html()

    def to_text(self) -> str:
        return self.tabulate(table_format="simple")

    def to_html(self) -> str:
        return self.tabulate(table_format="html")

    def tabulate(self, table_format: str = "simple") -> str:
        """Generate HTML table from job list."""
        rows = [
            [
                state.data_id,
                f"{state.status}" if state.status is not None else "-",
                (
                    f"{round(state.progress * 100)}%"
                    if state.progress is not None
                    else "-"
                ),
                state.message or "-",
                state.exception or "-",
            ]
            for state in self.states
        ]

        return tabulate.tabulate(
            rows,
            headers=["Data ID", "Status", "Progress", "Message", "Exception"],
            tablefmt=table_format,
        )

    def show(self):
        """Display the widget container."""
        print(self.to_text())

    def update(self):
        """Update the display."""
        print(self.to_text())

    def log(self, message: str):
        """Log a message to the output widget."""
        print(message)


class IPyPreloadDisplay(PreloadDisplay):
    def __init__(self, states: list[PreloadState]):
        super().__init__(states)
        from IPython import display

        self._ipy_display = display

    def show(self):
        """Display the widget container."""
        self._ipy_display.display(self.to_html())

    def update(self):
        """Update the display."""
        self._ipy_display.clear_output(wait=True)
        self._ipy_display.display(self.to_html())

    def log(self, message: str):
        """Log a message to the output widget."""
        self._ipy_display.display(message)


class IPyWidgetsPreloadDisplay(IPyPreloadDisplay):

    def __init__(self, states: list[PreloadState]):
        super().__init__(states)
        import ipywidgets

        self._state_table = ipywidgets.HTML(self.to_html())
        self._output = ipywidgets.Output()  # not used yet
        self._container = ipywidgets.VBox([self._state_table, self._output])

    def show(self):
        """Display the widget container."""
        self._ipy_display.display(self._container)

    def update(self):
        """Update the display."""
        self._state_table.value = self.to_html()

    def log(self, message: str):
        """Log a message to the output widget."""
        with self._output:
            print(message)


def _to_dict(obj: object):
    return {
        k: v
        for k, v in obj.__dict__.items()
        if isinstance(k, str) and not k.startswith("_") and v is not None
    }
