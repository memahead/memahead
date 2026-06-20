"""Plan modeling for forward-looking context compression.

A :class:`Plan` is the spine of memahead: it describes the multi-step workflow
an agent is executing. The compressor uses the steps that come *after* the
current one to decide which pieces of context are worth keeping. This is the
core idea behind the "what's ahead" name — retention is driven by the future,
not the past.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence

__all__ = ["Step", "Plan", "PlanGraph"]


@dataclass(frozen=True)
class Step:
    """A single step in an agent workflow.

    Attributes:
        name: A short, unique identifier for the step (e.g. ``"research"``).
        description: A natural-language description of what the step does.
            This text is what the retention scorer embeds, so make it
            descriptive of the information the step will need.
    """

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Step.name must be a non-empty string")

    def as_text(self) -> str:
        """Return the text used for embedding/matching this step."""

        if self.description:
            return f"{self.name}: {self.description}"
        return self.name


class Plan:
    """An ordered sequence of :class:`Step` objects.

    A plan behaves like an immutable ordered collection: it is iterable,
    indexable, and sized. Step names must be unique within a plan.

    Example:
        >>> plan = Plan([
        ...     Step("research", "Gather raw facts"),
        ...     Step("draft", "Write a first draft"),
        ... ])
        >>> [s.name for s in plan.remaining_from("research")]
        ['draft']
    """

    def __init__(self, steps: Iterable[Step | tuple[str, str] | str]):
        normalized: List[Step] = []
        for step in steps:
            normalized.append(self._coerce_step(step))
        if not normalized:
            raise ValueError("Plan must contain at least one step")

        seen: Dict[str, int] = {}
        for index, step in enumerate(normalized):
            if step.name in seen:
                raise ValueError(f"duplicate step name in plan: {step.name!r}")
            seen[step.name] = index

        self._steps: List[Step] = normalized
        self._index_by_name: Dict[str, int] = seen

    @staticmethod
    def _coerce_step(step: Step | tuple[str, str] | str) -> Step:
        if isinstance(step, Step):
            return step
        if isinstance(step, str):
            return Step(step)
        if isinstance(step, tuple) and len(step) == 2:
            return Step(step[0], step[1])
        raise TypeError(
            "each step must be a Step, a (name, description) tuple, or a str; "
            f"got {type(step)!r}"
        )

    # -- collection protocol ------------------------------------------------

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, index: int) -> Step:
        return self._steps[index]

    def __repr__(self) -> str:
        names = ", ".join(repr(s.name) for s in self._steps)
        return f"Plan([{names}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plan):
            return NotImplemented
        return self._steps == other._steps

    # -- lookups ------------------------------------------------------------

    @property
    def steps(self) -> List[Step]:
        """Return a copy of the steps list."""

        return list(self._steps)

    def names(self) -> List[str]:
        """Return the step names in order."""

        return [s.name for s in self._steps]

    def index_of(self, step_name: str) -> int:
        """Return the position of ``step_name``.

        Raises:
            KeyError: if no step with that name exists.
        """

        try:
            return self._index_by_name[step_name]
        except KeyError:
            raise KeyError(
                f"step {step_name!r} not in plan; known steps: {self.names()}"
            ) from None

    def get(self, step_name: str) -> Step:
        """Return the :class:`Step` named ``step_name``."""

        return self._steps[self.index_of(step_name)]

    def __contains__(self, step_name: object) -> bool:
        return step_name in self._index_by_name

    # -- the core forward-looking query -------------------------------------

    def remaining_from(self, step_name: str, *, inclusive: bool = False) -> List[Step]:
        """Return all steps that come *after* ``step_name``.

        This is the forward-looking horizon that drives retention scoring:
        context is kept based on how useful it is to these future steps.

        Args:
            step_name: The current step.
            inclusive: If ``True``, include the current step itself in the
                returned list. Defaults to ``False``.

        Returns:
            A list of steps following the current one (possibly empty if the
            current step is the last one).
        """

        idx = self.index_of(step_name)
        start = idx if inclusive else idx + 1
        return self._steps[start:]

    def completed_before(self, step_name: str) -> List[Step]:
        """Return all steps that come *before* ``step_name``."""

        idx = self.index_of(step_name)
        return self._steps[:idx]


class PlanGraph:
    """A non-linear plan expressed as a dependency DAG.

    Where :class:`Plan` models a strictly ordered workflow, :class:`PlanGraph`
    supports branching/merging workflows: each step declares the steps it
    depends on. ``remaining_from`` then returns the *downstream* steps that
    still depend (transitively) on the current step — i.e. the true forward
    horizon for retention scoring in a branching plan.

    Example:
        >>> g = PlanGraph()
        >>> g.add_step(Step("research", "gather facts"))
        >>> g.add_step(Step("draft", "write draft"), depends_on=["research"])
        >>> g.add_step(Step("cite", "add citations"), depends_on=["research"])
        >>> sorted(s.name for s in g.remaining_from("research"))
        ['cite', 'draft']
    """

    def __init__(self) -> None:
        self._steps: Dict[str, Step] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._dependents: Dict[str, List[str]] = {}

    def add_step(
        self,
        step: Step,
        depends_on: Sequence[str] | None = None,
    ) -> "PlanGraph":
        """Add a step with optional upstream dependencies.

        Returns ``self`` to allow chaining.
        """

        if step.name in self._steps:
            raise ValueError(f"duplicate step name in graph: {step.name!r}")
        deps = list(depends_on or [])
        for dep in deps:
            if dep not in self._steps:
                raise KeyError(
                    f"dependency {dep!r} for step {step.name!r} not yet added"
                )
        self._steps[step.name] = step
        self._dependencies[step.name] = deps
        self._dependents.setdefault(step.name, [])
        for dep in deps:
            self._dependents[dep].append(step.name)
        return self

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, step_name: object) -> bool:
        return step_name in self._steps

    def get(self, step_name: str) -> Step:
        try:
            return self._steps[step_name]
        except KeyError:
            raise KeyError(
                f"step {step_name!r} not in graph; known steps: "
                f"{list(self._steps)}"
            ) from None

    def dependencies_of(self, step_name: str) -> List[Step]:
        """Return the direct upstream steps of ``step_name``."""

        self.get(step_name)  # validate existence
        return [self._steps[d] for d in self._dependencies[step_name]]

    def remaining_from(self, step_name: str, *, inclusive: bool = False) -> List[Step]:
        """Return all steps transitively downstream of ``step_name``.

        Uses a breadth-first traversal over the dependents graph. The result
        order is deterministic (BFS order from the current step).
        """

        self.get(step_name)  # validate existence
        visited: set[str] = set()
        ordered: List[str] = []
        queue: List[str] = list(self._dependents.get(step_name, []))
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ordered.append(current)
            queue.extend(self._dependents.get(current, []))

        result_names = ([step_name] if inclusive else []) + ordered
        return [self._steps[name] for name in result_names]

    def topological_order(self) -> List[Step]:
        """Return steps in a dependency-respecting topological order."""

        indegree = {name: len(deps) for name, deps in self._dependencies.items()}
        ready = [name for name, deg in indegree.items() if deg == 0]
        ordered: List[str] = []
        while ready:
            ready.sort()  # deterministic ordering
            name = ready.pop(0)
            ordered.append(name)
            for dependent in self._dependents.get(name, []):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if len(ordered) != len(self._steps):
            raise ValueError("PlanGraph contains a cycle")
        return [self._steps[name] for name in ordered]
