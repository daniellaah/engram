import inspect
import json
import types
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import TypeAdapter, ValidationError

from engram.tools.result import ToolResult

type ToolFunction = Callable[..., object] | Callable[..., Awaitable[object]]


class ToolValidationError(ValueError):
    pass


@dataclass(slots=True)
class Tool:
    """A callable plus the JSON schema a model needs to invoke it."""

    name: str
    description: str
    function: ToolFunction
    parameters: dict[str, Any] | None = None
    _signature: inspect.Signature = field(init=False, repr=False)
    _hints: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name cannot be empty.")
        if not self.description.strip():
            raise ValueError("Tool description cannot be empty.")
        self._signature = inspect.signature(self.function)
        try:
            self._hints = get_type_hints(self.function)
        except (NameError, TypeError):
            self._hints = {}
        if self.parameters is None:
            self.parameters = self._build_schema()

    @classmethod
    def from_callable(
        cls,
        function: ToolFunction,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> "Tool":
        doc = inspect.getdoc(function) or ""
        summary = doc.splitlines()[0].strip() if doc else ""
        return cls(
            name=name or function.__name__,
            description=description or summary or f"Run {function.__name__}.",
            function=function,
        )

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {"type": "object", "properties": {}},
            "strict": False,
        }

    def validate(self, arguments: Mapping[str, Any] | str) -> dict[str, Any]:
        values = self._normalize_arguments(arguments)
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in self._signature.parameters.values()
        )
        unexpected = set(values).difference(self._signature.parameters)
        if unexpected and not accepts_keywords:
            names = ", ".join(sorted(unexpected))
            raise ToolValidationError(f"Unexpected argument(s) for {self.name}: {names}")

        validated: dict[str, Any] = {}
        for name, parameter in self._signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name not in values:
                if parameter.default is inspect.Parameter.empty:
                    raise ToolValidationError(f"Missing required argument for {self.name}: {name}")
                continue
            annotation = self._hints.get(name, parameter.annotation)
            if annotation is inspect.Parameter.empty:
                validated[name] = values[name]
                continue
            try:
                validated[name] = TypeAdapter(annotation).validate_python(values[name])
            except ValidationError as error:
                raise ToolValidationError(
                    f"Invalid argument '{name}' for {self.name}: {error}"
                ) from error
        if accepts_keywords:
            validated.update({name: values[name] for name in unexpected})
        return validated

    def invoke(self, arguments: Mapping[str, Any] | str) -> ToolResult:
        result = self.function(**self.validate(arguments))
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise RuntimeError(f"Tool {self.name} is asynchronous; use ainvoke().")
        return self._as_result(result)

    async def ainvoke(self, arguments: Mapping[str, Any] | str) -> ToolResult:
        values = self.validate(arguments)
        if inspect.iscoroutinefunction(self.function):
            result = await self.function(**values)
        else:
            result = self.function(**values)
            if inspect.isawaitable(result):
                result = await result
        return self._as_result(result)

    def _normalize_arguments(self, arguments: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(arguments, str):
            names = list(self._signature.parameters)
            if len(names) != 1:
                raise ToolValidationError(
                    "String input is only valid for single-parameter tools; "
                    f"{self.name} has {len(names)}."
                )
            return {names[0]: arguments}
        return dict(arguments)

    def _build_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, parameter in self._signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = self._hints.get(name, parameter.annotation)
            properties[name] = self._annotation_schema(annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
            else:
                properties[name]["default"] = parameter.default
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @staticmethod
    def _annotation_schema(annotation: Any) -> dict[str, Any]:
        if annotation is inspect.Parameter.empty or annotation is Any:
            return {}
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            values = [member.value for member in annotation]
            return {
                "type": _json_type(type(values[0])) if values else "string",
                "enum": values,
            }
        origin = get_origin(annotation)
        if origin is Literal:
            values = list(get_args(annotation))
            return {"type": _json_type(type(values[0])) if values else "string", "enum": values}
        if origin in (Union, types.UnionType):
            variants = [Tool._annotation_schema(item) for item in get_args(annotation)]
            return {"anyOf": variants}
        try:
            schema = TypeAdapter(annotation).json_schema()
        except (TypeError, ValueError):
            return {}
        schema.pop("title", None)
        return schema

    @staticmethod
    def _as_result(value: object) -> ToolResult:
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, str):
            return ToolResult.success(value)
        try:
            content = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            content = str(value)
        return ToolResult.success(content, data=value)


def tool(
    function: ToolFunction | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[ToolFunction], Tool]:
    """Turn a typed callable into an Engram tool."""

    def decorate(candidate: ToolFunction) -> Tool:
        return Tool.from_callable(candidate, name=name, description=description)

    if function is None:
        return decorate
    return decorate(function)


def _json_type(value_type: type[object]) -> str:
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }.get(value_type, "string")
