from dataclasses import dataclass
from pathlib import Path

from local_agent.agent.validators import parse_input_yaml


@dataclass(frozen=True)
class TrainableParameter:
    name: str
    min_value: float
    max_value: float
    logscale: bool


@dataclass(frozen=True)
class FixedParameter:
    name: str
    value: float


@dataclass(frozen=True)
class IntegratedVariable:
    name: str
    initial_value: float


@dataclass(frozen=True)
class SessionSpec:
    filename_data: str
    trainable_parameters: tuple[TrainableParameter, ...]
    fixed_parameters: tuple[FixedParameter, ...]
    integrated_variables: tuple[IntegratedVariable, ...]
    data_column_names: tuple[str, ...]
    data_column_observes: tuple[str, ...]
    observable_names: tuple[str, ...]
    stepsize_rtol: tuple[float, ...]
    stepsize_atol: tuple[float, ...]
    init_timestep: float
    max_steps: int

    def to_prompt_text(self) -> str:
        lines = [
            f"Dataset file: {self.filename_data}",
            "",
            "Trainable parameter order:",
        ]
        for index, parameter in enumerate(self.trainable_parameters):
            lines.append(
                f"{index}: {parameter.name} "
                f"(min={parameter.min_value}, max={parameter.max_value}, "
                f"logscale={parameter.logscale})"
            )

        lines.append("")
        lines.append("Fixed parameters:")
        if self.fixed_parameters:
            for parameter in self.fixed_parameters:
                lines.append(f"- {parameter.name} = {parameter.value}")
        else:
            lines.append("- none")

        lines.append("")
        lines.append("Integrated variable order:")
        for index, variable in enumerate(self.integrated_variables):
            lines.append(f"{index}: {variable.name} (initial={variable.initial_value})")

        lines.append("")
        lines.append("Dataset columns:")
        for index, name in enumerate(self.data_column_names):
            observes = self.data_column_observes[index]
            if observes == name:
                lines.append(f"{index}: {name}")
            else:
                lines.append(f"{index}: {name} observes {observes}")
        if self.data_column_names:
            lines.append("")
            lines.append("Runtime dataset columns after removing time:")
            for index, name in enumerate(self.data_column_names[1:]):
                observes = self.data_column_observes[index + 1]
                if observes == name:
                    lines.append(f"dataset[:, {index}]: {name}")
                else:
                    lines.append(f"dataset[:, {index}]: {name} observes {observes}")

        lines.append("")
        lines.append("Observable names:")
        if self.observable_names:
            for name in self.observable_names:
                lines.append(f"- {name}")
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                f"Solver rtol: {list(self.stepsize_rtol)}",
                f"Solver atol: {list(self.stepsize_atol)}",
                f"Initial timestep: {self.init_timestep}",
                f"Max steps: {self.max_steps}",
            ]
        )
        return "\n".join(lines)


def load_session_spec(input_yaml: Path) -> SessionSpec:
    reader = parse_input_yaml(input_yaml)
    trainable_parameters = tuple(
        TrainableParameter(
            name=reader.trainable_parameter_names[index],
            min_value=reader.min_axis_values[index],
            max_value=reader.max_axis_values[index],
            logscale=bool(reader.axis_logscale[index]),
        )
        for index in range(reader.n_search_axes)
    )
    fixed_parameters = tuple(
        FixedParameter(name=name, value=value)
        for name, value in zip(reader.fixed_parameter_names, reader.fixed_parameter_values)
    )
    integrated_variables = tuple(
        IntegratedVariable(name=name, initial_value=value)
        for name, value in zip(
            reader.integrated_variable_names,
            reader.integrated_variable_init_values,
        )
    )
    return SessionSpec(
        filename_data=reader.filename_data,
        trainable_parameters=trainable_parameters,
        fixed_parameters=fixed_parameters,
        integrated_variables=integrated_variables,
        data_column_names=tuple(reader.data_column_names),
        data_column_observes=tuple(reader.data_column_observes),
        observable_names=tuple(reader.observable_names),
        stepsize_rtol=tuple(reader.stepsize_rtol),
        stepsize_atol=tuple(reader.stepsize_atol),
        init_timestep=reader.init_timestep,
        max_steps=reader.max_steps,
    )
