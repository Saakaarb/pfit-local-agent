from pathlib import Path
import math

import yaml


class YAMLReader:
    def __init__(self):
        self.user_input_dirname = "inputs"
        self.generated_dirname = "generated"
        self.output_dirname = "outputs"

        self.n_search_axes = None
        self.min_axis_values = []
        self.max_axis_values = []
        self.axis_logscale = []
        self.trainable_parameter_names = []
        self.fixed_parameter_names = []
        self.fixed_parameter_values = []
        self.integrated_variable_names = []
        self.integrated_variable_init_values = []

        self.n_particles = None
        self.n_iters_pop = None
        self.processors = None
        self.pso_stepsize_rtol = None
        self.pso_stepsize_atol = None
        self.population_stepsize_rtol = None
        self.population_stepsize_atol = None
        self.algorithm = "DE"
        self.random_seed = None

        self.n_iters_grad = None
        self.stepsize_rtol = None
        self.stepsize_atol = None
        self.init_timestep = None
        self.max_steps = None
        self.integrator = "Tsit5"
        self.init_time = None
        self.init_value_lr = None
        self.end_value_lr = None
        self.transition_steps_lr = None
        self.decay_rate_lr = None
        self.error_loss = 1.0e30

        self.experiments = []
        self.filename_data = None
        self.data_column_index = []
        self.data_column_names = []
        self.data_column_observes = []
        self.observable_names = []
        self.write_results = False
        self.output_dir = None

    @classmethod
    def from_file(cls, path: Path) -> "YAMLReader":
        reader = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("YAML input must be a mapping")
        reader.read_yaml(data)
        return reader

    def read_yaml(self, data: dict) -> None:
        experiments = data.get("experiments") or []
        if not experiments:
            raise ValueError("YAML input must define at least one experiment")
        if not isinstance(experiments, list) or any(not isinstance(e, dict) for e in experiments):
            raise ValueError("experiments must be a list of mappings")
        paths = data.get("paths") or {}
        self.user_input_dirname = paths.get("user_input_dir", self.user_input_dirname)
        self.generated_dirname = paths.get("generated_dir", self.generated_dirname)
        self.output_dirname = paths.get("output_dir", self.output_dirname)

        model = data.get("model") or {}
        trainable = model.get("trainable_parameters") or []
        self.n_search_axes = len(trainable)
        for parameter in trainable:
            self.trainable_parameter_names.append(str(parameter["name"]))
            self.min_axis_values.append(float(parameter["min_val"]))
            self.max_axis_values.append(float(parameter["max_val"]))
            self.axis_logscale.append(bool(parameter.get("logscale", False)))

        for parameter in model.get("fixed_parameters") or []:
            self.fixed_parameter_names.append(str(parameter["name"]))
            self.fixed_parameter_values.append(float(parameter["value"]))

        for variable in model.get("integrated_variables") or []:
            self.integrated_variable_names.append(str(variable["name"]))
            self.integrated_variable_init_values.append(float(variable["init_val"]))
        for observable in model.get("observables") or []:
            self.observable_names.append(str(observable["name"]))

        common_columns = None
        for index, entry in enumerate(experiments):
            where = f"experiment {index + 1}"
            if not isinstance(entry, dict):
                raise ValueError(f"{where} must be a mapping")
            unknown = set(entry) - {"data_file", "columns", "initial_conditions"}
            if unknown:
                raise ValueError(f"{where}: unsupported keys {sorted(unknown)}")
            filename = entry.get("data_file")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(f"{where}: missing data_file")
            columns = entry.get("columns") or []
            if not isinstance(columns, list) or len(columns) < 2 or any(not isinstance(c, dict) or not isinstance(c.get("name"), str) or not c["name"].strip() for c in columns):
                raise ValueError(f"{where}: columns must declare time and every measurement/auxiliary column")
            names = [c["name"] for c in columns]
            if len(set(names)) != len(names):
                raise ValueError(f"{where}: column names must be unique")
            measurements = {c["name"] for c in columns[1:] if not c.get("uncertainty_of")}
            for column in columns:
                target = column.get("uncertainty_of")
                if target and (target not in measurements or target == column["name"]):
                    raise ValueError(f"{where}: uncertainty_of must name a measurement column")
            schema = [(c["name"], c.get("observes") or c["name"], c.get("uncertainty_of"), c.get("units")) for c in columns]
            if common_columns is not None and schema != common_columns:
                raise ValueError(f"{where}: all experiments must have the same ordered column meanings and units")
            common_columns = schema
            overrides = entry.get("initial_conditions", {})
            if overrides is None:
                overrides = {}
            if not isinstance(overrides, dict):
                raise ValueError(f"{where}: initial_conditions must be a mapping")
            overrides = dict(overrides)
            for name, value in overrides.items():
                if name not in self.integrated_variable_names:
                    raise ValueError(f"{where}: initial condition {name!r} is not an integrated variable")
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{where}: initial condition {name!r} must be finite") from exc
                if isinstance(value, bool) or not math.isfinite(number):
                    raise ValueError(f"{where}: initial condition {name!r} must be finite")
                overrides[name] = number
            self.experiments.append({"filename": filename, "ic_overrides": overrides, "columns": columns})

        # Positional column layout is shared and validated across every record.
        # These accessors retain the old single-record SessionSpec interface.
        self.filename_data = self.experiments[0]["filename"]
        self.data_column_names = [c["name"] for c in self.experiments[0]["columns"]]
        self.data_column_index = list(range(len(self.data_column_names)))
        self.data_column_observes = [c.get("observes") or c["name"] for c in self.experiments[0]["columns"]]

        population = data.get("population_opt") or {}
        self.n_particles = int(population.get("population_size", population.get("num_particles", 0)) or 0)
        self.n_iters_pop = int(population.get("num_iters", 0) or 0)
        self.processors = int(population.get("processors", 1) or 1)
        self.algorithm = str(population.get("algorithm", self.algorithm)).upper()
        if population.get("random_seed") is not None:
            self.random_seed = int(population["random_seed"])
        self.population_stepsize_rtol = _as_float_list(population.get("stepsize_rtol"))
        self.population_stepsize_atol = _as_float_list(population.get("stepsize_atol"))
        self.pso_stepsize_rtol = self.population_stepsize_rtol
        self.pso_stepsize_atol = self.population_stepsize_atol

        gradient = data.get("gradient_opt") or {}
        self.n_iters_grad = int(gradient.get("num_iters", 0) or 0)
        self.stepsize_rtol = _as_float_list(gradient.get("stepsize_rtol"))
        self.stepsize_atol = _as_float_list(gradient.get("stepsize_atol"))
        self.init_timestep = float(gradient.get("initial_timestep", 1e-6))
        if gradient.get("initial_time") is not None:
            self.init_time = float(gradient["initial_time"])
        self.max_steps = int(gradient.get("max_steps", 10000))
        self.integrator = str(gradient.get("integrator", self.integrator))
        self.init_value_lr = float(gradient.get("init_value_lr", 1e-4))
        self.end_value_lr = float(gradient.get("end_value_lr", 1e-5))
        self.transition_steps_lr = float(gradient.get("transition_steps_lr", 2000))
        self.decay_rate_lr = float(gradient.get("decay_rate_lr", 0.9))
        if gradient.get("error_loss") is not None:
            self.error_loss = float(gradient["error_loss"])

        output = data.get("output") or {}
        self.write_results = bool(output.get("write_results", False))

    def get_y0(self, experiment_idx: int) -> list[float]:
        values = dict(zip(self.integrated_variable_names, self.integrated_variable_init_values))
        values.update(self.experiments[experiment_idx]["ic_overrides"])
        return [values[name] for name in self.integrated_variable_names]

    def check_name_uniqueness(self):
        combined = (
            self.trainable_parameter_names
            + self.fixed_parameter_names
            + self.integrated_variable_names
        )
        if len(combined) > len(set(combined)):
            raise ValueError(
                "The provided names of every simulation element "
                "(trainable parameters, fixed parameters and integrated variable name) "
                "is not unique. Please provide unique names for each."
            )


def _as_float_list(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]
