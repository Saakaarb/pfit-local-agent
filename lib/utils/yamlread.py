from pathlib import Path

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
        experiment = experiments[0]
        self.filename_data = experiment.get("data_file")
        for index, column in enumerate(experiment.get("columns") or []):
            name = column.get("name")
            if name:
                self.data_column_names.append(str(name))
                self.data_column_index.append(index)
                self.data_column_observes.append(
                    str(column.get("observes") or name)
                )

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
