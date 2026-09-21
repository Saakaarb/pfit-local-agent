from local_agent.agent.session_spec import SessionSpec


def render_user_model_skeleton(spec: SessionSpec) -> str:
    trainable_lines = [
        f"        {parameter.name} = trainable_parameters['{parameter.name}']"
        for parameter in spec.trainable_parameters
    ]
    fixed_lines = [
        f"        {parameter.name} = fixed_parameters['{parameter.name}']"
        for parameter in spec.fixed_parameters
    ]
    state_lines = [
        f"        {variable.name} = y[{index}]"
        for index, variable in enumerate(spec.integrated_variables)
    ]
    derivative_names = [f"d{variable.name}dt" for variable in spec.integrated_variables]
    derivative_array = ", ".join(derivative_names)
    model_bindings = "\n".join(trainable_lines + fixed_lines + [""] + state_lines)
    loss_bindings = "\n".join(trainable_lines + fixed_lines)
    write_bindings = "\n".join(trainable_lines + fixed_lines)
    writeout_columns = 1 + 2 * len(spec.integrated_variables)

    return f'''import numpy as np

# Key:
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.yaml file
# N_col: number of columns in dataset, including time

def user_defined_system(t: float, y: np.ndarray, trainable_parameters: dict, fixed_parameters: dict, dataset: np.ndarray, t_eval: np.ndarray):
        # Arguments:
        # t: time
        # y: state vector of shape [Ny]
        # trainable_parameters: names and values from user_input.yaml
        # fixed_parameters: names and values from user_input.yaml
        # dataset: data array of shape [Nts, N_col-1]
        # t_eval: evaluation times of shape [Nts]

        # YAML-derived bindings. Do not reorder.
{model_bindings}

        # Define each derivative below.
{_render_unassigned_derivatives(derivative_names)}

        derivatives = np.array([{derivative_array}])
        return derivatives

def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):
        # Arguments:
        # solution_time: array of shape [Nts]
        # solution: simulated state array of shape [Nts, Ny]
        # dataset: data array of shape [Nts, N_col-1]
        # trainable_parameters: names and values from user_input.yaml
        # fixed_parameters: names and values from user_input.yaml

        # YAML-derived bindings. Do not reorder.
{loss_bindings}

        loss = 0.0
        # Define the scalar loss below.
        return loss

def writeout_description(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):
        # Arguments:
        # solution_time: array of shape [Nts]
        # solution: simulated state array of shape [Nts, Ny]
        # dataset: data array of shape [Nts, N_col-1]
        # trainable_parameters: names and values from user_input.yaml
        # fixed_parameters: names and values from user_input.yaml

        # YAML-derived bindings. Do not reorder.
{write_bindings}

        writeout_array = np.zeros([solution_time.shape[0], {writeout_columns}])
        # Define writeout columns below.
        return writeout_array
'''


def _render_unassigned_derivatives(derivative_names: list[str]) -> str:
    return "\n".join(f"        {name} = 0.0" for name in derivative_names)
