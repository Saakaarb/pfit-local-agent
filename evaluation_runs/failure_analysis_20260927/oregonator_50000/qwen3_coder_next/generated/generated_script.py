import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):
    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0**lin_unscaled, lin_unscaled)
    return unscaled

def _compute_loss_problem_jax(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    mask_X = jnp.isfinite(dataset[:, 0]) & jnp.isfinite(dataset[:, 2]) & (dataset[:, 2] > 0.0)
    mask_Z = jnp.isfinite(dataset[:, 1]) & jnp.isfinite(dataset[:, 3]) & (dataset[:, 3] > 0.0)
    resid_X = jnp.where(mask_X, (solution[:, 0] - dataset[:, 0]) / (dataset[:, 2] + 1e-12), 0.0)
    resid_Z = jnp.where(mask_Z, (solution[:, 2] - dataset[:, 1]) / (dataset[:, 3] + 1e-12), 0.0)
    count_X = jnp.sum(mask_X)
    count_Z = jnp.sum(mask_Z)
    loss_X = jnp.where(count_X > 0, jnp.sqrt(jnp.sum(resid_X * resid_X) / count_X), 1000000000000.0)
    loss_Z = jnp.where(count_Z > 0, jnp.sqrt(jnp.sum(resid_Z * resid_Z) / count_Z), 1000000000000.0)
    return (loss_X + loss_Z) / 2.0

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    eps1, eps2, q, f = unscaled_parameters
    trainable_parameters = {"eps1": eps1, "eps2": eps2, "q": q, "f": f}
    fixed_parameters = constants["fixed_parameters"]
    X = y[0]
    Y = y[1]
    Z = y[2]

    dXdt = (q * Y - X * Y + X * (1 - X)) / eps1
    dYdt = (-q * Y - X * Y + f * Z) / eps2
    dZdt = X - Z
    return jnp.array([dXdt, dYdt, dZdt])

@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Kvaerno5()
    t_eval = constants["t_eval"]
    other_args = {"constants": constants, "trainable_variables": trainable_variables}
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=constants["init_time"],
        t1=t_eval[-1],
        max_steps=50000,
        dt0=constants["init_timestep"],
        y0=constants["init_cond"],
        args=other_args,
        saveat=diffrax.SaveAt(ts=t_eval),
        throw=False,
        stepsize_controller=diffrax.PIDController(
            rtol=constants["stepsize_rtol"],
            atol=constants["stepsize_atol"],
        ),
    )
    return sol.ts, sol.ys, sol.result

@jax.jit
def _compute_loss_value(constants, trainable_variables, solution_time, solution):
    dataset = constants["dataset"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    eps1, eps2, q, f = unscaled_parameters
    trainable_parameters = {"eps1": eps1, "eps2": eps2, "q": q, "f": f}
    fixed_parameters = constants["fixed_parameters"]
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 0] - dataset[:, 0]) / (dataset[:, 2] + 1e-12)))
    loss += jnp.mean(jnp.square((solution[:, 2] - dataset[:, 1]) / (dataset[:, 3] + 1e-12)))
    loss = jnp.sqrt(loss / 2)
    return loss

@jax.jit
def _compute_loss_problem(constants, trainable_variables):
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = result != RESULTS.successful
    loss_value = _compute_loss_value(constants, trainable_variables, solution_time, solution)
    failed = failed | ~jnp.all(jnp.isfinite(solution)) | ~jnp.isfinite(loss_value)
    return jnp.where(failed, constants["error_loss"], loss_value)

def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    eps1, eps2, q, f = unscaled_parameters
    trainable_parameters = {"eps1": eps1, "eps2": eps2, "q": q, "f": f}
    fixed_parameters = constants["fixed_parameters"]
    writeout_array = np.zeros([solution_time.shape[0], 6])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    writeout_array[:, 5] = solution[:, 2]
    return writeout_array
