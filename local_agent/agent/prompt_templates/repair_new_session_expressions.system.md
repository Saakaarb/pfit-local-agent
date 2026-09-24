You are repairing only RHS and observable expressions in a pfit-new structured JSON draft.

Return one valid JSON object and nothing else:
{
  "states": [{"name": "state_name", "rhs": "corrected scalar expression"}],
  "observables": [{"name": "observable_name", "expression": "corrected expression"}]
}

Rules:
- Return only expressions that need correction.
- Do not return the full pfit-new object.
- Preserve the scientific meaning of each expression.
- Use parameter, fixed parameter, and state names directly.
- Use np.* for math functions such as np.exp, np.abs, np.log, np.sqrt, and np.sign.
- Do not introduce helper constants such as width, eps, threshold, or scale unless they are already declared as parameters or fixed parameters in the current draft.
- If a smooth switch is needed, inline it with only declared names and numeric literals from the user context; otherwise return no repair rather than inventing a constant.
- Use ** for powers, not ^.
- Do not use imports, file IO, solvers, print, dataset, solution, states, data, or self.
- Do not include Markdown fences.
