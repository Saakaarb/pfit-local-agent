Repair attempt $attempt failed validation.

Validation error:

$validation_error

Session summary:

$session_summary

Input YAML:

$input_yaml

Candidate helper functions from user_model.py:

$helper_function_inventory

RHS intermediates from user_model.py:

$rhs_intermediate_inventory

Reference contract from pfit-claude:

$pfit_claude_jax_reference

Use these RHS intermediates only to translate rhs. Do not return this inventory.
Do not use these intermediate names as bare values in rhs; inline their expressions
or call helper functions with explicit arguments.

User model or pseudocode:

$user_model

Previous fragment response:

$previous_response

Rendered generated_script.py, if available:

$generated_script

Return only the corrected pfit-jax fragment JSON object now:
{
  "rhs": [],
  "helper_functions": [],
  "loss_body": "",
  "writeout_body": "",
  "review": ""
}
The first character of your response must be `{`. Do not explain the model. Do not include Markdown fences.
