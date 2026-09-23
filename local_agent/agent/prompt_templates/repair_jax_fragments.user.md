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

Return the full corrected pfit-jax fragment JSON object now. Start from
previous_response, repair only the field named or implied by the validation
error, and copy all other fields exactly.
The first character of your response must be `{`. Do not explain the model. Do not include Markdown fences.
