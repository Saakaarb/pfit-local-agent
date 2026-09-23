Translate only the RHS from user_defined_system.

Session summary:

$session_summary

Input YAML:

$input_yaml

Already translated helper functions:

$translated_helper_functions

RHS intermediates from user_model.py:

$rhs_intermediate_inventory

Reference contract from pfit-claude:

$pfit_claude_jax_reference

Use RHS intermediates only to translate rhs. Do not return this inventory.
Do not use intermediate names as bare values in rhs; inline their expressions or
call helper functions with explicit arguments.

User model or pseudocode:

$user_model

Return only:
{
  "rhs": [],
  "helper_functions": [],
  "review": ""
}
The first character of your response must be `{`.
