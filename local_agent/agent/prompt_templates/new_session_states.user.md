Session files and user-provided context:

$session_context

Frozen dataset and parameters:
$frozen_parameters

First, in review, quote the supplied initial-condition clauses and expand any
group declarations into named states and values. A statement such as "the other
gating states are zero" supplies zero for every remaining member of the named
gating group; it does not need a separate equation for each member. Combine
these with the frozen experiment overrides. Then return the complete states JSON,
listing only states whose initial values remain unresolved in missing_inputs.
