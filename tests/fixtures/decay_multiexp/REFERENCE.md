Ported from pfit-claude origin/deployed_branch at 1b415d8. Input CSVs, YAML and
reference generated code are retained. Tests override optimizer settings locally
for bounded DE/L-BFGS regressions. These generated scripts are reference fixtures,
not evidence of live local-LLM translation. Sneyd reference_parameters.csv (where
present) is the saved reference fit, not asserted ground truth. The decay Python
user_model was added locally to express the same analytic equations and loss;
user_info.txt records the explicit reference loss for local checking.
