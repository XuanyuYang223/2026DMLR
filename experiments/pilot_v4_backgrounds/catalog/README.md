# v4 catalog artifacts

See [DATA_PLATFORM.md](../DATA_PLATFORM.md) for the main documentation and interface contract. `build_catalog.py` performs only CPU checks of mathematics and existing source-development evidence.

- `../TASK_CATALOG.csv`:57 existing task/template/encoding-interface records, with this round's roles in a separate column.
- `DIRECT_RELATION_DEFINITIONS.csv`:35 defining relations and proof sketches.
- `RELATION_GRAPH.csv`:132 closure edges under D8 input actions and affine answer maps, including composed actions.
- `AGGREGATE_RELATIONS.csv`: joint formulas and their proof scope.
- `BACKGROUND_DESIGN.json`: two targets, three backgrounds, fixed candidate slots, and reasons for changes.
- `BACKGROUND_RELATION_GRAPH.csv` / `BACKGROUND_AGGREGATE_RELATIONS.csv`: relations among actual nodes and missing members of joint formulas.
- `SOURCE_REFERENCE_CURVES.csv` / `SOURCE_ANSWER_DISTRIBUTIONS.csv` / `BACKGROUND_MATCHING.csv`: existing source single-task curves, training answer distributions, and remaining differences from background changes.
- `RELATION_AUDIT.json` / `CATALOG_AUDIT.json` / `SOURCE_REFERENCE_AUDIT.json`: check scope, results, source hashes, and read boundaries.
- `REPLACEMENT_RELATION_COUNTEREXAMPLES.csv`: eight-action answer-map counterexamples for six existing candidate pairs, including this round's two pairs; no claim of mathematical unrelatedness.

This explicitly registers existing relations and their composition closure. It is not a complete database of all mathematical relations or proof of neural-representation mechanisms or transfer benefits.
