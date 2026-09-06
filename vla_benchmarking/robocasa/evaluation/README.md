# RoboCasa evaluation

The evaluator fixes the PickPlace-21 task/seed denominator to the official
target scene split.  `preflight` plans all 21 × 10 cells, `smoke` can exercise
an explicitly authorized subset, and `full --execute-motion` invokes the live
RoboCasa bridge one cell at a time with resumable outputs.  Dependency,
geometry, horizon, and controller failures remain terminal rows.  It never
substitutes bbox overlap for RoboCasa's official task success predicate.
