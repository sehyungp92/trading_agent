# Golden Days — Regression Test Data

## What is a golden day?

A frozen dataset from a specific trading day with known-good outcomes and human feedback.
Used to regression-test the data pipeline, classification engine, and report quality.

## Directory structure

```
tests/golden_days/
  2026-02-15/
    raw_events/
      trades.json              # list of trade event dicts (as emitted by bots)
      missed.json              # list of missed opportunity dicts
    expected_classifications.json  # trade_id → {root_causes, process_quality_score}
    expected_curated/
      summary.json             # what the pipeline should produce
    human_feedback.json        # your actual corrections from that day
    reference_report.md        # the report you rated "good"
    metadata.json              # date, bots, top_anomaly, biggest_loss_driver, crowding_alerts
```

## How to add a new golden day

1. Pick a day with interesting data (mix of wins, losses, process failures, missed opps)
2. Export the raw events from the event queue for that date
3. Run the pipeline and verify the curated output matches your expectations
4. Save your human feedback/corrections for that day
5. Rate the daily report — if it was "good", save it as reference_report.md
6. Fill in metadata.json with the top anomaly, biggest loss driver, and any crowding alerts
7. Commit the directory under tests/golden_days/

## How regression tests use this

- `test_classification_accuracy`: Run pipeline on raw_events, compare root causes to expected_classifications
- `test_metric_stability`: Compare pipeline output metrics to expected_curated (within 5% tolerance)
- `test_report_quality_heuristics`: Generate report, verify it mentions top anomaly, biggest loss driver, and all crowding alerts
