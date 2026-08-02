# REQUIREMENTS.md — Concluding Layer (written by Planner, loop 1)

## Context

The programming layer has produced a runnable router in `code/` and predictions in `output.csv`.
Summarize truthfully — include failures and open risks, not just wins.

## Tasks

1. **CONCLUDER_ONE — `SUMMARY.md`** (repo root): what was built (architecture, layers, models
   used), eval results from `code/eval.py` (quote the real numbers), current situation, what
   we still need from the user, and ranked future improvements.

2. **CONCLUDER_TWO — solution README + submission check.** Update `README_SOLUTION.md` with
   exact Windows setup/run commands (uv venv, install, run, eval), env vars (none required;
   optional ones documented), then verify the submission contract: output.csv has exactly
   columns message_id,action,message_type,reason,confidence,evidence_message_ids and one row
   per message_id in dataset/messages.csv; report pass/fail per check.

## Rules

- Quote real command output for eval numbers; do not estimate.
- Append one §5.2 log entry to `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt`
  (parent_agent=Claude Code (Fable 5)).
