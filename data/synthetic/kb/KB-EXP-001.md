# Export Failure Runbook

Document owner: Platform Operations (fictional).
Effective: 10 September 2024.

## Symptoms
A customer reports that a CSV export stays in the "pending" state or ends with error EXP-504.

## Resolution steps
1. Check the export queue depth on the operations dashboard.
2. If the queue is above 500 items, restart the export worker and ask the customer to retry.
3. Confirm the signing certificate of the customs gateway has not expired.

## Export file lifetime
Generated export files are kept for 7 days and then deleted automatically.
