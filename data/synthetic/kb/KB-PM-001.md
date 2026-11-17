# Post-incident Review INC-7731 (Acme Freight, confidential)

Incident INC-7731: shipment label printing was unavailable for 3 hours on 4 March 2025.

Root cause: the label-service credential rotation job failed silently. The follow-up action is to add an alert on failed rotations. Review restricted to Acme Freight administrators.
