#!/usr/bin/env bash
# Smoke-test a running kev.serve instance: ./test_model.sh [port]
set -euo pipefail
PORT="${1:-8008}"

curl -sS -X POST "http://127.0.0.1:${PORT}/v1/systemone" \
  -H 'content-type: application/json' \
  -d '{
    "state": "I was charged twice for the same order last week and support has not replied in 3 days. This is unacceptable, fix it now.",
    "model": "kev-latest",
    "questions": {
      "billing": {"type": "noul", "instructions": "Is this about a billing problem?"},
      "tone": {
        "type": "choice",
        "instructions": "What is the customer tone?",
        "criteria": {"calm": null, "frustrated": null, "angry": null}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "very urgent", "critical"]
      }
    }
  }' | python3 -m json.tool
