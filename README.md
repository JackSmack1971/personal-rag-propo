# Personal RAG Chatbot (Propositional) — Starter (COST defaults wired)

The **Costs** tab now reads defaults from `.env` populated with values derived from your spreadsheet:

- COST_MONTHLY_QS = 6000  (Medium scenario → Queries/day × 30)
- COST_PROMPT_TOKENS = 300
- COST_COMPLETION_TOKENS = 300
- COST_PRICE_PER_1K = 0.000375
- COST_BASE_FIXED = 50.0

You can change these in `.env` at any time.
