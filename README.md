# Procurement Exception Advisor

An agentic decision-support system for evaluating emergency and sole-source procurement requests, identifying missing evidence, and producing audit-defensible recommendations.

## Initial MVP

The first version focuses on emergency procurement requests for a single pilot jurisdiction.

## Project structure

- `src/agent/` — reasoning loop and orchestration
- `src/mcp/` — MCP server, tools, resources, and prompts
- `src/rag/` — ingestion, chunking, retrieval, and citations
- `src/ui/` — user interface
- `data/policies/` — authoritative laws, policies, and procedures
- `data/cases/` — mock and test cases
- `data/templates/` — justification and audit-file templates
- `tests/` — automated tests
- `docs/` — design notes and architecture
- `config/` — application settings

## Mock evaluation cases

The initial emergency-procurement evaluation dataset is under `data/cases/`:

- `inputs/` — case facts and available-document descriptions that may be shown to the agent
- `expected/` — hidden answer keys used to evaluate the agent's recommendation and reasoning
- `schemas/` — JSON Schema definitions for the input and expected-result files
- `manifest.json` — index connecting each input case to its answer key

Do not include files from `expected/` in the agent's context during an evaluation run.
