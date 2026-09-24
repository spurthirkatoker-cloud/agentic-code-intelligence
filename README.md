# CodeLens AI - Agentic Code Intelligence

A Code Retrieval System that takes a large codebase and a natural-language query and returns the Top-10 most relevant code snippets.

## Architecture
Dense + BM25 + AST -> RRF -> Top-30 -> Cross-Encoder -> Top-10
