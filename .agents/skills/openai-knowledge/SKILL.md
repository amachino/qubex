---
name: openai-knowledge
description: Pull current OpenAI API and platform documentation through the official docs workflow. Use when a task depends on current OpenAI models, APIs, SDKs, or platform behavior and memory could be stale.
---

# OpenAI Knowledge

## Overview

Use current official OpenAI documentation before answering or implementing
OpenAI-related work. Prefer the docs MCP workflow and official domains only.

## Workflow

1. Prefer the official docs workflow.
   - If the built-in `openai-docs` skill is available, use it.
   - Otherwise use the OpenAI developer docs MCP server tools.
   - Only fall back to browsing `developers.openai.com` or
     `platform.openai.com` if MCP is unavailable.
2. Verify volatile facts.
   - Confirm model names, API surfaces, SDK behavior, and migration guidance
     instead of relying on memory.
   - Include exact dates when clarifying latest or time-sensitive guidance.
3. Cite sources.
   - Link the specific docs pages you used.
   - Keep quotes short and prefer paraphrase.
4. Translate docs into repo decisions.
   - Summarize the implication for the task in one short section: recommended
     API, risk, required code changes, and compatibility notes.
5. Stay official.
   - Do not mix unofficial blog posts, community answers, or third-party
     wrappers into the recommendation unless the user asks.
