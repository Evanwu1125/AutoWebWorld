---
name: fsm-review
description: Review and validate FSM JSON structure for completeness and correctness.
---

# FSM Review Skill

You now have expertise in reviewing FSM (Finite State Machine) JSON files for web application simulation.

## Review Checklist

### 1. Structure Validation
- Every page must have a unique `id`
- Every action must reference a valid target page `id`
- The `meta.terminal_pages` list must only contain existing page IDs
- Home page must exist and be reachable

### 2. Navigation Completeness
- Every non-terminal page should have at least one outgoing action
- Terminal pages should not have navigation actions
- Back navigation should be consistent

### 3. Content Quality
- Page names should be descriptive and match the theme
- Actions should have clear, user-facing labels
- No duplicate pages or orphaned nodes
