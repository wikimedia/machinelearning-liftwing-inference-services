---
description: Fetch a Gerrit change by ID and run a structured code review using the gerrit-reviewer agent. Use when the user wants feedback on a Gerrit patch — code quality, security, project conventions.
disable-model-invocation: false
argument-hint: "[change-id]"
context: fork
agent: gerrit-reviewer
---

# Review a Gerrit Patch

Fetch and review a Gerrit change (patch) using the Gerrit MCP tools.

## Steps

1. Use `mcp__gerrit__get_change_details` with the change ID from `$ARGUMENTS`
2. Use `mcp__gerrit__list_change_files` to get all modified files
3. Use `mcp__gerrit__get_file_diff` for each modified file to see the diffs
4. Use `mcp__gerrit__list_change_comments` to see existing review comments
5. Use `mcp__gerrit__get_commit_message` to read the commit message

## Review Output

Provide a structured review with:

- **Summary**: What the patch does (1-2 sentences)
- **Commit message**: Is it clear and well-formatted? (Wikimedia convention: `component: Subject` / `Why:` / `What:` / `Assisted-by:` / `Bug:` / `Change-Id:`)
- **Code quality**: Style, correctness, edge cases, potential bugs
- **Security**: Any security concerns (XSS, injection, secret leakage, auth bypass, etc.)
- **Testing**: Are there tests? Are they sufficient?
- **Comments**: Notable existing review comments and their status

If the project has a CLAUDE.md with project-specific conventions (services, hooks, schemas, etc.), check the patch against those.

## Input

$ARGUMENTS
