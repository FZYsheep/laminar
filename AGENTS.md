# Agent Guidelines

Read and follow [CONTRIBUTING.md](CONTRIBUTING.md) for development, validation, formatting, dependency, and release-hygiene requirements.

## Before Handing Off or Opening a PR

1. **Self-review until clean:** Iterate `review -> fix -> re-review` on all proposed changes against the target base branch (e.g., `git diff <base>...HEAD`, plus any staged/unstaged changes), checking correctness, edge cases, concurrency/fault-tolerance assumptions, test coverage, and release hygiene until no actionable issues remain.
   - Agent self-review is optional if the human contributor reviews the changes directly.
   - When running agent self-review, strongly prefer different model(s) (ideally from a different model family/provider) than the authoring model(s).
2. **Re-validate after the final fix:** Re-run relevant validation commands from [CONTRIBUTING.md](CONTRIBUTING.md).
3. **Require a final human-involved review:** Hand off for a final human-involved review pass (which may be agent-assisted) and, by default, report the exact authoring and self-review model(s) used (not just tool names) for the PR description; only omit them and suggest recommended review model family/families (exact versions not required) instead if the contributor asks not to disclose them for privacy reasons.
