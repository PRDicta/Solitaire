# Agent Onboarding Guide

Platform-agnostic onboarding documentation for AI agents integrating with Solitaire.

## Overview

This guide helps non-Claude agents (Hermes, OpenClaw, Gemini CLI, etc.) successfully onboard users to Solitaire without relying on CLAUDE.md or Claude-specific tooling.

## Prerequisites

- Python 3.10+
- Solitaire installed (`pip install -e .` or via install script)
- Understanding of your platform's tool-calling capabilities

## Quick Start

### 1. Installation Verification

Before starting onboarding, verify Solitaire is properly installed:

```bash
# Check CLI is available
solitaire --version

# Verify MCP server exists
ls mcp-server/server.py
```

### 2. Understanding the Onboarding Flow

The onboarding flow is a multi-step conversation that builds a user persona:

1. **Welcome** - Introduction and consent
2. **Intent** - What the user wants to accomplish
3. **Live Research** - Background research on user's domain
4. **Trait Proposal** - Suggested personality traits
5. **Working Style** - Communication preferences
6. **Interview** - Deep-dive questions
7. **Naming** - Persona name selection
8. **North Star** - Core values/goals
9. **Seed Questions** - Knowledge seeding
10. **Preview** - Review proposed persona
11. **Confirm** - User approval
12. **Apply** - Save persona and activate

### 3. CLI-Based Onboarding

For agents without MCP support, use the CLI commands:

#### Start the Flow

```bash
solitaire onboard create [--intent "user's stated goal"]
```

Returns JSON with:
- `step_id`: Current step identifier
- `step_type`: info/question/multiple_choice/preview/confirm
- `content`: Text to present to user
- `options`: (for multiple_choice) Available options

#### Process Each Step

```bash
solitaire onboard flow-step <step_id> "<user_response>"
```

Returns JSON with:
- `next_step_id`: Next step to process
- `complete`: Boolean indicating if flow is done
- `persona`: (if complete) The created persona object

### 4. Example Session

```bash
# Step 1: Start
$ solitaire onboard create --intent "help me write better code"
{
  "step_id": "welcome",
  "step_type": "info",
  "content": "Welcome to Solitaire! I'll help you create a personalized AI companion..."
}

# Step 2: Present to user, get response
# User says: "I'm ready"

$ solitaire onboard flow-step welcome "I'm ready"
{
  "step_id": "intent",
  "step_type": "question",
  "content": "What's your main goal or project you're working on?"
}

# Step 3: Continue with user's intent
$ solitaire onboard flow-step intent "Building a web app"
{
  "step_id": "working_style",
  "step_type": "multiple_choice",
  "content": "How do you prefer to work?",
  "options": ["Fast and iterative", "Thoughtful and thorough", "Collaborative", "Independent"]
}

# Continue until complete=true
```

### 5. Platform-Specific Integration

#### Hermes Agents

Hermes agents should:
1. Call `solitaire onboard create` via shell tool
2. Parse JSON response
3. Present `content` to user
4. Capture user response
5. Call `solitaire onboard flow-step` with step_id and response
6. Loop until `complete=true`

Example Hermes skill:
```yaml
name: solitaire_onboard
tools:
  - shell: solitaire onboard create
  - shell: solitaire onboard flow-step {step_id} {input}
```

#### OpenClaw Agents

OpenClaw agents can use the Python API directly:

```python
from solitaire import SolitaireEngine

engine = SolitaireEngine()
result = engine.onboard_start(intent="user goal")

while not result.get('complete'):
    # Present result['content'] to user
    user_input = get_user_input()
    result = engine.onboard_flow_step(
        result['step_id'], 
        user_input
    )
```

#### Gemini CLI

Gemini CLI agents should wrap the CLI commands:

```bash
# In your agent's tool definition
{
  "name": "solitaire_onboard",
  "command": "solitaire onboard {action}",
  "args": ["create|flow-step"]
}
```

### 6. Handling Different Step Types

#### Info Steps
- Simply display `content` to user
- Wait for acknowledgment (any input)
- Proceed to next step

#### Question Steps
- Display `content` as prompt
- Capture free-form user response
- Pass response to flow-step

#### Multiple Choice Steps
- Display `content`
- Present `options` as numbered list
- Capture selection (number or text)
- Pass to flow-step

#### Preview Steps
- Display `content` (shows proposed persona)
- Ask user to confirm or edit
- Pass "confirm" or edit request

#### Confirm Steps
- Final confirmation before saving
- User must explicitly approve
- Returns `complete=true` and persona object

### 7. Error Handling

Common issues and solutions:

**"solitaire: command not found"**
- Ensure Solitaire is installed: `pip install -e .`
- Check Python environment is activated

**JSON parse errors**
- Ensure user input is properly escaped
- Wrap multi-word inputs in quotes

**Step ID errors**
- Always use the exact `step_id` from previous response
- Don't hardcode step IDs (they're dynamic)

### 8. Post-Onboarding

Once `complete=true`:

1. The persona is automatically saved
2. Call `solitaire boot` to load the new persona
3. Begin normal operation with memory enabled

```bash
solitaire boot
```

Returns context to inject into system prompt.

## Testing Your Integration

Test the complete flow:

1. Start fresh: `rm -rf ~/.solitaire` (backup first!)
2. Run through all onboarding steps
3. Verify persona is created: `solitaire persona show`
4. Test memory: Have a conversation, then `solitaire recall`

## Tips for Good Onboarding

1. **Don't rush** - Let users think about their answers
2. **Explain why** - Briefly explain why you're asking each question
3. **Offer examples** - For open-ended questions, suggest examples
4. **Allow edits** - Users can revise answers before final confirm
5. **Save progress** - The flow is stateful; users can resume if interrupted

## Troubleshooting

### Flow hangs or crashes
- Check Python version: `python --version` (need 3.10+)
- Check logs: `~/.solitaire/logs/`
- Try resetting: `solitaire system reset`

### Persona not saving
- Check disk space
- Verify write permissions to `~/.solitaire/`
- Look for error messages in CLI output

### Steps not progressing
- Ensure you're passing the exact step_id from previous response
- Check that user input isn't empty
- Verify JSON is properly formatted

## Resources

- Main documentation: README.md
- Claude-specific guide: CLAUDE.md
- Agent overview: AGENTS.md
- Skill definition: skill/SKILL.md
- API reference: solitaire/core/onboarding_flow.py

## Support

For issues specific to your agent platform:
1. Check your platform's tool-calling documentation
2. Review the CLI reference in this guide
3. Open an issue with your platform details and error messages

---

*This guide is maintained for all agent platforms. Contributions welcome!*