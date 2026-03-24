# Getting Started with Solitaire

You installed Solitaire. Now what?

This guide walks you through your first session, from a blank database to a working
persona that remembers you. Total time: under 5 minutes if you skip everything
skippable, 10-15 if you engage with every step.

## What happens at first launch

When your host agent (Claude Code, Cowork, or whatever you're running Solitaire on)
starts a new session, Solitaire detects that no persona exists yet. Instead of
asking you to pick from a template library, it builds one with you from scratch.

The onboarding flow has ~13 steps. Every step after the first is skippable. You
can race through in under 2 minutes or take your time. The persona you create
improves with use either way.

## The steps

**1. Welcome.** Solitaire introduces itself. One button: "Let's go."

**2. What are you working on?** Tell Solitaire what you'll use it for. This is
the single most important input in the whole flow. A good answer ("managing client
deliverables for my consulting firm") gives Solitaire enough signal to set sensible
defaults for everything that follows. A vague answer ("stuff") still works, but
the defaults will be more generic.

**3. Research.** Solitaire uses your intent to infer which personality traits suit
your use case. If web search is available, it pulls context about your domain. If
not, it falls back to a built-in heuristic table covering 10 common domains
(business ops, finance, software, creative writing, health, legal, education,
gaming, research, personal productivity). You'll see the proposed traits before
anything is locked in.

**4. Trait proposal.** Solitaire shows you the seven disposition traits it inferred:
observance, assertiveness, conviction, warmth, humor, initiative, empathy. Each is
a value between 0.05 and 0.95. You can accept them, tweak individual values, or
skip and use defaults. These are starting points; they'll shift over time based on
how you interact.

**5. Working style.** Four quick preferences:

- Communication density: terse, balanced, or thorough
- Feedback style: gentle, direct, or blunt
- Initiative level: wait for instructions, suggest proactively, or act autonomously
- Pacing: methodical, adaptive, or fast-and-loose

Each choice adjusts your traits by a small amount (capped at +/-0.15). Skip if you
don't have a preference.

**6. Personality interview (optional).** Five questions that calibrate how your
persona handles nuance: how much autonomy you want it to take, how you respond to
pushback, how warm or formal the tone should be, whether humor fits, how much
detail you expect. Each answer shifts traits by up to +/-0.30 total across all
questions. Skip the whole thing if the trait proposal already looks right.

**7. Naming.** Solitaire suggests a name based on your domain (e.g., "The Analyst"
for finance work, "The Architect" for software). Accept, modify, or type your own.

**8. North star (optional).** A sentence or two defining what your persona cares
about most. This anchors long-term behavior. Example: "I want to catch the things
my user misses and say them plainly." If you skip it, the persona still works. It
just won't have an explicit orienting principle.

**9. Seed questions.** Five quick facts about you: your name, your role, what you're
working on right now, what tools you use, and one open-ended "anything else." These
go straight into the knowledge graph so your persona has context from turn one
instead of learning everything from scratch over the first few sessions.

**10. Preview and confirm.** Review everything. You can redo the interview, redo the
working style questions, start over entirely, or confirm. Confirming creates the
persona and drops you into your first real session.

## After onboarding

From this point, Solitaire operates in the background. Your host agent handles the
integration, but here is what is happening under the hood:

- **Every session start**: Solitaire boots, loads your persona, and injects relevant
  context (recent memories, identity state, session residue from last time) into the
  model's prompt.
- **Every turn**: Your messages and the model's responses are ingested into the
  knowledge graph. Entities are extracted. The identity graph updates.
- **Every session end**: Retrieval weights adjust based on what the model actually
  used. Session residue captures the texture of the conversation.

You don't need to run any commands manually unless you want to. The standard
lifecycle (boot, auto-recall, ingest-turn, end) is handled by the host agent.

## Progressive revelation

Solitaire doesn't show you everything on day one. Features surface as you accumulate
sessions:

- **Session 1**: Boot, ingest, respond. The basics.
- **Session 3**: Identity traits become visible. You can see how the persona is
  developing.
- **Session 5**: Pattern detection surfaces. Solitaire starts flagging recurring
  themes, knowledge gaps, and behavioral patterns.
- **Session 10**: Full system access. Self-learning metrics, tool-finding, weight
  adjustment visibility.

If you're a power user and want everything immediately: `solitaire reveal all`.

## What if I want to start over?

Run onboarding again:

```bash
solitaire onboard create
```

This walks you through the same flow. Your existing persona and data are untouched
until you confirm a new one.

## What if I have multiple use cases?

Create additional personas. Each persona has its own trait profile, identity graph,
and knowledge partition. Switch between them at boot:

```bash
solitaire boot --persona analyst
solitaire boot --persona creative
```

Context stays siloed. Your finance persona won't see your creative writing sessions
unless you explicitly share entries between them.

## Common questions

**How much does my persona change over time?**
Traits drift gradually based on interaction patterns. The drift is small per session
(capped) and transparent. You can inspect current values anytime with
`solitaire identity show`.

**Can I edit my traits directly?**
Yes. `solitaire identity set assertiveness 0.7` or use the onboarding flow to
recalibrate.

**Where is my data stored?**
Everything lives in a local SQLite database (`rolodex.db`) in your workspace folder.
Nothing leaves your machine unless you configure an API key for enhanced mode, in
which case only the summarization calls go to the API. Your raw data stays local.

**What's enhanced mode vs. verbatim mode?**
Without an API key, Solitaire uses heuristic extraction and keyword search. All
features work. With an API key (`ANTHROPIC_API_KEY`), ingestion uses LLM-powered
summarization for richer entity extraction and topic routing. Both modes produce
a working persona. Enhanced mode is better at surfacing subtle patterns over time.

**What if I skip everything in onboarding?**
You get a persona with neutral defaults (all traits at 0.5), a generic name, no
north star, and an empty rolodex. It works. It just takes a few more sessions for
the persona to develop meaningful context. The system is designed so that skipping
is never punished, only slower.
