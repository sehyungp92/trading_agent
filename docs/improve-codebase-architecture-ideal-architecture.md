# The Ideal Architecture Implied by `improve-codebase-architecture`

This is the architecture the skill *implies* — read on its own terms. The skill is not a program; it is an **agent skill**: prose instructions an AI executes, composed with sibling skills it delegates to. So the unit of decomposition is not a code module or a typed data structure — it is the **skill** and the **shared artifact**. The ideal architecture is what you get when you apply the skill's own design principles (**module, interface, depth, seam, adapter, leverage, locality**, plus the **deletion test** and the two adjacent maxims) to that composition.

This supersedes the code-module framing of the earlier guide, which imposed a program structure the skill never asks for. Where that guide invented classes and function signatures, this one describes the composition the skill actually names: an orchestrator, four sibling capabilities, three persistent artifacts, and a vocabulary spine.

---

## 1. The reading: a skill is a module

The whole architecture follows once you accept one mapping:

| Design term | In this system |
|---|---|
| **Module** | A skill (`improve-codebase-architecture`, `/grilling`, `/domain-modeling`, `/codebase-design`) or the `Explore` subagent |
| **Interface** | A skill's frontmatter description + its invocation contract — the small surface you touch to use it |
| **Implementation** | The skill's body — the reasoning and steps it hides |
| **Depth** | Simple to invoke, complex in what it does. `/grilling` is deep: you hand it a candidate, it runs an entire design-tree walk |
| **Seam** | A delegation boundary — the point where the orchestrator calls a sub-skill and could swap what sits behind it |
| **Adapter** | A thin wrapper over an external capability (the `Explore` subagent; the write-to-temp-and-open presentation step) |
| **Leverage** | The shared vocabulary — a small fixed set of terms that constrains every output of every phase |
| **Locality** | One home per concern: domain writes live only in `/domain-modeling`, the report scaffold only in `HTML-REPORT.md` |

The **deletion test** applies to skills directly: *would inlining this sub-skill into the orchestrator concentrate complexity, or scatter it?* A sub-skill earns its existence when inlining it would spray its judgement across the orchestrator. That test, run against every element below, is what makes this architecture *ideal* rather than merely *workable*.

---

## 2. System shape

Three layers. A thin conductor on top, deep capabilities in the middle, a shared spine underneath that every layer reads.

```
                         ┌───────────────────────────────────────────────┐
   ORCHESTRATOR          │        improve-codebase-architecture           │
   (thin conductor)      │   sequences phases · owns the one-way gate     │
                         └───────────────┬───────────────────────────────┘
                                         │ delegates across seams
        ┌────────────────┬───────────────┼───────────────┬────────────────┐
        ▼                ▼               ▼                ▼                ▼
  DEEP SIBLINGS   Explore subagent   /grilling      /domain-modeling  /codebase-design
                  (codebase walk)  (design walk)   (model writes)   (vocabulary + design-it-twice)
        └────────────────┴───────────────┴───────────────┴────────────────┘
                                         │ every element reads/writes ↓
   SHARED SPINE   ┌──────────────────────────────────────────────────────────┐
   (state + vocab)│  Vocabulary: CONTEXT.md (domain) + /codebase-design (arch) │
                  │  Artifacts:  CONTEXT.md · docs/adr/ · HTML-REPORT.md       │
                  └──────────────────────────────────────────────────────────┘
```

The **narrow waist** in this reading is not a data type — it is the **shared vocabulary plus the set of candidates carried in the report**. Everything upstream (reading context, exploring) produces candidates named in the shared terms; everything downstream (presenting, grilling, recording) consumes that same named set. The vocabulary is what lets the phases stay independent: a candidate described as *"the Order intake module is shallow — deleting it concentrates complexity"* means the same thing to the report, to `/grilling`, and to `/domain-modeling` without any of them re-deriving it.

---

## 3. The spine: shared vocabulary (the highest-leverage element)

Two glossaries flow through everything, and keeping them exact is the single highest-leverage discipline in the system.

**Domain vocabulary** comes from `CONTEXT.md` — it gives names to good seams. When a candidate concerns order intake, it is *"the Order intake module,"* never *"the FooBarHandler"* and never *"the Order service."*

**Architecture vocabulary** comes from `/codebase-design` — module, interface, depth, seam, adapter, leverage, locality — together with its principles: the deletion test, *"the interface is the test surface,"* and *"one adapter = hypothetical seam, two = real."*

The discipline: use these terms exactly, in every phase and every sub-skill, and never drift into *component, service, API,* or *boundary.* This is **leverage** in the precise sense — a small, fixed surface (the two glossaries) that constrains a large surface (every finding, every card, every design decision). Change the glossary once and every future report changes with it. Drift in the vocabulary is not cosmetic; it is an architecture defect, because it breaks the narrow waist that keeps the phases speaking the same language.

`/codebase-design` is therefore not just a phase-3 helper — it is the **vocabulary authority** for the entire run. Treat its glossary as loaded before exploration even begins, so findings come back correctly named.

---

## 4. The persistent artifacts (the model layer)

Four artifacts hold the system's state. Each has exactly one home, and read/write ownership is deliberately asymmetric.

`CONTEXT.md` is the **domain model** — the source of domain names. It is *read* in phase 1 to seed exploration and *written* only in phase 3, and only through `/domain-modeling` (created lazily if absent). Many elements read it; one writes it. That asymmetry is what makes it trustworthy.

`docs/adr/` is the **decision log**. It is *read* in phase 1 so the review does not re-litigate settled decisions, and *appended* in phase 3 when a candidate is rejected for a load-bearing reason. ADRs are a boundary the review respects, not one it argues with — except where friction is real enough to warrant reopening one, which is surfaced explicitly.

`HTML-REPORT.md` is the **render scaffold** — the Tailwind + Mermaid patterns, card structure, and diagram templates. Keeping it separate from the orchestrator is a **locality** decision: it stops the conductor from becoming a markup factory, so phase-sequencing never couples to CDN details.

The **temp report** (`<tmpdir>/architecture-review-<timestamp>.html`) is deliberately disposable, per-run, and never lands in the repo. It is state that exists only long enough to be looked at.

---

## 5. The orchestrator skill (the thin conductor)

`improve-codebase-architecture` is the top-level module, and its whole virtue is being **thin**. Its interface: `disable-model-invocation: true` (invoked deliberately, never auto-triggered), a description that states its purpose, and the three-phase contract. Its job is to *sequence* — read context, dispatch exploration, own the report, hand off to grilling — and to *hold the one-way gate* (no interfaces proposed before phase 3). It does no design work itself.

Deletion test: delete the orchestrator and call the sub-skills directly, and you lose the phase sequencing and the gate — every invocation would re-improvise the order and might leak interface proposals into the report. Inlining **concentrates** that coordination. Keep it.

But the test cuts both ways, and this is the orchestrator's one failure mode: if it ever starts absorbing sub-skill logic — doing the design walk itself, inventing report markup, deciding when an ADR is warranted — it stops being deep (simple invocation, orchestration hidden) and becomes a shallow god-module where all the complexity has pooled. The orchestrator stays ideal only while it stays a conductor.

---

## 6. The deep sibling skills (the seams)

Each sibling is a deep module reached across a delegation seam. The orchestrator names the seam; the sibling hides the work.

**`Explore` subagent** (via the Agent tool) — invoked with the codebase and the seeded vocabulary; returns friction observations, including the result of applying the deletion test to anything suspected of being shallow. Hidden: the entire organic codebase walk. It is an **adapter** over a non-deterministic capability — kept as thin as possible, seeded with the glossary so findings come back named, and asked for observations rather than judgement.

**`/grilling`** — invoked with the chosen candidate; runs the interactive design-tree walk (constraints, dependencies, the shape of the deepened module, what sits behind the seam, which tests survive). Deletion test: inline it and the orchestrator balloons from conductor to designer as the walk logic floods in. **Concentrates.** A real seam.

**`/domain-modeling`** — the **single write boundary** for the domain model. Invoked as decisions crystallise; it adds new terms to `CONTEXT.md`, sharpens fuzzy ones, and offers an ADR when a rejection is load-bearing. The judgement of *when* an ADR is warranted (only when a future explorer would need the reason to avoid re-suggesting the same thing — never for ephemeral or self-evident reasons) lives entirely inside this skill. Deletion test: inline it and that judgement scatters across the grilling loop, each spot with its own half-remembered version of the rule. **Concentrates.** A real seam, and the reason the domain model can be trusted: all writes go through one door.

**`/codebase-design`** — vocabulary authority throughout, and in phase 3 the owner of the **design-it-twice** parallel sub-agent pattern for exploring alternative interfaces. Deletion test: inline the glossary and vocabulary drifts per-invocation; inline the design-it-twice pattern and interface exploration loses its structure. **Concentrates.** A real seam.

Every seam here is *real* by the skill's own maxim, because a genuinely distinct capability sits behind each — not a hypothetical one you might one day want.

---

## 7. The phase contract and its gates

The pipeline is **Explore → Present → Grill**, and each boundary is a seam because a distinct capability sits behind it.

Phase 1 → 2 crosses the `Explore` subagent seam: raw codebase becomes named findings. Phase 2 → 3 crosses the user-choice seam: the report presents candidates and the run pauses on the explicit question *"Which of these would you like to explore?"* Phase 3 spins the `/grilling` loop, with `/domain-modeling` and `/codebase-design` invoked inline as decisions land.

The load-bearing gate is the **one-way constraint**: phase 2 describes solutions in plain English and *stops* — no interfaces are proposed until phase 3. This is the orchestrator's most important owned rule. It exists because an interface proposed before the design walk is an interface invented without its constraints; the gate forces interface design to happen where the constraints are gathered. The architecture is only ideal if this gate holds — a report that proposes interfaces has collapsed phase 3 into phase 2 and lost the seam between them.

---

## 8. Where the architecture deliberately adds no structure

*"One adapter = hypothetical seam, two = real"* is a rule about restraint, and the ideal architecture obeys it by *not* abstracting where only one implementation exists.

There is one way to explore (the `Explore` subagent) — so there is no "exploration strategy" abstraction. There is one way to present (write a fresh timestamped file to the temp dir and open it cross-platform) — so there is no pluggable report sink, no `PresentationStrategy`. Each is a single concrete adapter, and it stays that way until a second implementation forces a real seam into existence — at which point you will have two implementations to derive the abstraction from, which is the only way to get it right.

The one place the architecture *does* keep two things separate is the vocabulary: domain and architecture glossaries stay two glossaries, because the distinction between *what a thing is called in this business* and *what shape it has structurally* is real, not hypothetical. Two real sources, two homes.

---

## 9. "The interface is the test surface" — for skills

A skill's test surface is its observable output — the thing you inspect to know it worked. The architecture is arranged so those surfaces are made deliberately visible.

The orchestrator's test surface is **the HTML report itself**: you open it and eyeball whether each candidate has the right card (Files, Problem, Solution, Benefits, before/after diagram, strength badge), whether the vocabulary is exact, whether ADR conflicts are flagged sparingly, and whether the Top recommendation is present and justified. The report is not a side effect — it *is* how you evaluate the skill, which is why it is a self-contained file a human looks at rather than a silent internal step.

Each sibling's test surface is likewise its output: `/grilling` is judged by the design decisions it surfaces, `/domain-modeling` by the diffs it makes to `CONTEXT.md` and `docs/adr/`. Designing each skill so its observable output is the thing worth inspecting is what keeps the composition reviewable — you never have to reach inside a skill to know whether it did its job.

---

## 10. Faithful constraints (the skill's verbatim obligations)

The architecture is only correct if it honours every hard constraint the skill states. These are not derived — they are the skill's own words, and any implementation must satisfy them:

- **Temp-only report.** Resolve the temp dir from `$TMPDIR`, falling back to `/tmp` (or `%TEMP%` on Windows); write `architecture-review-<timestamp>.html` so each run is fresh; open it (`xdg-open` / `open` / `start`); tell the user the absolute path. Nothing lands in the repo.
- **CDN rendering.** Tailwind and Mermaid via CDN, self-contained file. Mermaid where relationships are graph-shaped (call graphs, dependencies, sequences); hand-built divs/SVG for editorial visuals (mass diagrams, cross-sections, collapse animations). Every candidate gets a before/after visualisation.
- **Card contents.** Files, Problem, Solution, Benefits (in terms of locality and leverage, and how tests improve), before/after diagram, and a recommendation-strength badge (`Strong` / `Worth exploring` / `Speculative`). Close with a Top recommendation section.
- **No interfaces in phase 2**, then ask *"Which of these would you like to explore?"*
- **ADR conflicts, sparingly.** Only surface a conflict when the friction is real enough to warrant revisiting the ADR, marked with a warning callout carrying the reopen reason. Do not enumerate every refactor an ADR forbids.
- **Lazy `CONTEXT.md`.** Absent on read → proceed; first term worth recording → create it. All domain writes go through `/domain-modeling`.
- **ADR offers only for load-bearing rejections** — framed as recording it so future reviews don't re-suggest the same thing; skip ephemeral and self-evident reasons.
- **Vocabulary discipline.** CONTEXT.md terms for the domain, `/codebase-design` terms for the architecture; never *component / service / API / boundary.*
- **`disable-model-invocation: true`** — this skill is invoked deliberately, never auto-triggered.

---

## Appendix A: the deletion test on the composition

| Element | Inline it into the orchestrator and… | Verdict |
|---|---|---|
| Shared vocabulary (`/codebase-design` glossary) | vocabulary drifts per run; every skill reinvents "what's a seam" | concentrates → keep as authority |
| `CONTEXT.md` (domain model) | domain names re-derived ad hoc each phase | concentrates → keep as single source |
| `HTML-REPORT.md` (scaffold) | orchestrator becomes a markup factory | concentrates → keep separate |
| `Explore` subagent | walk non-determinism spreads into sequencing | concentrates → keep, keep thin |
| `/grilling` | conductor becomes designer; walk logic floods in | concentrates → keep |
| `/domain-modeling` | "is this ADR-worthy?" scatters across the loop | concentrates → keep (the write door) |
| `/codebase-design` design-it-twice | interface exploration loses its structure | concentrates → keep |
| The orchestrator itself | phase sequencing and the one-way gate are lost | concentrates → keep, keep thin |

Every element that stays passes the test the skill applies to the codebases it reviews: removing it would concentrate complexity, not merely move it. That is the mark of a deep module — and here, of a deep skill.

---

## Appendix B: what is implied versus what I inferred

In the spirit of not overstating: the layers, seams, and constraints above are **implied by the skill directly** — the three phases, the delegation to `/grilling`, `/domain-modeling`, and `/codebase-design`, the shared artifacts, the vocabulary discipline, and every constraint in §10 are the skill's own structure and words.

What I **inferred** is the framing that ties them together: treating each skill as a module, each delegation as a seam, and using the skill's own deletion test to justify the composition. The skill prescribes its vocabulary for the suggestions it emits about *other* codebases; turning that same lens on *itself* is the interpretive move — a faithful one, because the skill is a composition of skills and the principles it teaches are exactly the ones that judge such a composition, but a move the skill states nowhere explicitly. It is the reading that best answers *"the ideal architecture implied by the skill,"* and it is an interpretation, not a transcription.
