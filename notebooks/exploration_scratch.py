# %% [markdown]
# # 🛩️ ADREP Causal Graph Reasoning — Event Extraction Brainstorm
#
# **Project:** Causal Graph Reasoning for Explainable ADREP Classification
# **Team:** Whoopie Wanjiru, Theophilus Owiti, Ronnie Delyon (CMU-Africa, MSEAI)
# **Course:** 18-786 Deep Learning
#
# ---
#
# ## The Big Picture (from the proposal)
#
# The existing **SDCPS pipeline** (transformer + multi-LLM consensus) classifies aviation
# safety reports into ICAO ADREP categories, but **26.94% land in "OTHER"** — meaning the
# system couldn't confidently assign a specific incident type (CFIT, LOC-I, etc.).
#
# **Our goal:** Build a **targeted refinement module** that takes ONLY the "OTHER" reports
# and tries to reclassify them by:
# 1. **Extracting structured events** from the narrative text
# 2. **Building causal graphs** (DAGs) from those events
# 3. **Using a GNN** (GAT/GCN) to learn "incident signatures" from the graph
# 4. **Reclassifying** into a specific ADREP category if confidence is high enough
#
# **Today's focus:** Step 1 — Event Extraction. We're testing the schema:
# > **(ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME)**
#
# ---

# %% [markdown]
# ## 1. Load and Explore the Data

# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("data_aviation.csv")

print(f"📊 Dataset shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")
print(f"\n🏷️  Category distribution:")
print(df['final_category'].value_counts())

# %% [markdown]
# **Key Observation:** All 173 rows are labeled **OTHER** — this IS the subset
# of ambiguous reports that the SDCPS pipeline couldn't classify. These are our
# targets for the causal refinement module.

# %%
# Column-level summary
print("=" * 70)
print("COLUMN SUMMARY")
print("=" * 70)
for col in df.columns:
    non_null = df[col].notna().sum()
    non_empty = (df[col].fillna('').astype(str).str.strip().str.len() > 0).sum()
    avg_len = df[col].fillna('').astype(str).str.len().mean()
    print(f"\n📌 {col}")
    print(f"   Non-null: {non_null}/{len(df)}  |  Non-empty: {non_empty}/{len(df)}")
    print(f"   Avg string length: {avg_len:.0f} chars")
    if non_empty > 0 and non_empty < 20:
        unique_vals = df[col].dropna().unique()[:10]
        print(f"   Sample values: {list(unique_vals[:5])}")

# %% [markdown]
# ## 2. Understanding the Existing Columns
#
# The dataset already has structured fields from the original data source:
#
# | Column | What it is | Useful for our event schema? |
# |--------|-----------|----------------------------|
# | `narrative_1` | Full incident narrative text | **PRIMARY INPUT** — extract events from this |
# | `synopsis` | Short summary of incident | Secondary context |
# | `assessments` | Human Factors, Environment, etc. | Maps to **TRIGGER** partially |
# | `events` | Event types (e.g., "Conflict NMAC") | Maps to **OUTCOME** |
# | `events_6` | Response actions taken | Maps to **OUTCOME** (corrective actions) |
# | `phase` | Flight phase (Climb, Cruise, etc.) | **Direct map to PHASE** |
# | `component` | Aircraft system involved | **Direct map to SYSTEM** |
# | `component_4` | System status (Malfunctioning, etc.) | Enriches SYSTEM |
# | `person_1_8` | Human factors (Situational Awareness, etc.) | Maps to **TRIGGER** |

# %%
# Let's look at the existing structured fields to understand what's already there
print("=" * 70)
print("EXISTING PHASE VALUES (maps to our PHASE entity)")
print("=" * 70)
phase_vals = df['phase'].dropna().str.strip()
print(phase_vals.value_counts())

# %%
print("\n" + "=" * 70)
print("EXISTING EVENTS (maps to our OUTCOME entity)")
print("=" * 70)
# Events are semicolon-separated
all_events = df['events'].dropna().str.split(';').explode().str.strip()
print(all_events.value_counts().head(20))

# %%
print("\n" + "=" * 70)
print("EXISTING ASSESSMENTS (maps to our TRIGGER entity)")
print("=" * 70)
all_assessments = df['assessments'].dropna().str.split(';').explode().str.strip()
print(all_assessments.value_counts().head(15))

# %%
print("\n" + "=" * 70)
print("EXISTING PERSON/HUMAN FACTORS (enriches TRIGGER/ACTOR)")
print("=" * 70)
all_person = df['person_1_8'].dropna().str.split(';').explode().str.strip()
print(all_person.value_counts().head(15))

# %%
print("\n" + "=" * 70)
print("EXISTING COMPONENT (maps to SYSTEM)")
print("=" * 70)
print(f"Only {df['component'].notna().sum()} out of {len(df)} reports have component data")
print(df['component'].dropna().value_counts())

# %% [markdown]
# ## 3. Sample Narratives — What We're Working With
#
# Let's look at a few narratives to understand the language and think about
# what our event extractor needs to handle.

# %%
# Pick a few diverse narratives
for idx in [0, 2, 5, 30, 73]:
    if idx < len(df):
        print(f"\n{'='*80}")
        print(f"REPORT #{idx}")
        print(f"Phase: {df.loc[idx, 'phase']}  |  Events: {str(df.loc[idx, 'events'])[:80]}")
        print(f"{'='*80}")
        narrative = df.loc[idx, 'narrative_1']
        print(narrative[:500])
        print("..." if len(narrative) > 500 else "")

# %% [markdown]
# ## 4. 🧠 Brainstorm: Event Extraction Schema
#
# ---
#
# ### Our Schema: `(ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME)`
#
# Let's think about what each entity type REALLY means in aviation context
# and what the challenges are.
#
# ---
#
# ### 4.1 ACTOR — Who is involved?
#
# **Examples from the data:**
# - "Captain", "First Officer", "Flight Instructor", "Student Pilot"
# - "ATC Controller", "Tower Controller", "TRACON Controller"
# - "UAS Pilot", "Remote Pilot"
# - "Helicopter pilot", "The crew"
#
# **Challenges:**
# - Often implicit ("we" = flight crew)
# - Multiple actors in same report
# - Sometimes the actor IS the system ("autopilot descended")
#
# **Extraction approach ideas:**
# - Rule-based: regex for pilot titles, "ATC", crew references
# - NER: fine-tune for aviation-specific person roles
# - LLM prompt: "Who are the actors in this narrative?"
#
# ---
#
# ### 4.2 SYSTEM — What aircraft/equipment system?
#
# **Examples from the data:**
# - "GPS & Other Satellite Navigation", "Communication Systems"
# - "Autopilot", "TCAS", "EGPWS/GPWS"
# - "FMS/FMC", "Altimeter", "ADSB"
# - Aircraft type: "C172", "A320", "UH-60"
#
# **Challenges:**
# - Only 9/173 reports have the `component` field filled!
# - Systems are mentioned implicitly ("the box" = FMS, "scope" = radar display)
# - Aviation jargon: "FLCH mode", "APP mode", "G1000"
#
# **Extraction approach ideas:**
# - Dictionary lookup: build aviation system dictionary
# - NER: tag equipment/systems mentioned in text
# - Hybrid: dictionary + context-based NER
#
# ---
#
# ### 4.3 PHASE — Flight phase when event occurred
#
# **Examples from the data (already structured!):**
# - Takeoff/Launch, Initial Climb, Climb
# - Cruise, Descent
# - Initial Approach, Final Approach, Landing
#
# **Good news:** 170/173 reports already have this field! We can:
# - USE the existing `phase` column directly (low-hanging fruit)
# - ALSO extract phases from text for validation / multi-phase events
#
# **Challenge:** Some reports span multiple phases (e.g., "Takeoff / Launch; Initial Climb")
#
# ---
#
# ### 4.4 TRIGGER — What initiated the event chain?
#
# **Examples from the data:**
# - Human Factors: "Communication Breakdown", "Situational Awareness",
#   "Workload", "Fatigue", "Confusion", "Distraction"
# - Environmental: "Weather", "Icing"
# - Procedural: "Published Material / Policy", "FAR violation"
# - Equipment: "Malfunctioning", "Software and Automation"
#
# **Challenges:**
# - THE HARDEST ENTITY TO EXTRACT from free text
# - Triggers are often described with causal language:
#   "due to", "because of", "resulted from", "caused by"
# - Multiple triggers per event chain
# - Root cause vs. contributing factors
#
# **Extraction approach ideas:**
# - Causal pattern matching: "due to X", "caused by X", "because X"
# - Dependency parsing: find subjects of causal verbs
# - LLM extraction: ask for "initiating events"
# - USE existing `assessments` + `person_1_8` as partial ground truth
#
# ---
#
# ### 4.5 OUTCOME — What happened as a result?
#
# **Examples from the data:**
# - "Conflict NMAC" (Near Mid-Air Collision)
# - "CFTT / CFIT" (Controlled Flight Into Terrain)
# - "Loss Of Aircraft Control"
# - "Unstabilized Approach"
# - "Flight Crew Took Evasive Action"
# - "Flight Crew Executed Go Around"
#
# **Good news:** The `events` and `events_6` columns already capture this!
#
# **Challenge:** Distinguishing between:
# - The BAD outcome (what went wrong)
# - The CORRECTIVE outcome (what was done to fix it)
#
# ---

# %% [markdown]
# ## 5. 🔬 Quick Experiment: Can We Extract Events with Simple Patterns?
#
# Before going to LLMs or fine-tuned NER, let's see what simple rule-based
# extraction gives us. This helps us understand the data better.

# %%
import re

def extract_actors_simple(text):
    """Simple rule-based actor extraction."""
    patterns = [
        r'\b(Captain|First Officer|FO|CA|PF|PM|pilot flying|pilot monitoring)\b',
        r'\b(Flight Instructor|CFI|student pilot|student)\b',
        r'\b(Controller|Tower|TRACON|Approach|Center|CIC)\b',
        r'\b(Flight Attendant|FA\d?)\b',
        r'\b(Dispatcher|RPIC|Remote Pilot)\b',
        r'\b(crew|flight crew)\b',
    ]
    actors = set()
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        actors.update(m.strip() for m in matches)
    return list(actors) if actors else ['Unknown']

def extract_triggers_simple(text):
    """Simple causal pattern matching for triggers."""
    patterns = [
        r'(?:due to|because of|caused by|resulted from|contributing factor)\s+([^.;,]{5,60})',
        r'(?:failure to|failed to)\s+([^.;,]{5,60})',
        r'(?:distracted by|confused by)\s+([^.;,]{5,60})',
    ]
    triggers = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        triggers.extend(m.strip() for m in matches)
    return triggers if triggers else ['Not explicitly stated']

def extract_systems_simple(text):
    """Simple dictionary-based system extraction."""
    systems = {
        'TCAS': 'TCAS', 'EGPWS': 'EGPWS', 'GPWS': 'GPWS',
        'autopilot': 'Autopilot', 'AP': 'Autopilot',
        'FMS': 'FMS', 'FMC': 'FMS', 'G1000': 'Avionics',
        'GPS': 'GPS', 'ADSB': 'ADS-B', 'ADS-B': 'ADS-B',
        'CPDLC': 'CPDLC', 'transponder': 'Transponder',
        'altimeter': 'Altimeter', 'PAPI': 'PAPI', 'VASI': 'VASI',
        'ILS': 'ILS', 'RNAV': 'RNAV', 'LOC': 'Localizer',
        'radio': 'Radio', 'CTAF': 'Radio', 'frequency': 'Radio',
        'drone': 'UAS', 'UAS': 'UAS', 'quadcopter': 'UAS', 'sUAS': 'UAS',
    }
    found = set()
    for keyword, system in systems.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
            found.add(system)
    return list(found) if found else ['Not identified']

# Test on a few reports
print("=" * 80)
print("SIMPLE RULE-BASED EXTRACTION TEST")
print("=" * 80)

for idx in [0, 2, 5, 30, 73]:
    if idx < len(df):
        text = df.loc[idx, 'narrative_1']
        print(f"\n{'─'*60}")
        print(f"Report #{idx} | Phase: {df.loc[idx, 'phase']}")
        print(f"{'─'*60}")
        print(f"Synopsis: {df.loc[idx, 'synopsis'][:120]}...")
        print(f"\n  ACTORS:   {extract_actors_simple(text)}")
        print(f"  SYSTEMS:  {extract_systems_simple(text)}")
        print(f"  PHASE:    {df.loc[idx, 'phase']}")
        print(f"  TRIGGERS: {extract_triggers_simple(text)}")
        print(f"  OUTCOME:  {str(df.loc[idx, 'events'])[:80]}")

# %% [markdown]
# ## 6. Coverage Analysis — How much can we already get?

# %%
# Run extraction on ALL reports and check coverage
results = []
for idx, row in df.iterrows():
    text = row['narrative_1']
    actors = extract_actors_simple(text)
    systems = extract_systems_simple(text)
    triggers = extract_triggers_simple(text)

    results.append({
        'idx': idx,
        'has_actor': actors != ['Unknown'],
        'has_system': systems != ['Not identified'],
        'has_phase': pd.notna(row['phase']),
        'has_trigger': triggers != ['Not explicitly stated'],
        'has_outcome': pd.notna(row['events']),
        'n_actors': len(actors) if actors != ['Unknown'] else 0,
        'n_systems': len(systems) if systems != ['Not identified'] else 0,
        'n_triggers': len(triggers) if triggers != ['Not explicitly stated'] else 0,
    })

coverage_df = pd.DataFrame(results)

print("=" * 60)
print("COVERAGE WITH SIMPLE RULE-BASED EXTRACTION")
print("=" * 60)
print(f"\n  ACTOR found:   {coverage_df['has_actor'].sum():>4}/{len(df)} ({coverage_df['has_actor'].mean()*100:.1f}%)")
print(f"  SYSTEM found:  {coverage_df['has_system'].sum():>4}/{len(df)} ({coverage_df['has_system'].mean()*100:.1f}%)")
print(f"  PHASE found:   {coverage_df['has_phase'].sum():>4}/{len(df)} ({coverage_df['has_phase'].mean()*100:.1f}%)")
print(f"  TRIGGER found: {coverage_df['has_trigger'].sum():>4}/{len(df)} ({coverage_df['has_trigger'].mean()*100:.1f}%)")
print(f"  OUTCOME found: {coverage_df['has_outcome'].sum():>4}/{len(df)} ({coverage_df['has_outcome'].mean()*100:.1f}%)")

all_five = (coverage_df['has_actor'] & coverage_df['has_system'] &
            coverage_df['has_phase'] & coverage_df['has_trigger'] &
            coverage_df['has_outcome'])
print(f"\n  ALL 5 entities: {all_five.sum():>4}/{len(df)} ({all_five.mean()*100:.1f}%)")
print(f"\n  Avg actors/report:   {coverage_df['n_actors'].mean():.1f}")
print(f"  Avg systems/report:  {coverage_df['n_systems'].mean():.1f}")
print(f"  Avg triggers/report: {coverage_df['n_triggers'].mean():.1f}")

# %% [markdown]
# ## 7. 💡 Brainstorm: Key Questions and Decisions
#
# ---
#
# ### Q1: Rule-based vs LLM-based vs NER for event extraction?
#
# | Approach | Pros | Cons | Best for |
# |----------|------|------|----------|
# | **Rule-based** (regex, patterns) | Fast, predictable, no GPU | Low recall, brittle to variations | PHASE, SYSTEM (dictionary) |
# | **LLM prompting** (Gemini/GPT) | High recall, handles nuance | Slow, expensive, inconsistent format | TRIGGER, ACTOR (complex) |
# | **Fine-tuned NER** (BERT-CRF) | Good precision+recall, fast inference | Needs annotated training data! | All entities (if we have labels) |
# | **Hybrid** | Best of both worlds | More complex pipeline | Our likely approach |
#
# **💡 Proposed strategy:**
# - **PHASE:** Use existing column (98% coverage) + text validation
# - **SYSTEM:** Dictionary lookup (easy wins) + LLM for implicit mentions
# - **ACTOR:** Rule-based patterns (high coverage expected) + NER refinement
# - **TRIGGER:** LLM-based extraction (too complex for rules) OR causal patterns
# - **OUTCOME:** Use existing `events` column + merge with `events_6`
#
# ---
#
# ### Q2: What about events that have MULTIPLE of each entity?
#
# Example: Report #2 has TWO aircraft (Aircraft X pilot, Aircraft Y/helicopter),
# MULTIPLE triggers (both pilots failed to communicate), and MULTIPLE outcomes
# (NMAC, evasive action taken).
#
# **Options:**
# - **Flatten:** One event = one tuple (lose multi-factor structure)
# - **Multi-valued:** Allow lists per entity (ACTOR = [Pilot X, Pilot Y])
# - **Multiple events:** Extract N separate events per report
#
# **💡 For the graph: Multiple events → multiple nodes → richer graph structure!**
# This is actually what makes the graph approach powerful.
#
# ---
#
# ### Q3: How do go from events to a CAUSAL GRAPH?
#
# The schema gives us ENTITIES (nodes), but we still need EDGES (causal links).
#
# **Where do edges come from?**
# - TRIGGER → OUTCOME (causal: "ice caused engine failure")
# - ACTOR → ACTION (agency: "pilot took evasive action")
# - PHASE → TRIGGER (context: "during approach, weather deteriorated")
# - SYSTEM → TRIGGER (mechanism: "autopilot disengaged, leading to deviation")
#
# **💡 This is the CRITICAL piece the proposal calls out:**
# > "How will you extract causal edges, not just entities?"
#
# **Options for edge extraction:**
# 1. Template-based: fixed edge types between entity types
# 2. Dependency parsing: find causal verbs linking entities
# 3. LLM-based: prompt for structured triples
# 4. Co-occurrence: entities in same sentence → edge
#
# ---
#
# ### Q4: What's our IMMEDIATE next step?
#
# **Recommended approach — start small:**
#
# 1. ✅ **TODAY:** Load data, understand columns, brainstorm (THIS NOTEBOOK)
# 2. 🔜 **NEXT:** Pick 10-20 reports, manually annotate events as ground truth
# 3. 🔜 **THEN:** Try LLM extraction (Gemini) on those 10-20 to compare
# 4. 🔜 **THEN:** Build first causal graph from extracted events
# 5. 🔜 **LATER:** Scale up, evaluate, iterate
#
# ---

# %% [markdown]
# ## 8. Narrative Length Analysis — Will token limits be an issue?

# %%
# Check narrative lengths
df['narrative_len'] = df['narrative_1'].str.len()
df['synopsis_len'] = df['synopsis'].str.len()

# Rough token estimate (1 token ≈ 4 chars for English)
df['est_tokens'] = df['narrative_len'] / 4

print("Narrative Length Statistics:")
print(f"  Min:    {df['narrative_len'].min():>6} chars ({df['est_tokens'].min():>5.0f} est. tokens)")
print(f"  Mean:   {df['narrative_len'].mean():>6.0f} chars ({df['est_tokens'].mean():>5.0f} est. tokens)")
print(f"  Median: {df['narrative_len'].median():>6.0f} chars ({df['est_tokens'].median():>5.0f} est. tokens)")
print(f"  Max:    {df['narrative_len'].max():>6} chars ({df['est_tokens'].max():>5.0f} est. tokens)")
print(f"\n  Reports > 2000 tokens: {(df['est_tokens'] > 2000).sum()}")
print(f"  Reports > 4000 tokens: {(df['est_tokens'] > 4000).sum()}")

# %%
# Clean up temp columns
df.drop(columns=['narrative_len', 'synopsis_len', 'est_tokens'], inplace=True)

# %% [markdown]
# ## 9. Summary and Next Steps
#
# ### What we know:
# - **173 "OTHER" reports** — all ambiguous, all need reclassification
# - Narratives are **long** (avg ~400 tokens) — rich but complex
# - We already have useful structured data:
#   - `phase` → 98% coverage (use directly for PHASE)
#   - `events` → 100% coverage (use for OUTCOME)
#   - `assessments` → 100% (use as TRIGGER hints)
#   - `person_1_8` → 85% (human factors for TRIGGER)
# - `component` is **very sparse** (only 5%) → SYSTEM must come from text
# - Simple rule-based extraction gives us decent ACTOR/SYSTEM coverage
# - TRIGGER extraction is hard — needs LLM or deep NLP
#
# ### Decision: Go hybrid
# 1. **Use existing columns** for PHASE and OUTCOME (free data!)
# 2. **Rule-based + dictionary** for ACTOR and SYSTEM
# 3. **LLM-based extraction** for TRIGGER (too nuanced for rules)
# 4. **Validate everything** against manually annotated subset (10-20 reports)
#
# ### Immediate TODO:
# - [ ] Pick 10-20 representative reports for manual annotation
# - [ ] Design the LLM prompt for event extraction
# - [ ] Build first causal graph from one report (proof of concept)
# - [ ] Decide on graph representation (NetworkX? PyG HeteroData?)
