from __future__ import annotations

import json
import random
import re
from typing import Any

FACETS_LIST = ["problem", "mechanism", "setting", "outcome"]


def build_facet_extraction_prompt(abstract_fr: str, abstract_en: str, body_fr_1500: str) -> str:
    return f"""You are a scientific research analyst. Your task is to extract the 4 core facets and key terminology of the research paper described below.

Abstract (French):
{abstract_fr}

Abstract (English):
{abstract_en}

Body excerpt (French, first 1500 words):
{body_fr_1500}

Instructions:
Extract the following facets in English. Each facet description must be concise and strictly at most 25 words:
1. "problem": the task or problem the paper addresses.
2. "mechanism": the method, algorithm or technical approach.
3. "setting": data, corpus, language, domain, conditions or experimental setup.
4. "outcome": results, findings or the claimed contribution/evaluation.
5. "key_terms": a list of 4 to 8 distinctive technical terms or keywords used in the paper.

Output strictly valid JSON matching this schema with no markdown wrapping or extra commentary:
{{
  "problem": "...",
  "mechanism": "...",
  "setting": "...",
  "outcome": "...",
  "key_terms": ["term1", "term2", "term3", "..."]
}}"""


def build_v0_prompt(facets: dict[str, Any]) -> str:
    return f"""You are a scientific research assistant. Write a clear, natural English research idea description combining all four facets below.
Do not disguise or alter the terminology. Keep it natural and fluent.
Length requirement: at most 80 words.

Facets:
- Problem: {facets.get('problem', '')}
- Mechanism: {facets.get('mechanism', '')}
- Setting: {facets.get('setting', '')}
- Outcome: {facets.get('outcome', '')}

Write only the research idea description text, nothing else."""


def build_v1_prompt(facets: dict[str, Any]) -> str:
    key_terms_str = ", ".join(facets.get("key_terms", []))
    return f"""You are a computer science researcher rewriting a research query under terminology drift and identifier scrubbing.

Original Facets:
- Problem: {facets.get('problem', '')}
- Mechanism: {facets.get('mechanism', '')}
- Setting: {facets.get('setting', '')}
- Outcome: {facets.get('outcome', '')}

Key terms that MUST NOT appear: [{key_terms_str}]

Transformation Rules:
1. Terminology Drift (D1): Rewrite so that NONE of the key terms appear. Use hypernyms, functional descriptions, or vocabulary from an adjacent computing community.
2. Identifier Scrub (D5): Completely remove all proper names, author names, acronyms, dataset/system/benchmark names, and specific numbers. Replace them with generic functional descriptions.
3. Facet Preservation: All four facets (problem, mechanism, setting, outcome) must remain clearly identifiable from the description.
4. Length requirement: between 40 and 80 words. Fluent English only.

Output only the rewritten query text, nothing else."""


def select_v2_facets(seed: int, paper_idx: int) -> list[str]:
    """Select exactly two distinct facets deterministically using seed 13."""
    rng = random.Random(seed + paper_idx * 100)
    return rng.sample(FACETS_LIST, 2)


def build_v2_prompt(facets: dict[str, Any], kept_facets: list[str]) -> str:
    kept_info = {f: facets.get(f, "") for f in kept_facets}
    kept_str = "\n".join(f"- {k.capitalize()}: {v}" for k, v in kept_info.items())
    dropped_str = ", ".join([f for f in FACETS_LIST if f not in kept_facets])
    return f"""You are a computer science researcher writing a research query based on an incomplete subset of facets.

You must include ONLY the following two kept facets:
{kept_str}

(Do NOT include details for the dropped facets: {dropped_str})

Transformation Rules:
1. Facet Subset (D2): Express only the ideas in the kept facets.
2. Identifier Scrub (D5): Remove all proper names, acronyms, dataset/system names, and specific numbers. Replace them with generic descriptions.
3. Length requirement: between 30 and 70 words. Fluent English only.

Output only the query text, nothing else."""


def build_v3_roundtrip_prompt(facets: dict[str, Any]) -> str:
    return f"""You are an expert bilingual scientific translator.
Take the following research idea description and simulate an English-to-German-to-English (EN -> DE -> EN) paraphrasing round trip, followed by identifier scrubbing.

Original Facets:
- Problem: {facets.get('problem', '')}
- Mechanism: {facets.get('mechanism', '')}
- Setting: {facets.get('setting', '')}
- Outcome: {facets.get('outcome', '')}

Instructions:
1. Express the idea as if translated through German syntax and back into English (yielding slightly different phrasing and syntactic shifts).
2. Identifier Scrub (D5): Ensure no proper names, acronyms, benchmark/dataset names, or specific numbers remain.
3. Length: between 40 and 80 words. Fluent English only.

Output only the final English query text, nothing else."""


def build_judge_prompt(query_text: str, target_facets: list[str]) -> str:
    target_str = ", ".join(target_facets)
    return f"""You are an objective peer-review judge evaluating a search query variant generated from a scientific paper.

Candidate Query:
\"{query_text}\"

Target Facets that should be preserved: [{target_str}]

Definitions of Facets:
- "problem": the task or problem addressed.
- "mechanism": the method, algorithm, or technical approach.
- "setting": the data, language, domain, conditions, or experimental setup.
- "outcome": results, findings, or claimed evaluation.

Evaluation Criteria:
1. preserved_facets: List which of the 4 facets (problem, mechanism, setting, outcome) are explicitly or clearly preserved in the query.
2. leaks_identifiers: Boolean (true/false). Set true if the query leaks any proper names, benchmark/dataset acronyms/names (e.g. BERT, GPT-2, TAC 2009, Moodle), or exact numbers. Set false if identifiers are scrubbed.
3. fluent: Boolean (true/false). Set true if the text is fluent, natural English with good grammar.

Output strictly valid JSON with this exact structure:
{{
  "preserved_facets": ["..."],
  "leaks_identifiers": false,
  "fluent": true
}}"""


def lexical_leak(query: str, title: str) -> float:
    """Share of distinct non-stopword title tokens (len>=5) that appear in the query."""
    if not title:
        return 0.0
    toks = {t for t in re.findall(r"\w+", title.lower()) if len(t) >= 5}
    q = set(re.findall(r"\w+", query.lower()))
    return len(toks & q) / max(1, len(toks))

