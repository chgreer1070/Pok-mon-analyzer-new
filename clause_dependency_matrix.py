"""Utilities for constructing clause dependency matrices.

This module provides a lightweight heuristic parser that breaks natural
language text into clauses and estimates relationships between those clauses.
The resulting relationships are captured inside a square matrix where each
cell ``(i, j)`` stores the strongest dependency inferred between clause ``i``
and clause ``j``.

The implementation favours readability and explainability over linguistic
completeness. It relies on recursive logic to progressively split sentences
into clause fragments and to apply relationship rules until the full matrix is
populated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitRule:
    """Defines how a keyword divides text into clause fragments."""

    keyword: str
    attach_to: str = "right"  # either "left" or "right"


@dataclass(frozen=True)
class ClauseFragment:
    """Intermediate representation used during recursive sentence splitting."""

    text: str
    leading_keyword: Optional[str] = None


@dataclass(frozen=True)
class Clause:
    """Represents a final clause used in the dependency matrix."""

    index: int
    text: str
    trigger: Optional[str] = None


@dataclass
class ClauseDependencyMatrix:
    """Encapsulates the matrix and provides convenience helpers."""

    clauses: Sequence[Clause]
    matrix: List[List[int]]
    relations: Dict[Tuple[int, int], str] = field(default_factory=dict)

    def relation_between(self, origin: int, target: int) -> Optional[str]:
        """Return the named relation between two clauses if it exists."""

        return self.relations.get((origin, target))

    def pretty_print(self) -> str:
        """Return a multiline string showing the matrix alongside clause text."""

        if not self.clauses:
            return "(no clauses detected)"

        headers = [f"C{i}" for i in range(len(self.clauses))]
        header_row = "    " + "  ".join(headers)
        rows = [header_row]
        for idx, clause in enumerate(self.clauses):
            numeric = "  ".join(f"{value:>2}" for value in self.matrix[idx])
            rows.append(f"C{idx}: {numeric}  | {clause.text}")
        return "\n".join(rows)

    def strongest_relations(self, limit: int = 5) -> List[Tuple[int, int, str, int]]:
        """Return strongest relations sorted by weight descending.

        Parameters
        ----------
        limit:
            Maximum number of relations to include in the result.
        """

        if limit <= 0:
            return []

        ranked: List[Tuple[int, int, str, int]] = [
            (origin, target, relation, self.matrix[origin][target])
            for (origin, target), relation in self.relations.items()
        ]
        ranked.sort(key=lambda item: (-item[3], item[0], item[1]))
        return ranked[:limit]

    def to_dict(self, top_n: int = 5) -> Dict[str, Any]:
        """Serialize the matrix and metadata to a JSON-friendly dictionary."""

        return {
            "clauses": [
                {"index": clause.index, "text": clause.text, "trigger": clause.trigger}
                for clause in self.clauses
            ],
            "matrix": self.matrix,
            "relations": [
                {
                    "origin": origin,
                    "target": target,
                    "relation": relation,
                    "weight": self.matrix[origin][target],
                }
                for (origin, target), relation in sorted(self.relations.items())
            ],
            "top_relations": [
                {
                    "origin": origin,
                    "target": target,
                    "relation": relation,
                    "weight": weight,
                }
                for origin, target, relation, weight in self.strongest_relations(limit=top_n)
            ],
        }


# ---------------------------------------------------------------------------
# Configuration: rule definitions and weights
# ---------------------------------------------------------------------------


SPLIT_RULES: Sequence[SplitRule] = (
    SplitRule("because", "right"),
    SplitRule("since", "right"),
    SplitRule("as", "right"),
    SplitRule("if", "right"),
    SplitRule("when", "right"),
    SplitRule("while", "right"),
    SplitRule("although", "right"),
    SplitRule("though", "right"),
    SplitRule("unless", "right"),
    SplitRule("until", "right"),
    SplitRule("whereas", "right"),
    SplitRule("but", "right"),
    SplitRule("however", "right"),
    SplitRule("yet", "right"),
    SplitRule("so", "right"),
    SplitRule("therefore", "right"),
    SplitRule("thus", "right"),
    SplitRule("hence", "right"),
    SplitRule("and", "right"),
    SplitRule("which", "right"),
    SplitRule("that", "right"),
)


RELATION_WEIGHTS: Dict[str, int] = {
    "sequence": 1,
    "elaboration": 2,
    "condition": 3,
    "contrast": 4,
    "cause": 5,
    "result": 6,
}


TRIGGER_RULES: Dict[str, Tuple[str, str]] = {
    "if": ("condition", "forward"),
    "when": ("condition", "forward"),
    "while": ("condition", "forward"),
    "unless": ("condition", "forward"),
    "until": ("condition", "forward"),
    "because": ("cause", "backward"),
    "since": ("cause", "backward"),
    "as": ("cause", "backward"),
    "so": ("result", "backward"),
    "therefore": ("result", "backward"),
    "thus": ("result", "backward"),
    "hence": ("result", "backward"),
    "although": ("contrast", "forward"),
    "though": ("contrast", "forward"),
    "whereas": ("contrast", "backward"),
    "but": ("contrast", "backward"),
    "however": ("contrast", "backward"),
    "yet": ("contrast", "backward"),
    "and": ("sequence", "backward"),
    "which": ("elaboration", "backward"),
    "that": ("elaboration", "backward"),
}


# ---------------------------------------------------------------------------
# Clause extraction helpers (recursive logic)
# ---------------------------------------------------------------------------


def _recursive_clause_split(text: str, leading_keyword: Optional[str] = None) -> List[ClauseFragment]:
    """Split text into clause fragments using recursive keyword rules."""

    text = text.strip(" ,;:\n\t")
    if not text:
        return []

    lowered = text.lower()
    for rule in SPLIT_RULES:
        pattern = re.compile(rf"\b{re.escape(rule.keyword)}\b", re.IGNORECASE)
        match = pattern.search(lowered)
        if match:
            start, end = match.span()
            left = text[:start]
            right = text[end:]
            fragments: List[ClauseFragment] = []
            if left.strip():
                next_leading = rule.keyword if rule.attach_to == "left" else leading_keyword
                fragments.extend(_recursive_clause_split(left, next_leading))
            if right.strip():
                next_leading = rule.keyword if rule.attach_to == "right" else leading_keyword
                fragments.extend(_recursive_clause_split(right, next_leading))
            return fragments

    for delimiter in (",", ";", ":"):
        if delimiter in text:
            left, right = text.split(delimiter, 1)
            fragments = []
            if left.strip():
                fragments.extend(_recursive_clause_split(left, leading_keyword))
            if right.strip():
                fragments.extend(_recursive_clause_split(right, None))
            return fragments

    return [ClauseFragment(text=text, leading_keyword=leading_keyword)]


def _tokenize_clauses(text: str) -> List[Clause]:
    """Break a block of text into ordered clauses."""

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    fragments: List[ClauseFragment] = []
    for sentence in sentences:
        fragments.extend(_recursive_clause_split(sentence))

    clauses: List[Clause] = []
    for index, fragment in enumerate(fragment for fragment in fragments if fragment.text):
        clause_text = fragment.text.strip()
        trigger = fragment.leading_keyword.lower() if fragment.leading_keyword else None
        clauses.append(Clause(index=index, text=clause_text, trigger=trigger))
    return clauses


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------


def _set_relation(
    matrix: List[List[int]],
    relations: Dict[Tuple[int, int], str],
    origin: int,
    target: int,
    relation: str,
) -> None:
    weight = RELATION_WEIGHTS[relation]
    if matrix[origin][target] == 0 or weight >= matrix[origin][target]:
        matrix[origin][target] = weight
        relations[(origin, target)] = relation


def _apply_trigger_relations(
    clauses: Sequence[Clause],
    matrix: List[List[int]],
    relations: Dict[Tuple[int, int], str],
    index: int,
) -> None:
    if index >= len(clauses):
        return

    clause = clauses[index]
    if clause.trigger and clause.trigger in TRIGGER_RULES:
        relation, direction = TRIGGER_RULES[clause.trigger]
        if direction == "forward" and index + 1 < len(clauses):
            _set_relation(matrix, relations, index, index + 1, relation)
        elif direction == "backward" and index - 1 >= 0:
            _set_relation(matrix, relations, index, index - 1, relation)
        elif direction == "bidirectional" and index - 1 >= 0:
            _set_relation(matrix, relations, index, index - 1, relation)
            _set_relation(matrix, relations, index - 1, index, relation)

    if index + 1 < len(clauses):
        _set_relation(matrix, relations, index, index + 1, "sequence")

    _apply_trigger_relations(clauses, matrix, relations, index + 1)


def build_clause_dependency_matrix(text: str) -> ClauseDependencyMatrix:
    """Return a dependency matrix built from ``text``."""

    clauses = _tokenize_clauses(text)
    size = len(clauses)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    relations: Dict[Tuple[int, int], str] = {}

    _apply_trigger_relations(clauses, matrix, relations, 0)

    return ClauseDependencyMatrix(clauses=clauses, matrix=matrix, relations=relations)


def summarize_dependencies(matrix: ClauseDependencyMatrix, top_n: int = 5) -> str:
    """Return a concise natural-language summary for a matrix result."""

    if not matrix.clauses:
        return "No clauses were detected."

    clause_count = len(matrix.clauses)
    relation_count = len(matrix.relations)
    summary_lines = [
        f"Detected {clause_count} clauses and {relation_count} directed relations.",
    ]

    ranked = matrix.strongest_relations(limit=top_n)
    if ranked:
        summary_lines.append("Strongest inferred dependencies:")
        for origin, target, relation, weight in ranked:
            origin_text = matrix.clauses[origin].text
            target_text = matrix.clauses[target].text
            summary_lines.append(
                f"- C{origin} → C{target}: {relation} (weight {weight}) "
                f"[{origin_text!r} -> {target_text!r}]"
            )
    else:
        summary_lines.append("No explicit dependency triggers were identified.")

    return "\n".join(summary_lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clause dependency matrix from text.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="Raw text to analyze.")
    source_group.add_argument("--file", help="Path to a UTF-8 text file to analyze.")
    parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of strongest relations to include in summaries.",
    )
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be zero or greater.")
    return args


def _load_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text

    with open(args.file, "r", encoding="utf-8") as handle:
        return handle.read()


def _main() -> None:
    args = _parse_args()
    text = _load_input_text(args)
    matrix = build_clause_dependency_matrix(text)

    if args.format == "json":
        payload = matrix.to_dict(top_n=args.top)
        payload["summary"] = summarize_dependencies(matrix, top_n=args.top)
        print(json.dumps(payload, indent=2))
        return

    print(matrix.pretty_print())
    print("\n" + summarize_dependencies(matrix, top_n=args.top))


if __name__ == "__main__":  # pragma: no cover - command-line entrypoint
    _main()
