from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from human_eval.rubric import (
    RubricEvaluator,
    disagreement_triggers,
    quadratic_weighted_kappa,
)
from services.human_eval_store import SQLiteHumanEvalStore


class HumanEvalService:
    _BLIND_FIELDS = (
        "query",
        "turns",
        "scene",
        "risk_level",
        "capability_tags",
        "context",
        "agent_answer",
        "approval_records",
        "planner_steps",
        "tool_calls",
        "trace",
        "evidence",
        "references",
        "policy_context",
    )

    def __init__(
        self,
        store: SQLiteHumanEvalStore,
        rubric: Optional[RubricEvaluator] = None,
        metrics_path: Path | str = "config/evaluation_metrics.yml",
    ) -> None:
        self.store = store
        self.rubric = rubric or RubricEvaluator()
        self.metrics = yaml.safe_load(Path(metrics_path).read_text(encoding="utf-8"))

    def create_batch(
        self,
        *,
        tenant_id: str,
        created_by: str,
        name: str,
        dataset_version: str,
        items: Sequence[Mapping[str, Any]],
        reviewer_ids: Sequence[str],
        qc_rate: float = 0.10,
        seed: int = 20260811,
    ) -> Dict[str, Any]:
        if not name.strip():
            raise ValueError("batch name is required")
        if not dataset_version.strip():
            raise ValueError("dataset_version is required")
        if not items:
            raise ValueError("at least one evaluation item is required")
        reviewers = [str(value).strip() for value in reviewer_ids if str(value).strip()]
        if len(reviewers) != len(set(reviewers)) or len(reviewers) < 2:
            raise ValueError("at least two unique reviewers are required")
        if not 0 <= qc_rate <= 1:
            raise ValueError("qc_rate must be between 0 and 1")

        prepared_items = self._prepare_items(items, qc_rate=qc_rate, seed=seed)
        assignments = self._assign_reviewers(prepared_items, reviewers, seed=seed)
        batch_id = self.store.create_batch(
            tenant_id=tenant_id,
            name=name.strip(),
            dataset_version=dataset_version.strip(),
            rubric_version=self.rubric.version,
            assignments_per_item=2,
            qc_rate=qc_rate,
            seed=seed,
            created_by=created_by,
            items=prepared_items,
            assignments=assignments,
        )
        return self.progress(batch_id=batch_id, tenant_id=tenant_id)

    def next_task(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        reviewer_id: str,
    ) -> Optional[Dict[str, Any]]:
        task = self.store.claim_next(
            batch_id=batch_id,
            tenant_id=tenant_id,
            reviewer_id=reviewer_id,
        )
        if task is None:
            return None
        return {**task, "rubric": self.rubric.public_definition()}

    def submit(
        self,
        *,
        assignment_id: str,
        tenant_id: str,
        reviewer_id: str,
        submission: Mapping[str, Any],
    ) -> Dict[str, Any]:
        evaluated = self.rubric.evaluate(submission)
        return self.store.submit_annotation(
            assignment_id=assignment_id,
            tenant_id=tenant_id,
            reviewer_id=reviewer_id,
            payload=evaluated.payload,
            overall_score=evaluated.overall_score,
            passed=evaluated.passed,
        )

    def progress(self, *, batch_id: str, tenant_id: str) -> Dict[str, Any]:
        bundle = self.store.batch_bundle(batch_id, tenant_id)
        batch = bundle["batch"]
        assignments = [assignment for item in bundle["items"] for assignment in item["assignments"]]
        assignment_status = Counter(item["status"] for item in assignments)
        item_count = len(bundle["items"])
        completed_items = sum(
            bool(item["assignments"])
            and all(assignment["status"] == "submitted" for assignment in item["assignments"])
            for item in bundle["items"]
        )
        disagreement_items = self._disagreement_items(bundle)
        unadjudicated = sum(not item["adjudication"] for item in disagreement_items)
        qc_status = Counter(
            item["qc_status"] for item in bundle["items"] if item["qc_selected"]
        )
        submitted = assignment_status.get("submitted", 0)
        return {
            "batch_id": batch_id,
            "name": batch["name"],
            "dataset_version": batch["dataset_version"],
            "rubric_version": batch["rubric_version"],
            "status": batch["status"],
            "item_count": item_count,
            "completed_items": completed_items,
            "assignment_count": len(assignments),
            "submitted_assignments": submitted,
            "assignment_completion_rate": round(submitted / len(assignments), 4)
            if assignments
            else 0.0,
            "assignment_status": dict(sorted(assignment_status.items())),
            "disagreement_count": len(disagreement_items),
            "pending_adjudication_count": unadjudicated,
            "qc_selected_count": sum(qc_status.values()),
            "qc_status": dict(sorted(qc_status.items())),
            "created_at": batch["created_at"],
            "closed_at": batch["closed_at"],
        }

    def disagreements(self, *, batch_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        bundle = self.store.batch_bundle(batch_id, tenant_id)
        output = []
        for item in self._disagreement_items(bundle):
            output.append(
                {
                    "item_id": item["item_id"],
                    "case_id": item["case_id"],
                    "payload": item["blind_payload"],
                    "oracle_context": item["oracle_payload"],
                    "triggers": item["disagreement_triggers"],
                    "adjudicated": item["adjudication"] is not None,
                    "annotations": [
                        self._public_annotation(
                            assignment["annotation"],
                            reviewer_alias=self._reviewer_alias(
                                batch_id, assignment["reviewer_id"]
                            ),
                        )
                        for assignment in item["assignments"]
                    ],
                }
            )
        return output

    def adjudicate(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        item_id: str,
        adjudicator_id: str,
        submission: Mapping[str, Any],
    ) -> Dict[str, Any]:
        bundle = self.store.batch_bundle(batch_id, tenant_id)
        item = next((item for item in bundle["items"] if item["item_id"] == item_id), None)
        if item is None:
            raise KeyError(item_id)
        annotations = [
            assignment["annotation"]
            for assignment in item["assignments"]
            if assignment["annotation"] is not None
        ]
        if len(annotations) != 2:
            raise ValueError("two current annotations are required before adjudication")
        triggers = disagreement_triggers(annotations[0], annotations[1])
        if not triggers:
            raise ValueError("adjudication is only allowed for a disagreement item")
        if adjudicator_id in {
            assignment["reviewer_id"] for assignment in item["assignments"]
        }:
            raise ValueError("adjudicator must be independent from both reviewers")
        evaluated = self.rubric.evaluate(submission)
        return self.store.save_adjudication(
            batch_id=batch_id,
            tenant_id=tenant_id,
            item_id=item_id,
            adjudicator_id=adjudicator_id,
            payload=evaluated.payload,
            overall_score=evaluated.overall_score,
            passed=evaluated.passed,
            triggers=triggers,
        )

    def qc_queue(self, *, batch_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        bundle = self.store.batch_bundle(batch_id, tenant_id)
        return [
            {
                "item_id": item["item_id"],
                "case_id": item["case_id"],
                "payload": item["blind_payload"],
                "oracle_context": item["oracle_payload"],
                "qc_status": item["qc_status"],
                "assignments": [
                    {
                        "assignment_id": assignment["assignment_id"],
                        "reviewer": self._reviewer_alias(
                            batch_id, assignment["reviewer_id"]
                        ),
                        "status": assignment["status"],
                        "revision": assignment["revision"],
                        "annotation": self._public_annotation(
                            assignment["annotation"],
                            reviewer_alias=self._reviewer_alias(
                                batch_id, assignment["reviewer_id"]
                            ),
                        ),
                    }
                    for assignment in item["assignments"]
                ],
            }
            for item in bundle["items"]
            if item["qc_selected"] and item["qc_status"] in {"pending", "returned"}
        ]

    def review_qc(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        item_id: str,
        reviewer_id: str,
        decision: str,
        note: str,
        returned_assignments: Sequence[str] = (),
    ) -> Dict[str, Any]:
        if not note.strip():
            raise ValueError("QC review requires a note")
        return self.store.record_qc(
            batch_id=batch_id,
            tenant_id=tenant_id,
            item_id=item_id,
            reviewer_id=reviewer_id,
            decision=decision,
            note=note,
            returned_assignments=returned_assignments,
        )

    def report(self, *, batch_id: str, tenant_id: str) -> Dict[str, Any]:
        bundle = self.store.batch_bundle(batch_id, tenant_id)
        pairs = []
        all_annotations = []
        final_labels = []
        reviewer_stats: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        disagreements = self._disagreement_items(bundle)
        trigger_counts = Counter(
            trigger for item in disagreements for trigger in item["disagreement_triggers"]
        )
        for item in bundle["items"]:
            annotations = [
                assignment["annotation"]
                for assignment in item["assignments"]
                if assignment["annotation"] is not None
            ]
            for assignment in item["assignments"]:
                if assignment["annotation"] is not None:
                    alias = self._reviewer_alias(batch_id, assignment["reviewer_id"])
                    reviewer_stats[alias].append(assignment["annotation"])
            all_annotations.extend(annotations)
            if len(annotations) == 2:
                pairs.append(tuple(annotations))
                triggers = disagreement_triggers(annotations[0], annotations[1])
                if item["adjudication"] is not None:
                    final_labels.append(item["adjudication"])
                elif not triggers:
                    final_labels.append(self._dual_consensus(annotations))

        dimension_metrics = {}
        for dimension_id in self.rubric.dimension_ids:
            score_pairs = [
                (first["scores"][dimension_id], second["scores"][dimension_id])
                for first, second in pairs
                if first["valid"]
                and second["valid"]
                and first["scores"][dimension_id] is not None
                and second["scores"][dimension_id] is not None
            ]
            flattened = [score for pair in score_pairs for score in pair]
            dimension_metrics[dimension_id] = {
                "pair_count": len(score_pairs),
                "weighted_kappa": quadratic_weighted_kappa(score_pairs),
                "exact_agreement_rate": round(
                    sum(first == second for first, second in score_pairs) / len(score_pairs), 4
                )
                if score_pairs
                else None,
                "mean_score": round(mean(flattened), 4) if flattened else None,
            }

        core_dimensions = self._metric("weighted_kappa").get("core_dimensions", [])
        core_kappas = [
            dimension_metrics[dimension_id]["weighted_kappa"]
            for dimension_id in core_dimensions
            if dimension_metrics[dimension_id]["weighted_kappa"] is not None
        ]
        valid_pairs = [pair for pair in pairs if pair[0]["valid"] and pair[1]["valid"]]
        veto_agreement = [
            set(first["vetoes"]) == set(second["vetoes"]) for first, second in pairs
            if first["valid"] and second["valid"]
        ]
        pass_agreement = [
            first["passed"] == second["passed"] for first, second in valid_pairs
        ]
        invalid_annotations = sum(not item["valid"] for item in all_annotations)
        invalid_cases = sum(not item["valid"] for item in final_labels)
        independent_overall_scores = [
            float(item["overall_score"])
            for item in all_annotations
            if item["overall_score"] is not None
        ]
        valid_final_labels = [item for item in final_labels if item["valid"]]
        final_overall_scores = [
            float(item["overall_score"])
            for item in valid_final_labels
            if item["overall_score"] is not None
        ]
        safety_vetoes = {
            veto_id
            for veto_id, rule in self.rubric.veto_rules.items()
            if "safety" in rule["forces_scores"]
        }
        confidence = Counter(item["confidence"] for item in all_annotations)
        minimum_pairs = int(self.metrics["conventions"].get("minimum_gated_group_size", 20))
        kappa_metric = self._metric("weighted_kappa")
        veto_metric = self._metric("veto_agreement_rate")
        invalid_metric = self._metric("invalid_case_rate")
        macro_kappa = round(mean(core_kappas), 4) if core_kappas else None
        minimum_core_kappa = min(core_kappas) if core_kappas else None
        veto_agreement_rate = (
            round(sum(veto_agreement) / len(veto_agreement), 4)
            if veto_agreement
            else None
        )
        invalid_annotation_rate = (
            round(invalid_annotations / len(all_annotations), 4) if all_annotations else None
        )
        invalid_rate = (
            round(invalid_cases / len(final_labels), 4) if final_labels else None
        )
        if len(pairs) < minimum_pairs:
            gate = {
                "status": "insufficient_sample",
                "minimum_pair_count": minimum_pairs,
                "observed_pair_count": len(pairs),
                "passed": None,
                "failures": [],
            }
        else:
            failures = []
            if macro_kappa is None or macro_kappa < float(kappa_metric["gate"]["threshold"]):
                failures.append("macro_weighted_kappa_below_threshold")
            minimum_threshold = float(
                kappa_metric["gate"]["minimum_per_core_dimension"]
            )
            if minimum_core_kappa is None or minimum_core_kappa < minimum_threshold:
                failures.append("core_dimension_kappa_below_threshold")
            if veto_agreement_rate is None or veto_agreement_rate < float(
                veto_metric["gate"]["threshold"]
            ):
                failures.append("veto_agreement_below_threshold")
            if invalid_rate is None or invalid_rate >= float(
                invalid_metric["gate"]["threshold"]
            ):
                failures.append("invalid_case_rate_above_threshold")
            gate = {
                "status": "evaluated",
                "minimum_pair_count": minimum_pairs,
                "observed_pair_count": len(pairs),
                "passed": not failures,
                "failures": failures,
            }

        return {
            "batch": self.progress(batch_id=batch_id, tenant_id=tenant_id),
            "pair_count": len(pairs),
            "annotation_count": len(all_annotations),
            "dimension_metrics": dimension_metrics,
            "core_weighted_kappa_macro": macro_kappa,
            "minimum_core_dimension_kappa": minimum_core_kappa,
            "veto_agreement_rate": veto_agreement_rate,
            "pass_fail_agreement_rate": round(sum(pass_agreement) / len(pass_agreement), 4)
            if pass_agreement
            else None,
            "invalid_case_rate": invalid_rate,
            "independent_invalid_annotation_rate": invalid_annotation_rate,
            "resolved_case_count": len(final_labels),
            "pending_final_count": len(bundle["items"]) - len(final_labels),
            "human_overall_score": round(mean(final_overall_scores), 4)
            if final_overall_scores
            else None,
            "human_case_pass_rate": round(
                sum(bool(item["passed"]) for item in valid_final_labels)
                / len(valid_final_labels),
                4,
            )
            if valid_final_labels
            else None,
            "critical_veto_rate": round(
                sum(bool(item["vetoes"]) for item in valid_final_labels)
                / len(valid_final_labels),
                4,
            )
            if valid_final_labels
            else None,
            "safety_compliance_rate": round(
                sum(
                    item["scores"]["safety"] == 3
                    and not (set(item["vetoes"]) & safety_vetoes)
                    for item in valid_final_labels
                )
                / len(valid_final_labels),
                4,
            )
            if valid_final_labels
            else None,
            "adjudication_rate": round(
                sum(item["adjudication"] is not None for item in bundle["items"])
                / len(pairs),
                4,
            )
            if pairs
            else None,
            "independent_mean_overall_score": round(
                mean(independent_overall_scores), 4
            )
            if independent_overall_scores
            else None,
            "confidence_distribution": dict(sorted(confidence.items())),
            "disagreement_count": len(disagreements),
            "disagreement_trigger_counts": dict(sorted(trigger_counts.items())),
            "adjudicated_disagreement_count": sum(
                item["adjudication"] is not None for item in disagreements
            ),
            "reviewer_metrics": {
                reviewer: {
                    "annotation_count": len(annotations),
                    "mean_duration_seconds": round(
                        mean(item["duration_seconds"] for item in annotations), 3
                    ),
                    "invalid_count": sum(not item["valid"] for item in annotations),
                }
                for reviewer, annotations in sorted(reviewer_stats.items())
            },
            "quality_gate": gate,
            "note": "Agreement is calculated from independent current annotations; adjudication does not rewrite reviewer scores.",
        }

    def export(self, *, batch_id: str, tenant_id: str) -> Dict[str, Any]:
        bundle = self.store.batch_bundle(batch_id, tenant_id)
        records = []
        pending = 0
        for item in bundle["items"]:
            annotations = [
                assignment["annotation"]
                for assignment in item["assignments"]
                if assignment["annotation"] is not None
            ]
            triggers = (
                disagreement_triggers(annotations[0], annotations[1])
                if len(annotations) == 2
                else ["annotations_incomplete"]
            )
            final = None
            if item["adjudication"] is not None:
                final = {
                    "source": "adjudication",
                    **{
                        key: value
                        for key, value in item["adjudication"].items()
                        if key not in {"adjudicator_id"}
                    },
                }
            elif len(annotations) == 2 and not triggers:
                final = self._dual_consensus(annotations)
            else:
                pending += 1
            records.append(
                {
                    "case_id": item["case_id"],
                    "payload": item["blind_payload"],
                    "qc_status": item["qc_status"],
                    "disagreement_triggers": triggers,
                    "independent_annotations": [
                        self._public_annotation(
                            assignment["annotation"],
                            reviewer_alias=self._reviewer_alias(
                                batch_id, assignment["reviewer_id"]
                            ),
                        )
                        for assignment in item["assignments"]
                        if assignment["annotation"] is not None
                    ],
                    "final": final,
                }
            )
        return {
            "schema_version": 1,
            "batch_id": batch_id,
            "dataset_version": bundle["batch"]["dataset_version"],
            "rubric_version": bundle["batch"]["rubric_version"],
            "batch_status": bundle["batch"]["status"],
            "record_count": len(records),
            "pending_final_count": pending,
            "records": records,
        }

    def close_batch(self, *, batch_id: str, tenant_id: str, actor_id: str) -> Dict[str, Any]:
        progress = self.progress(batch_id=batch_id, tenant_id=tenant_id)
        failures = []
        if progress["submitted_assignments"] != progress["assignment_count"]:
            failures.append("assignments_incomplete")
        qc_status = progress["qc_status"]
        if any(status != "accepted" and count for status, count in qc_status.items()):
            failures.append("qc_incomplete")
        if progress["pending_adjudication_count"]:
            failures.append("adjudications_incomplete")
        if failures:
            raise ValueError(f"batch cannot close: {', '.join(failures)}")
        self.store.close_batch(batch_id, tenant_id, actor_id)
        return self.progress(batch_id=batch_id, tenant_id=tenant_id)

    def audit_events(self, *, batch_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        return self.store.audit_events(batch_id, tenant_id)

    def _prepare_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        qc_rate: float,
        seed: int,
    ) -> List[Dict[str, Any]]:
        case_ids = [str(item.get("case_id") or "").strip() for item in items]
        if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
            raise ValueError("items require unique non-empty case_id values")
        qc_count = math.ceil(len(items) * qc_rate) if qc_rate > 0 else 0
        rng = random.Random(seed)
        qc_indices = set(rng.sample(range(len(items)), qc_count)) if qc_count else set()
        prepared = []
        for ordinal, item in enumerate(items):
            query = item.get("query")
            answer = item.get("agent_answer")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{case_ids[ordinal]}: query is required")
            if not isinstance(answer, str):
                raise ValueError(f"{case_ids[ordinal]}: agent_answer must be a string")
            blind_payload = {
                field: self._sanitize_blind(item[field])
                for field in self._BLIND_FIELDS
                if field in item
            }
            blind_payload["query"] = query
            blind_payload["agent_answer"] = answer
            prepared.append(
                {
                    "case_id": case_ids[ordinal],
                    "blind_payload": blind_payload,
                    "oracle_payload": dict(item.get("oracle") or item.get("expected") or {}),
                    "ordinal": ordinal,
                    "qc_selected": ordinal in qc_indices,
                }
            )
        return prepared

    @staticmethod
    def _assign_reviewers(
        items: Sequence[Mapping[str, Any]], reviewer_ids: Sequence[str], *, seed: int
    ) -> List[Dict[str, Any]]:
        reviewers = list(reviewer_ids)
        random.Random(seed).shuffle(reviewers)
        assignments = []
        reviewer_count = len(reviewers)
        for index, item in enumerate(items):
            first_index = index % reviewer_count
            if reviewer_count == 2:
                second_index = 1 - first_index
            else:
                second_index = (index + 1 + index // reviewer_count) % reviewer_count
                if second_index == first_index:
                    second_index = (second_index + 1) % reviewer_count
            for slot, reviewer_index in enumerate((first_index, second_index), start=1):
                assignments.append(
                    {
                        "case_id": item["case_id"],
                        "reviewer_id": reviewers[reviewer_index],
                        "reviewer_slot": slot,
                    }
                )
        return assignments

    @classmethod
    def _sanitize_blind(cls, value: Any) -> Any:
        hidden_keys = {
            "model",
            "model_name",
            "model_version",
            "provider",
            "experiment",
            "experiment_group",
            "variant",
            "expected",
            "oracle",
            "developer_label",
        }
        if isinstance(value, Mapping):
            return {
                str(key): cls._sanitize_blind(item)
                for key, item in value.items()
                if str(key).casefold() not in hidden_keys
            }
        if isinstance(value, list):
            return [cls._sanitize_blind(item) for item in value]
        return value

    def _disagreement_items(self, bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
        output = []
        minimum_gap = int(
            self.rubric.config["adjudication_triggers"]["minimum_dimension_score_gap"]
        )
        for item in bundle["items"]:
            annotations = [
                assignment["annotation"]
                for assignment in item["assignments"]
                if assignment["annotation"] is not None
            ]
            if len(annotations) != 2:
                continue
            triggers = disagreement_triggers(
                annotations[0], annotations[1], minimum_score_gap=minimum_gap
            )
            if triggers:
                output.append({**item, "disagreement_triggers": triggers})
        return output

    def _dual_consensus(self, annotations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        first, second = annotations
        if not first["valid"]:
            invalid_reason = first["invalid_reason"]
            if invalid_reason != second["invalid_reason"]:
                raise ValueError("invalid reason disagreement requires adjudication")
            return {
                "source": "dual_consensus",
                "valid": False,
                "invalid_reason": invalid_reason,
                "scores": {dimension_id: None for dimension_id in self.rubric.dimension_ids},
                "vetoes": [],
                "overall_score": None,
                "passed": None,
            }
        scores = {}
        for dimension_id in self.rubric.dimension_ids:
            left = first["scores"][dimension_id]
            right = second["scores"][dimension_id]
            if left is None or right is None:
                scores[dimension_id] = None
            else:
                scores[dimension_id] = round((left + right) / 2, 4)
        overall = self.rubric.overall_score(scores)
        return {
            "source": "dual_mean",
            "valid": True,
            "invalid_reason": None,
            "scores": scores,
            "vetoes": list(first["vetoes"]),
            "overall_score": overall,
            "passed": self.rubric.case_passed(scores, first["vetoes"], overall_score=overall),
        }

    def _metric(self, metric_id: str) -> Dict[str, Any]:
        return next(item for item in self.metrics["metrics"] if item["id"] == metric_id)

    @staticmethod
    def _reviewer_alias(batch_id: str, reviewer_id: str) -> str:
        digest = hashlib.sha256(f"{batch_id}:{reviewer_id}".encode("utf-8")).hexdigest()
        return f"reviewer-{digest[:8]}"

    @staticmethod
    def _public_annotation(
        annotation: Optional[Mapping[str, Any]], *, reviewer_alias: str
    ) -> Optional[Dict[str, Any]]:
        if annotation is None:
            return None
        hidden = {"reviewer_id", "annotation_id"}
        return {
            "reviewer": reviewer_alias,
            **{key: value for key, value in annotation.items() if key not in hidden},
        }
