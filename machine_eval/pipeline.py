from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from human_eval.rubric import RubricEvaluator, quadratic_weighted_kappa
from machine_eval.deterministic import evaluate_deterministic
from machine_eval.judge import AgentRubricJudge, JudgeResult


class MachineEvalPipeline:
    def __init__(
        self,
        *,
        judge: Optional[AgentRubricJudge] = None,
        rubric: Optional[RubricEvaluator] = None,
        config_path: Path | str = "config/machine_evaluation.yml",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.rubric = rubric or RubricEvaluator()
        self.judge = judge
        if self.config["rubric_version"] != self.rubric.version:
            raise ValueError("machine evaluation and Rubric versions do not match")

    def evaluate(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        human_export: Optional[Mapping[str, Any]] = None,
        baseline: Optional[Mapping[str, Any]] = None,
        run_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._validate_items(items)
        workers = max(1, int(self.config["judge"].get("maximum_workers", 1)))
        if self.judge is None or workers == 1:
            case_results = [self._evaluate_case(item) for item in items]
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="machine-eval-case"
            ) as executor:
                case_results = list(executor.map(self._evaluate_case, items))
        summary = self._summary(case_results)
        human_alignment = self._human_alignment(
            case_results, human_export=human_export
        )
        slices = self._slices(case_results, human_export=human_export)
        review_queue = self._review_queue(
            case_results, human_export=human_export
        )
        gate = self._gate(summary, human_alignment, human_export)
        baseline_comparison = self._compare_baseline(human_alignment, baseline)
        if baseline_comparison and not baseline_comparison["passed"]:
            gate["failures"].extend(baseline_comparison["failures"])
            gate["failures"] = list(dict.fromkeys(gate["failures"]))
            if gate["status"] == "evaluated":
                gate["passed"] = False
        metadata = dict(run_metadata or {})
        metadata.setdefault(
            "case_set_sha256",
            hashlib.sha256(
                "\n".join(sorted(row["case_id"] for row in case_results)).encode("utf-8")
            ).hexdigest(),
        )
        metadata.setdefault(
            "dataset_versions",
            sorted(
                {
                    str(row["dataset_version"])
                    for row in case_results
                    if row.get("dataset_version")
                }
            ),
        )
        metadata.setdefault(
            "splits",
            sorted({str(row["split"]) for row in case_results if row.get("split")}),
        )
        return {
            "schema_version": 1,
            "pipeline_version": self.config["pipeline_version"],
            "rubric_version": self.rubric.version,
            "prompt_version": self.config["judge"]["prompt_version"],
            "judge_id": self.judge.judge_id if self.judge is not None else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": self._sha256(self.config_path),
            "rubric_sha256": self._sha256(self.rubric.path),
            "run_metadata": metadata,
            "summary": summary,
            "human_alignment": human_alignment,
            "slices": slices,
            "review_queue": review_queue,
            "baseline": baseline_comparison,
            "production_gate": gate,
            "cases": case_results,
            "status_note": (
                "Machine scores are candidates until calibrated against a closed, fully "
                "resolved independent human-evaluation batch."
            ),
        }

    def _evaluate_case(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        deterministic = evaluate_deterministic(item)
        judge_attempts = 0
        judge_result = JudgeResult(
                status="not_run",
                scores={dimension_id: None for dimension_id in self.rubric.dimension_ids},
                vetoes=[],
                rationales={},
                overall_score=None,
                passed=None,
                error_code=None,
            )
        if self.judge is not None:
            maximum_attempts = max(
                1, int(self.config["judge"].get("maximum_attempts", 1))
            )
            while judge_attempts < maximum_attempts:
                judge_attempts += 1
                judge_result = self.judge.evaluate(item)
                if judge_result.status == "evaluated":
                    break
        hybrid = self._hybrid(judge_result, deterministic)
        labels = item.get("labels") or {}
        return {
            "case_id": item["case_id"],
            "family_id": item.get("family_id") or item["case_id"],
            "dataset_version": item.get("dataset_version"),
            "split": item.get("split"),
            "category": item.get("category"),
            "scene": item.get("scene"),
            "risk_level": item.get("risk_level") or labels.get("risk_level"),
            "capability_tags": item.get("capability_tags")
            or labels.get("capability_tags")
            or [],
            "model_metadata": dict(item.get("model_metadata") or {}),
            "latency_ms": item.get("latency_ms"),
            "estimated_cost": item.get("estimated_cost"),
            "cost_mode": item.get("cost_mode"),
            "tokens_in": item.get("tokens_in"),
            "tokens_out": item.get("tokens_out"),
            "deterministic": deterministic,
            "judge": judge_result.to_dict(),
            "judge_attempts": judge_attempts,
            "hybrid": hybrid,
        }

    def _hybrid(
        self,
        judge: JudgeResult,
        deterministic: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if judge.status != "evaluated":
            return None
        scores = dict(judge.scores)
        overrides = []

        def cap(dimension: str, maximum: int, reason: str) -> None:
            previous = scores.get(dimension)
            if previous is not None and previous > maximum:
                scores[dimension] = maximum
                overrides.append(
                    {
                        "dimension": dimension,
                        "from": previous,
                        "to": maximum,
                        "reason": reason,
                    }
                )

        rules = self.config["hybrid_overrides"]
        if not deterministic["outcome_match"]:
            rule = rules["missing_required_outcome"]
            cap(rule["dimension"], int(rule["maximum_score"]), "outcome_mismatch")
        if deterministic["expected_tools"] and deterministic["tool_recall"] == 0:
            rule = rules["missing_all_required_tools"]
            cap(rule["dimension"], int(rule["maximum_score"]), "all_required_tools_missing")
        elif (
            deterministic["tool_recall"] < 1
            or deterministic["tool_precision"] < 1
            or deterministic["parameter_accuracy"] < 1
        ):
            rule = rules["incomplete_tool_or_parameter_match"]
            cap(rule["dimension"], int(rule["maximum_score"]), "tool_or_parameter_mismatch")
        if deterministic["missing_facts"] and not deterministic["matched_facts"]:
            rule = rules["missing_all_required_facts"]
            cap(rule["dimension"], int(rule["maximum_score"]), "all_required_facts_missing")
        elif deterministic["fact_coverage"] < 1:
            rule = rules["incomplete_required_facts"]
            cap(rule["dimension"], int(rule["maximum_score"]), "required_facts_incomplete")
        if not deterministic["citation_requirement_met"]:
            rule = rules["invalid_required_citation"]
            cap(rule["dimension"], int(rule["maximum_score"]), "citation_requirement_failed")
        if not deterministic["artifact_requirement_met"]:
            rule = rules["missing_required_artifact"]
            cap(rule["dimension"], int(rule["maximum_score"]), "artifact_requirement_failed")
        overall = self.rubric.overall_score(scores)
        return {
            "scores": scores,
            "vetoes": list(judge.vetoes),
            "overall_score": overall,
            "passed": self.rubric.case_passed(
                scores, judge.vetoes, overall_score=overall
            ),
            "deterministic_overrides": overrides,
        }

    def _summary(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        judged = [row for row in rows if row["judge"]["status"] == "evaluated"]
        judge_errors = [row for row in rows if row["judge"]["status"] == "error"]
        hybrid = [row["hybrid"] for row in rows if row["hybrid"] is not None]
        deterministic = [row["deterministic"] for row in rows]
        error_codes = Counter(
            row["judge"]["error_code"] for row in judge_errors if row["judge"]["error_code"]
        )
        return {
            "case_count": len(rows),
            "deterministic_pass_rate": self._avg(deterministic, "passed"),
            "outcome_accuracy": self._avg(deterministic, "outcome_match"),
            "tool_selection_accuracy": self._avg(
                deterministic, "tool_selection_accuracy"
            ),
            "tool_parameter_accuracy": self._avg(deterministic, "parameter_accuracy"),
            "fact_coverage": self._avg(deterministic, "fact_coverage"),
            "citation_validity": self._avg(
                [row for row in deterministic if row["citation_required"]],
                "citation_validity",
            ),
            "judge_evaluated_count": len(judged),
            "judge_error_count": len(judge_errors),
            "judge_not_run_count": len(rows) - len(judged) - len(judge_errors),
            "judge_coverage": round(len(judged) / len(rows), 4) if rows else 0.0,
            "judge_error_rate": round(len(judge_errors) / len(rows), 4) if rows else 0.0,
            "judge_error_codes": dict(sorted(error_codes.items())),
            "machine_resolved_count": len(hybrid),
            "machine_pass_rate": self._avg(hybrid, "passed"),
            "machine_overall_score": self._avg(hybrid, "overall_score"),
            "performance": self._performance_summary(rows),
        }

    @staticmethod
    def _performance_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
        costs = [
            float(row["estimated_cost"])
            for row in rows
            if row.get("estimated_cost") is not None
        ]
        tokens = [
            int(row.get("tokens_in") or 0) + int(row.get("tokens_out") or 0)
            for row in rows
            if row.get("tokens_in") is not None or row.get("tokens_out") is not None
        ]
        if not latencies and not costs and not tokens:
            return {"status": "not_available"}
        output: Dict[str, Any] = {"status": "available"}
        if latencies:
            ordered = sorted(latencies)
            index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
            output.update(
                {
                    "latency_case_count": len(latencies),
                    "latency_mean_ms": round(mean(latencies), 4),
                    "latency_p95_ms": round(ordered[index], 4),
                }
            )
        if costs:
            output.update(
                {
                    "cost_case_count": len(costs),
                    "estimated_cost_mean": round(mean(costs), 6),
                    "cost_modes": sorted(
                        {str(row.get("cost_mode") or "unknown") for row in rows if row.get("estimated_cost") is not None}
                    ),
                }
            )
        if tokens:
            output.update(
                {
                    "token_case_count": len(tokens),
                    "token_mean": round(mean(tokens), 4),
                }
            )
        return output

    def _human_alignment(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        human_export: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if human_export is None:
            return {
                "status": "not_available",
                "pair_count": 0,
                "reason": "human_labels_missing",
            }
        if human_export.get("schema_version") != 1:
            return {
                "status": "incompatible",
                "pair_count": 0,
                "reason": "human_export_schema_mismatch",
            }
        if human_export.get("rubric_version") != self.rubric.version:
            return {
                "status": "incompatible",
                "pair_count": 0,
                "reason": "human_export_rubric_mismatch",
            }
        human_records = human_export.get("records") or []
        human_case_ids = [record.get("case_id") for record in human_records]
        if any(not case_id for case_id in human_case_ids) or len(human_case_ids) != len(
            set(human_case_ids)
        ):
            raise ValueError("human export requires unique non-empty case_id values")
        human_by_case = {
            record["case_id"]: record.get("final")
            for record in human_records
            if record.get("final") is not None
        }
        pairs = [
            (row, human_by_case[row["case_id"]])
            for row in rows
            if row["hybrid"] is not None and row["case_id"] in human_by_case
        ]
        valid_pairs = [pair for pair in pairs if pair[1].get("valid")]
        tp = sum(machine["hybrid"]["passed"] and human["passed"] for machine, human in valid_pairs)
        fp = sum(machine["hybrid"]["passed"] and not human["passed"] for machine, human in valid_pairs)
        fn = sum(not machine["hybrid"]["passed"] and human["passed"] for machine, human in valid_pairs)
        tn = sum(not machine["hybrid"]["passed"] and not human["passed"] for machine, human in valid_pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dimension_metrics = {}
        for dimension_id in self.rubric.dimension_ids:
            score_pairs = [
                (machine["hybrid"]["scores"][dimension_id], human["scores"][dimension_id])
                for machine, human in valid_pairs
                if machine["hybrid"]["scores"][dimension_id] is not None
                and human["scores"][dimension_id] is not None
            ]
            dimension_metrics[dimension_id] = {
                "pair_count": len(score_pairs),
                "mae": round(mean(abs(a - b) for a, b in score_pairs), 4)
                if score_pairs
                else None,
                "exact_agreement_rate": round(
                    sum(a == b for a, b in score_pairs) / len(score_pairs), 4
                )
                if score_pairs
                else None,
                "weighted_kappa": quadratic_weighted_kappa(score_pairs),
            }
        safety_vetoes = {
            veto_id
            for veto_id, rule in self.rubric.veto_rules.items()
            if "safety" in rule["forces_scores"]
        }
        safety_false_negatives = [
            machine["case_id"]
            for machine, human in valid_pairs
            if (
                human["scores"].get("safety") is not None
                and human["scores"]["safety"] <= 1
                or set(human.get("vetoes") or []) & safety_vetoes
            )
            and machine["hybrid"]["scores"].get("safety", 3) > 1
            and not (set(machine["hybrid"]["vetoes"]) & safety_vetoes)
        ]
        veto_false_negatives = [
            machine["case_id"]
            for machine, human in valid_pairs
            if human.get("vetoes")
            and not set(machine["hybrid"]["vetoes"]) & set(human["vetoes"])
        ]
        core_dimensions = next(
            metric["core_dimensions"]
            for metric in yaml.safe_load(
                Path("config/evaluation_metrics.yml").read_text(encoding="utf-8")
            )["metrics"]
            if metric["id"] == "weighted_kappa"
        )
        core_mae = [
            dimension_metrics[dimension]["mae"]
            for dimension in core_dimensions
            if dimension_metrics[dimension]["mae"] is not None
        ]
        core_kappa = [
            dimension_metrics[dimension]["weighted_kappa"]
            for dimension in core_dimensions
            if dimension_metrics[dimension]["weighted_kappa"] is not None
        ]
        return {
            "status": "evaluated",
            "human_batch_id": human_export.get("batch_id"),
            "human_batch_status": human_export.get("batch_status"),
            "human_pending_final_count": human_export.get("pending_final_count"),
            "pair_count": len(pairs),
            "valid_pair_count": len(valid_pairs),
            "pass_confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "pass_precision": round(precision, 4),
            "pass_recall": round(recall, 4),
            "pass_f1": round(f1, 4),
            "pass_agreement": round((tp + tn) / len(valid_pairs), 4)
            if valid_pairs
            else None,
            "dimension_metrics": dimension_metrics,
            "core_dimension_mae_macro": round(mean(core_mae), 4) if core_mae else None,
            "core_weighted_kappa_macro": round(mean(core_kappa), 4)
            if core_kappa
            else None,
            "safety_false_negative_count": len(safety_false_negatives),
            "safety_false_negative_case_ids": safety_false_negatives,
            "veto_false_negative_count": len(veto_false_negatives),
            "veto_false_negative_case_ids": veto_false_negatives,
        }

    def _gate(
        self,
        summary: Mapping[str, Any],
        alignment: Mapping[str, Any],
        human_export: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        thresholds = self.config["calibration_gate"]
        if self.judge is None:
            return {"status": "blocked", "passed": None, "failures": ["judge_not_run"]}
        failures = []
        if not self.judge.judge_id.strip() or self.judge.judge_id == "configured-chat-model":
            failures.append("judge_id_not_pinned")
        if summary["judge_coverage"] < float(thresholds["minimum_judge_coverage"]):
            failures.append("judge_coverage_below_threshold")
        if summary["judge_error_rate"] > float(
            self.config["judge"]["maximum_error_rate"]
        ):
            failures.append("judge_error_rate_above_threshold")
        if alignment["status"] != "evaluated":
            failures.append("human_labels_missing")
            return {"status": "blocked", "passed": None, "failures": failures}
        if thresholds["require_closed_human_batch"] and human_export.get("batch_status") != "closed":
            failures.append("human_batch_not_closed")
        if thresholds["require_all_human_labels_resolved"] and int(
            human_export.get("pending_final_count") or 0
        ):
            failures.append("human_labels_unresolved")
        if alignment["pair_count"] < int(thresholds["minimum_human_pairs"]):
            failures.append("human_pair_count_below_threshold")
            return {"status": "insufficient_sample", "passed": None, "failures": failures}
        checks = (
            (alignment["pass_f1"] < float(thresholds["minimum_pass_f1"]), "pass_f1_below_threshold"),
            (
                alignment["pass_agreement"] is None
                or alignment["pass_agreement"] < float(thresholds["minimum_pass_agreement"]),
                "pass_agreement_below_threshold",
            ),
            (
                alignment["core_dimension_mae_macro"] is None
                or alignment["core_dimension_mae_macro"]
                > float(thresholds["maximum_core_dimension_mae"]),
                "core_dimension_mae_above_threshold",
            ),
            (
                alignment["core_weighted_kappa_macro"] is None
                or alignment["core_weighted_kappa_macro"]
                < float(thresholds["minimum_core_weighted_kappa"]),
                "core_weighted_kappa_below_threshold",
            ),
            (
                alignment["safety_false_negative_count"]
                > int(thresholds["maximum_safety_false_negatives"]),
                "safety_false_negative_detected",
            ),
            (
                alignment["veto_false_negative_count"]
                > int(thresholds["maximum_veto_false_negatives"]),
                "veto_false_negative_detected",
            ),
        )
        failures.extend(message for failed, message in checks if failed)
        return {"status": "evaluated", "passed": not failures, "failures": failures}

    def _slices(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        human_export: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        minimum = int(self.config["slicing"]["minimum_reported_group_size"])
        human_by_case = {
            record["case_id"]: record.get("final")
            for record in (human_export or {}).get("records") or []
            if record.get("final") is not None and record["final"].get("valid")
        }
        grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {
            dimension: defaultdict(list)
            for dimension in self.config["slicing"]["dimensions"]
        }
        for row in rows:
            for dimension in ("scene", "category", "risk_level"):
                grouped[dimension][str(row.get(dimension) or "unknown")].append(row)
            for tag in row.get("capability_tags") or []:
                grouped["capability_tag"][str(tag)].append(row)
        output = {}
        for dimension, groups in grouped.items():
            output[dimension] = {}
            for name, group_rows in sorted(groups.items()):
                if len(group_rows) < minimum:
                    continue
                aligned = [
                    (row, human_by_case[row["case_id"]])
                    for row in group_rows
                    if row["hybrid"] is not None and row["case_id"] in human_by_case
                ]
                slice_metrics = {
                    "case_count": len(group_rows),
                    "judge_coverage": round(
                        sum(row["judge"]["status"] == "evaluated" for row in group_rows)
                        / len(group_rows),
                        4,
                    ),
                    "machine_pass_rate": self._avg(
                        [row["hybrid"] for row in group_rows if row["hybrid"] is not None],
                        "passed",
                    ),
                    "deterministic_pass_rate": self._avg(
                        [row["deterministic"] for row in group_rows], "passed"
                    ),
                }
                if aligned:
                    slice_metrics["human_aligned_count"] = len(aligned)
                    slice_metrics["pass_agreement"] = round(
                        sum(
                            row["hybrid"]["passed"] == human["passed"]
                            for row, human in aligned
                        )
                        / len(aligned),
                        4,
                    )
                    slice_metrics["false_positive_count"] = sum(
                        row["hybrid"]["passed"] and not human["passed"]
                        for row, human in aligned
                    )
                    slice_metrics["false_negative_count"] = sum(
                        not row["hybrid"]["passed"] and human["passed"]
                        for row, human in aligned
                    )
                    score_errors = [
                        abs(row["hybrid"]["scores"][dimension_id] - human["scores"][dimension_id])
                        for row, human in aligned
                        for dimension_id in self.rubric.dimension_ids
                        if row["hybrid"]["scores"][dimension_id] is not None
                        and human["scores"][dimension_id] is not None
                    ]
                    slice_metrics["dimension_mae_macro"] = round(mean(score_errors), 4)
                output[dimension][name] = slice_metrics
        return output

    def _compare_baseline(
        self,
        alignment: Mapping[str, Any],
        baseline: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if baseline is None:
            return None
        policy = self.config["baseline_policy"]
        failures = []
        if policy["require_approved_baseline"] and baseline.get("status") != "approved":
            failures.append("baseline_not_approved")
        if policy["require_same_pipeline_version"] and baseline.get(
            "pipeline_version"
        ) != self.config["pipeline_version"]:
            failures.append("baseline_pipeline_version_mismatch")
        if policy["require_same_rubric_version"] and baseline.get(
            "rubric_version"
        ) != self.rubric.version:
            failures.append("baseline_rubric_version_mismatch")
        expected = baseline.get("human_alignment") or {}
        deltas = {}
        required_metrics = (
            "pass_f1",
            "pass_agreement",
            "core_dimension_mae_macro",
            "safety_false_negative_count",
            "veto_false_negative_count",
        )
        for metric in required_metrics:
            if isinstance(alignment.get(metric), (int, float)) and isinstance(
                expected.get(metric), (int, float)
            ):
                deltas[metric] = round(float(alignment[metric]) - float(expected[metric]), 4)
        if len(deltas) != len(required_metrics):
            failures.append("baseline_alignment_metrics_missing")
        if deltas.get("pass_f1", 0) < -float(policy["maximum_pass_f1_regression"]):
            failures.append("pass_f1_regressed")
        if deltas.get("pass_agreement", 0) < -float(
            policy["maximum_pass_agreement_regression"]
        ):
            failures.append("pass_agreement_regressed")
        if deltas.get("core_dimension_mae_macro", 0) > float(
            policy["maximum_core_mae_increase"]
        ):
            failures.append("core_dimension_mae_regressed")
        if not policy["allow_new_safety_false_negative"] and deltas.get(
            "safety_false_negative_count", 0
        ) > 0:
            failures.append("new_safety_false_negative")
        if not policy["allow_new_veto_false_negative"] and deltas.get(
            "veto_false_negative_count", 0
        ) > 0:
            failures.append("new_veto_false_negative")
        return {
            "status": "evaluated" if not failures else "inconclusive",
            "comparable": not failures,
            "passed": not failures,
            "deltas": deltas,
            "failures": failures,
        }

    def _review_queue(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        human_export: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        human_by_case = {
            record["case_id"]: record.get("final")
            for record in (human_export or {}).get("records") or []
            if record.get("final") is not None and record["final"].get("valid")
        }
        entries = []
        safety_vetoes = {
            veto_id
            for veto_id, rule in self.rubric.veto_rules.items()
            if "safety" in rule["forces_scores"]
        }
        for row in rows:
            reasons = []
            priority = 3
            human = human_by_case.get(row["case_id"])
            hybrid = row["hybrid"]
            if human is not None and hybrid is not None:
                human_safety_failure = (
                    human["scores"].get("safety") is not None
                    and human["scores"]["safety"] <= 1
                ) or bool(set(human.get("vetoes") or []) & safety_vetoes)
                machine_safety_failure = (
                    hybrid["scores"].get("safety") is not None
                    and hybrid["scores"]["safety"] <= 1
                ) or bool(set(hybrid.get("vetoes") or []) & safety_vetoes)
                if human_safety_failure and not machine_safety_failure:
                    reasons.append("safety_false_negative")
                    priority = 0
                if human.get("vetoes") and not set(hybrid["vetoes"]) & set(
                    human["vetoes"]
                ):
                    reasons.append("veto_false_negative")
                    priority = 0
                if hybrid["passed"] != human["passed"]:
                    reasons.append(
                        "pass_false_positive" if hybrid["passed"] else "pass_false_negative"
                    )
                    priority = min(priority, 1)
            if row["judge"]["status"] == "error":
                reasons.append(f"judge_error:{row['judge']['error_code']}")
                priority = min(priority, 1)
            if hybrid is not None and hybrid["deterministic_overrides"]:
                reasons.extend(
                    f"hybrid_override:{override['reason']}"
                    for override in hybrid["deterministic_overrides"]
                )
                priority = min(priority, 2)
            if row["deterministic"]["failures"]:
                reasons.extend(
                    f"deterministic:{failure}"
                    for failure in row["deterministic"]["failures"]
                )
                priority = min(priority, 2)
            if reasons:
                entries.append(
                    {
                        "case_id": row["case_id"],
                        "priority": {0: "P0", 1: "P1", 2: "P2"}[priority],
                        "scene": row.get("scene"),
                        "risk_level": row.get("risk_level"),
                        "reasons": list(dict.fromkeys(reasons)),
                    }
                )
        entries.sort(
            key=lambda entry: (
                {"P0": 0, "P1": 1, "P2": 2}[entry["priority"]],
                entry["case_id"],
            )
        )
        counts = Counter(entry["priority"] for entry in entries)
        return {
            "case_count": len(entries),
            "priority_counts": {
                priority: counts.get(priority, 0) for priority in ("P0", "P1", "P2")
            },
            "cases": entries,
        }

    @staticmethod
    def _validate_items(items: Sequence[Mapping[str, Any]]) -> None:
        if not items:
            raise ValueError("machine evaluation requires at least one item")
        case_ids = [str(item.get("case_id") or "") for item in items]
        if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
            raise ValueError("machine evaluation items require unique non-empty case_id values")
        for item in items:
            if not isinstance(item.get("agent_answer"), str):
                raise ValueError(f"{item['case_id']}: agent_answer must be a string")
            if not isinstance(item.get("expected"), Mapping):
                raise ValueError(f"{item['case_id']}: expected must be an object")
            for field in ("latency_ms", "estimated_cost"):
                value = item.get(field)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise ValueError(f"{item['case_id']}: {field} must be a non-negative finite number")
            for field in ("tokens_in", "tokens_out"):
                value = item.get(field)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise ValueError(f"{item['case_id']}: {field} must be a non-negative integer")

    @staticmethod
    def _avg(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        return round(mean(float(row[key]) for row in rows), 4) if rows else None

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
