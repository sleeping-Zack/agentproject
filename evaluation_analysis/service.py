"""Paired Agent experiment analysis and guarded iteration recommendations."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from uuid import uuid4

import yaml

from evaluation_analysis.statistics import (
    paired_binary_test,
    paired_bootstrap_mean_delta,
    paired_sign_test,
    percentile,
)
from human_eval.rubric import RubricEvaluator


DEFAULT_CONFIG_PATH = Path("config/evaluation_analysis.yml")
DEFAULT_MACHINE_CONFIG_PATH = Path("config/machine_evaluation.yml")


class EvaluationAnalysisService:
    """Compare two complete phase-four reports over the exact same cases."""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        *,
        rubric: Optional[RubricEvaluator] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.rubric = rubric or RubricEvaluator()
        self._validate_config()
        self.safety_vetoes = {
            veto_id
            for veto_id, rule in self.rubric.veto_rules.items()
            if "safety" in rule.get("forces_scores", {})
        }

    @staticmethod
    def report_sha256(report: Mapping[str, Any]) -> str:
        """Return the canonical digest that a baseline approval must bind to."""

        return EvaluationAnalysisService._sha256_json(report)

    def validate_baseline_for_approval(self, report: Mapping[str, Any]) -> str:
        """Validate a phase-four baseline and return the digest an approval must bind."""

        normalized = self._validate_report(report, name="baseline")
        lineage_failures = self._trusted_lineage_failures(normalized, side="baseline")
        if lineage_failures:
            raise ValueError(
                "baseline evaluator lineage is untrusted: " + ", ".join(lineage_failures)
            )
        gate = normalized.get("production_gate") or {}
        if (
            gate.get("status") != "evaluated"
            or gate.get("passed") is not True
            or gate.get("failures") != []
        ):
            raise ValueError("baseline report is not production ready")
        return self.report_sha256(report)

    def analyze(
        self,
        baseline_report: Mapping[str, Any],
        candidate_report: Mapping[str, Any],
        *,
        experiment: Mapping[str, Any],
        baseline_approval: Optional[Mapping[str, Any]] = None,
        report_id: Optional[str] = None,
        generated_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a non-mutating analysis report.

        ``promotion`` can only recommend human approval. This method never changes a
        model, prompt, dataset, baseline, or deployment.
        """

        normalized_experiment = self._validate_experiment(experiment)
        baseline = self._validate_report(baseline_report, name="baseline")
        candidate = self._validate_report(candidate_report, name="candidate")
        baseline_hash = self.report_sha256(baseline_report)
        candidate_hash = self.report_sha256(candidate_report)
        baseline_identity = self._identity(baseline)
        candidate_identity = self._identity(candidate)

        comparability = self._comparability(
            baseline,
            candidate,
            mode=normalized_experiment["mode"],
        )
        evaluator_gate = self._evaluator_gate(
            baseline,
            candidate,
            mode=normalized_experiment["mode"],
        )
        approval_failures, approval_warnings = self._approval_status(
            mode=normalized_experiment["mode"],
            approval=baseline_approval,
            baseline_sha256=baseline_hash,
        )

        baseline_by_case = {row["case_id"]: row for row in baseline["cases"]}
        candidate_by_case = {row["case_id"]: row for row in candidate["cases"]}
        common_case_ids = sorted(set(baseline_by_case) & set(candidate_by_case))
        resolved_case_ids = [
            case_id
            for case_id in common_case_ids
            if self._resolved(baseline_by_case[case_id])
            and self._resolved(candidate_by_case[case_id])
        ]
        unresolved_count = len(common_case_ids) - len(resolved_case_ids)
        if unresolved_count:
            if normalized_experiment["mode"] == "promotion":
                evaluator_gate["status"] = "failed"
                evaluator_gate["reasons"].append("incomplete_paired_outcomes")
            else:
                comparability["status"] = (
                    "partial" if comparability["status"] == "comparable" else comparability["status"]
                )
                comparability["warnings"].append("incomplete_paired_outcomes_excluded")

        can_compute = comparability["status"] != "not_comparable" and bool(
            resolved_case_ids
        )
        if can_compute:
            analysis = self._paired_analysis(
                baseline_by_case,
                candidate_by_case,
                resolved_case_ids,
                mode=normalized_experiment["mode"],
                slice_dimensions=normalized_experiment["predeclared_slices"],
            )
            analysis["safety"].update(self._p0_regression(baseline, candidate))
        else:
            analysis = self._empty_analysis(
                pair_count=0,
                reason=(
                    "input_reports_not_comparable"
                    if comparability["status"] == "not_comparable"
                    else "no_resolved_pairs"
                ),
            )
        if normalized_experiment["mode"] == "promotion":
            analysis = self._aggregate_only_promotion_analysis(analysis)

        evidence = self._evidence(
            analysis["quality_comparison"],
            pair_count=len(resolved_case_ids) if can_compute else 0,
            unresolved_count=unresolved_count,
            can_compute=can_compute,
        )
        release_decision = self._release_decision(
            mode=normalized_experiment["mode"],
            comparability=comparability,
            evaluator_gate=evaluator_gate,
            evidence=evidence,
            approval_failures=approval_failures,
            safety=analysis["safety"],
            quality=analysis["quality_comparison"],
            performance=analysis["performance"],
            slices=analysis["slices"],
        )

        diagnostic_allowed = (
            normalized_experiment["mode"] == "diagnostic"
            and comparability["status"] != "not_comparable"
        )
        bad_cases = (
            self._bad_cases(
                baseline_by_case,
                candidate_by_case,
                resolved_case_ids,
                analysis["safety"],
            )
            if diagnostic_allowed
            else []
        )
        root_cause_summary = self._root_cause_summary(bad_cases)
        recommendations = (
            self._recommendations(bad_cases) if diagnostic_allowed else []
        )
        regression_candidates = (
            self._regression_candidates(bad_cases, candidate_by_case)
            if diagnostic_allowed
            else {"proposed": [], "excluded": []}
        )

        limitations = self._limitations(
            mode=normalized_experiment["mode"],
            performance=analysis["performance"],
            unresolved_count=unresolved_count,
        )
        comparability["warnings"] = self._unique(
            [*comparability["warnings"], *approval_warnings]
        )
        comparability["reasons"] = self._unique(comparability["reasons"])
        evaluator_gate["warnings"] = self._unique(evaluator_gate["warnings"])
        evaluator_gate["reasons"] = self._unique(evaluator_gate["reasons"])

        timestamp = generated_at or datetime.now(timezone.utc).isoformat()
        self._require_timestamp(timestamp, "generated_at")
        final_report_id = report_id or f"analysis-{uuid4()}"
        self._require_string(final_report_id, "report_id")
        return {
            "schema_version": 1,
            "analysis_version": self.config["analysis_version"],
            "report_id": final_report_id,
            "generated_at": timestamp,
            "analysis_config_sha256": self._sha256_bytes(
                self.config_path.read_bytes()
            ),
            "input_reports": {
                "baseline": {"sha256": baseline_hash, "identity": baseline_identity},
                "candidate": {"sha256": candidate_hash, "identity": candidate_identity},
            },
            "experiment": normalized_experiment,
            "comparability": comparability,
            "evaluator_gate": evaluator_gate,
            "evidence": evidence,
            "quality_comparison": analysis["quality_comparison"],
            "performance": analysis["performance"],
            "safety": analysis["safety"],
            "slices": analysis["slices"],
            "case_transitions": analysis["case_transitions"],
            "bad_cases": bad_cases,
            "root_cause_summary": root_cause_summary,
            "recommendations": recommendations,
            "regression_candidates": regression_candidates,
            "release_decision": release_decision,
            "limitations": limitations,
        }

    def _validate_config(self) -> None:
        if self.config.get("schema_version") != 1:
            raise ValueError("evaluation analysis config schema_version must be 1")
        if self.config.get("rubric_version") != self.rubric.version:
            raise ValueError("evaluation analysis and Rubric versions do not match")
        statistics = self.config.get("statistics") or {}
        if int(statistics.get("minimum_paired_cases") or 0) < 2:
            raise ValueError("minimum_paired_cases must be at least 2")
        if not 0 < float(statistics.get("confidence_level") or 0) < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if not 0 < float(statistics.get("alpha") or 0) < 1:
            raise ValueError("alpha must be between 0 and 1")

    def _validate_experiment(self, experiment: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(experiment, Mapping):
            raise ValueError("experiment must be an object")
        experiment_id = self._require_string(
            experiment.get("experiment_id"), "experiment.experiment_id"
        )
        mode = experiment.get("mode")
        if mode not in {"diagnostic", "promotion"}:
            raise ValueError("experiment.mode must be diagnostic or promotion")
        hypothesis = self._require_string(
            experiment.get("hypothesis"), "experiment.hypothesis"
        )
        change = self._require_string(experiment.get("change"), "experiment.change")
        primary_metric = experiment.get(
            "primary_metric", self.config["decision_policy"]["primary_metric"]
        )
        if primary_metric != self.config["decision_policy"]["primary_metric"]:
            raise ValueError("experiment.primary_metric is not supported by this config")
        allowed_slices = set(
            self.config["experiment_policy"]["predeclared_slice_dimensions"]
        )
        predeclared = experiment.get("predeclared_slices") or sorted(allowed_slices)
        if (
            not isinstance(predeclared, list)
            or not predeclared
            or any(not isinstance(value, str) for value in predeclared)
            or not set(predeclared) <= allowed_slices
        ):
            raise ValueError("experiment.predeclared_slices contains unsupported values")
        return {
            "experiment_id": experiment_id,
            "mode": mode,
            "hypothesis": hypothesis,
            "change": change,
            "primary_metric": primary_metric,
            "predeclared_slices": list(dict.fromkeys(predeclared)),
        }

    def _validate_report(
        self, report: Mapping[str, Any], *, name: str
    ) -> Dict[str, Any]:
        if not isinstance(report, Mapping):
            raise ValueError(f"{name} report must be an object")
        if report.get("schema_version") != self.config["machine_report_schema_version"]:
            raise ValueError(f"{name} report has an unsupported schema_version")
        cases = report.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{name} report cases must be a non-empty array")
        case_ids = []
        for row in cases:
            if not isinstance(row, Mapping):
                raise ValueError(f"{name} report cases must contain objects")
            case_id = self._require_string(row.get("case_id"), f"{name}.case_id")
            case_ids.append(case_id)
            hybrid = row.get("hybrid")
            if hybrid is not None:
                if not isinstance(hybrid, Mapping):
                    raise ValueError(f"{name}.{case_id}.hybrid must be an object or null")
                if type(hybrid.get("passed")) is not bool:
                    raise ValueError(f"{name}.{case_id}.hybrid.passed must be a boolean")
                overall_score = self._finite_number(
                    hybrid.get("overall_score"),
                    f"{name}.{case_id}.hybrid.overall_score",
                )
                if not 0 <= overall_score <= 3:
                    raise ValueError(
                        f"{name}.{case_id}.hybrid.overall_score must be between 0 and 3"
                    )
                scores = hybrid.get("scores")
                if not isinstance(scores, Mapping):
                    raise ValueError(f"{name}.{case_id}.hybrid.scores must be an object")
                if set(scores) != set(self.rubric.dimension_ids):
                    raise ValueError(
                        f"{name}.{case_id}.hybrid.scores must contain every Rubric dimension"
                    )
                for dimension, value in scores.items():
                    if value is not None:
                        if type(value) is not int or value not in {0, 1, 2, 3}:
                            raise ValueError(
                                f"{name}.{case_id}.hybrid.scores.{dimension} "
                                "must be an integer from 0 to 3 or null"
                            )
                vetoes = hybrid.get("vetoes")
                if not isinstance(vetoes, list) or len(vetoes) != len(set(vetoes)):
                    raise ValueError(
                        f"{name}.{case_id}.hybrid.vetoes must be a unique list"
                    )
                unknown_vetoes = set(vetoes) - set(self.rubric.veto_rules)
                if unknown_vetoes:
                    raise ValueError(
                        f"{name}.{case_id}.hybrid contains unknown vetoes: "
                        f"{sorted(unknown_vetoes)}"
                    )
                for veto_id in vetoes:
                    for dimension, forced_score in self.rubric.veto_rules[veto_id][
                        "forces_scores"
                    ].items():
                        if scores.get(dimension) != forced_score:
                            raise ValueError(
                                f"{name}.{case_id}.hybrid veto {veto_id} requires "
                                f"{dimension}={forced_score}"
                            )
                calculated_overall = self.rubric.overall_score(scores)
                if not math.isclose(
                    overall_score,
                    calculated_overall,
                    rel_tol=0,
                    abs_tol=10 ** (-self.rubric.precision),
                ):
                    raise ValueError(
                        f"{name}.{case_id}.hybrid.overall_score does not match its scores"
                    )
                calculated_pass = self.rubric.case_passed(
                    scores,
                    vetoes,
                    overall_score=calculated_overall,
                )
                if hybrid["passed"] != calculated_pass:
                    raise ValueError(
                        f"{name}.{case_id}.hybrid.passed does not match the Rubric result"
                    )
            for field in ("latency_ms", "estimated_cost"):
                if row.get(field) is not None:
                    value = self._finite_number(row[field], f"{name}.{case_id}.{field}")
                    if value < 0:
                        raise ValueError(f"{name}.{case_id}.{field} cannot be negative")
            cost_mode = row.get("cost_mode")
            if cost_mode is not None and (
                not isinstance(cost_mode, str) or not cost_mode.strip()
            ):
                raise ValueError(f"{name}.{case_id}.cost_mode must be a non-empty string")
            for field in ("tokens_in", "tokens_out"):
                value = row.get(field)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{name}.{case_id}.{field} must be a non-negative integer")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"{name} report has duplicate case_id values")
        return dict(report)

    def _comparability(
        self,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        policy = self.config["decision_policy"]
        reasons = []
        warnings = []
        checks = (
            (
                "pipeline_version",
                "pipeline_version_mismatch",
                policy["require_same_pipeline_version"],
            ),
            (
                "rubric_version",
                "rubric_version_mismatch",
                policy["require_same_rubric_version"],
            ),
            (
                "prompt_version",
                "prompt_version_mismatch",
                policy["require_same_prompt_version"],
            ),
            ("judge_id", "judge_id_mismatch", policy["require_same_judge_id"]),
            (
                "config_sha256",
                "machine_config_fingerprint_mismatch",
                policy["require_same_config_sha256"],
            ),
            (
                "rubric_sha256",
                "rubric_fingerprint_mismatch",
                policy["require_same_rubric_sha256"],
            ),
        )
        for field, reason, required in checks:
            if not required:
                continue
            if not self._nonempty_string(baseline.get(field)) or not self._nonempty_string(
                candidate.get(field)
            ):
                reasons.append(reason.replace("_mismatch", "_missing"))
            elif baseline.get(field) != candidate.get(field):
                reasons.append(reason)

        for report, side in ((baseline, "baseline"), (candidate, "candidate")):
            reasons.extend(self._trusted_lineage_failures(report, side=side))

        baseline_identity = self._identity(baseline)
        candidate_identity = self._identity(candidate)
        if policy["require_same_dataset_version"] and baseline_identity[
            "dataset_versions"
        ] != candidate_identity["dataset_versions"]:
            reasons.append("dataset_version_mismatch")
        baseline_dataset_hash = baseline_identity.get("dataset_sha256")
        candidate_dataset_hash = candidate_identity.get("dataset_sha256")
        if policy["require_same_dataset_sha256"]:
            if not baseline_dataset_hash or not candidate_dataset_hash:
                if mode == "promotion":
                    reasons.append("dataset_fingerprint_missing")
                else:
                    warnings.append("dataset_fingerprint_missing_derived_case_set_only")
            elif baseline_dataset_hash != candidate_dataset_hash:
                reasons.append("dataset_fingerprint_mismatch")

        baseline_ids = sorted(row["case_id"] for row in baseline["cases"])
        candidate_ids = sorted(row["case_id"] for row in candidate["cases"])
        if policy["require_same_case_set"] and baseline_ids != candidate_ids:
            reasons.append("case_set_mismatch")
        for case_id in sorted(set(baseline_ids) & set(candidate_ids)):
            before = next(row for row in baseline["cases"] if row["case_id"] == case_id)
            after = next(row for row in candidate["cases"] if row["case_id"] == case_id)
            for field in ("family_id", "dataset_version", "split", "scene", "category", "risk_level"):
                if before.get(field) != after.get(field):
                    reasons.append("case_metadata_mismatch")
                    break
            if sorted(before.get("capability_tags") or []) != sorted(
                after.get("capability_tags") or []
            ):
                reasons.append("case_metadata_mismatch")
        for report, identity, side in (
            (baseline, baseline_identity, "baseline"),
            (candidate, candidate_identity, "candidate"),
        ):
            derived = self._case_set_sha256(report["cases"])
            claimed = identity.get("case_set_sha256")
            if claimed and claimed != derived:
                reasons.append(f"{side}_case_set_fingerprint_invalid")
            if identity.get("claimed_dataset_versions") != identity["dataset_versions"]:
                reasons.append(f"{side}_dataset_version_metadata_invalid")
            if identity.get("claimed_splits") != identity["splits"]:
                reasons.append(f"{side}_split_metadata_invalid")

        baseline_splits = baseline_identity["splits"]
        candidate_splits = candidate_identity["splits"]
        if baseline_splits != candidate_splits:
            reasons.append("split_mismatch")
        splits = set(baseline_splits) | set(candidate_splits)
        if self.config["experiment_policy"]["forbid_mixed_splits"] and len(splits) != 1:
            reasons.append("mixed_split_comparison")
        if mode == "diagnostic":
            allowed = set(self.config["experiment_policy"]["diagnostic_allowed_splits"])
            if not splits <= allowed:
                reasons.append("test_split_forbidden_for_diagnostic")
        else:
            allowed = set(self.config["experiment_policy"]["promotion_allowed_splits"])
            if splits != allowed:
                reasons.append("promotion_requires_test_only")

        if baseline.get("pipeline_version") != self.config["pipeline_version"] or candidate.get(
            "pipeline_version"
        ) != self.config["pipeline_version"]:
            reasons.append("unsupported_pipeline_version")
        if baseline.get("rubric_version") != self.config["rubric_version"] or candidate.get(
            "rubric_version"
        ) != self.config["rubric_version"]:
            reasons.append("unsupported_rubric_version")
        return {
            "status": "not_comparable" if reasons else "comparable",
            "reasons": self._unique(reasons),
            "warnings": warnings,
        }

    def _evaluator_gate(
        self,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        issues = []
        for report in (baseline, candidate):
            gate = report.get("production_gate") or {}
            if (
                gate.get("status") != "evaluated"
                or gate.get("passed") is not True
                or gate.get("failures") != []
            ):
                issues.append("source_report_not_production_ready")
            judge_id = report.get("judge_id")
            if not isinstance(judge_id, str) or not judge_id.strip() or judge_id == "configured-chat-model":
                issues.append("judge_id_not_pinned")
        issues = self._unique(issues)
        if not issues:
            return {"status": "passed", "reasons": [], "warnings": []}
        if mode == "promotion":
            return {"status": "failed", "reasons": issues, "warnings": []}
        return {"status": "warning", "reasons": [], "warnings": issues}

    def _approval_status(
        self,
        *,
        mode: str,
        approval: Optional[Mapping[str, Any]],
        baseline_sha256: str,
    ) -> tuple[list[str], list[str]]:
        approved = isinstance(approval, Mapping) and approval.get("status") == "approved"
        if mode == "diagnostic":
            return [], [] if approved else ["baseline_unapproved"]
        if not approved:
            return ["baseline_not_approved"], []
        incomplete = any(
            not isinstance(approval.get(field), str) or not approval[field].strip()
            for field in ("approver", "approved_at", "report_sha256")
        )
        if incomplete:
            return ["baseline_approval_incomplete"], []
        try:
            self._require_timestamp(approval["approved_at"], "baseline_approval.approved_at")
        except ValueError:
            return ["baseline_approval_incomplete"], []
        if approval["report_sha256"] != baseline_sha256:
            return ["baseline_digest_mismatch"], []
        return [], []

    def _paired_analysis(
        self,
        baseline_by_case: Mapping[str, Mapping[str, Any]],
        candidate_by_case: Mapping[str, Mapping[str, Any]],
        case_ids: Sequence[str],
        *,
        mode: str,
        slice_dimensions: Sequence[str],
    ) -> Dict[str, Any]:
        baseline_pass = [bool(baseline_by_case[case_id]["hybrid"]["passed"]) for case_id in case_ids]
        candidate_pass = [bool(candidate_by_case[case_id]["hybrid"]["passed"]) for case_id in case_ids]
        baseline_score = [
            float(baseline_by_case[case_id]["hybrid"]["overall_score"])
            for case_id in case_ids
        ]
        candidate_score = [
            float(candidate_by_case[case_id]["hybrid"]["overall_score"])
            for case_id in case_ids
        ]
        pass_metric = self._metric(
            [float(value) for value in baseline_pass],
            [float(value) for value in candidate_pass],
            mode=mode,
            binary=True,
        )
        score_metric = self._metric(baseline_score, candidate_score, mode=mode)

        dimensions = {}
        for dimension in self.rubric.dimension_ids:
            pairs = [
                (
                    baseline_by_case[case_id]["hybrid"]["scores"].get(dimension),
                    candidate_by_case[case_id]["hybrid"]["scores"].get(dimension),
                )
                for case_id in case_ids
            ]
            applicable = [pair for pair in pairs if pair[0] is not None and pair[1] is not None]
            if applicable:
                dimensions[dimension] = self._metric(
                    [float(pair[0]) for pair in applicable],
                    [float(pair[1]) for pair in applicable],
                    mode=mode,
                )

        safety = self._safety(
            baseline_by_case, candidate_by_case, case_ids
        )
        performance = self._performance(
            baseline_by_case, candidate_by_case, case_ids, mode=mode
        )
        slices = self._slice_analysis(
            baseline_by_case,
            candidate_by_case,
            case_ids,
            mode=mode,
            dimensions=slice_dimensions,
        )
        binary_test = pass_metric["paired_test"]
        return {
            "quality_comparison": {
                "pass_rate": pass_metric,
                "overall_score": score_metric,
                "dimensions": dimensions,
            },
            "performance": performance,
            "safety": safety,
            "slices": slices,
            "case_transitions": {
                "improved": binary_test["candidate_wins"],
                "regressed": binary_test["baseline_wins"],
                "unchanged": binary_test["ties"],
                "improved_case_ids": [
                    case_id
                    for case_id, before, after in zip(
                        case_ids, baseline_pass, candidate_pass
                    )
                    if not before and after
                ],
                "regressed_case_ids": [
                    case_id
                    for case_id, before, after in zip(
                        case_ids, baseline_pass, candidate_pass
                    )
                    if before and not after
                ],
            },
        }

    @staticmethod
    def _aggregate_only_promotion_analysis(analysis: Mapping[str, Any]) -> Dict[str, Any]:
        """Remove all frozen-test case identities while preserving release guardrails."""

        sanitized = dict(analysis)
        transitions = analysis.get("case_transitions") or {}
        sanitized["case_transitions"] = {
            field: int(transitions.get(field) or 0)
            for field in ("improved", "regressed", "unchanged")
        }
        safety = analysis.get("safety") or {}
        sanitized["safety"] = {
            field: int(safety.get(field) or 0)
            for field in (
                "new_failure_count",
                "resolved_failure_count",
                "new_l3_failure_count",
                "new_veto_count",
                "resolved_veto_count",
                "new_p0_count",
                "resolved_p0_count",
            )
        }
        return sanitized

    def _metric(
        self,
        baseline: Sequence[float],
        candidate: Sequence[float],
        *,
        mode: str,
        binary: bool = False,
    ) -> Dict[str, Any]:
        iterations = int(
            self.config["statistics"][
                "promotion_bootstrap_iterations"
                if mode == "promotion"
                else "diagnostic_bootstrap_iterations"
            ]
        )
        bootstrap = paired_bootstrap_mean_delta(
            baseline,
            candidate,
            iterations=iterations,
            confidence_level=float(self.config["statistics"]["confidence_level"]),
            seed=int(self.config["statistics"]["random_seed"]),
        )
        paired = (
            paired_binary_test(
                [bool(value) for value in baseline],
                [bool(value) for value in candidate],
            )
            if binary
            else paired_sign_test(baseline, candidate)
        )
        return {
            "n": bootstrap["n"],
            "baseline": bootstrap["baseline_mean"],
            "candidate": bootstrap["candidate_mean"],
            "delta": bootstrap["delta"],
            "confidence_interval": bootstrap["confidence_interval"],
            "two_sided_p_value": paired["two_sided_p_value"],
            "paired_test": paired,
            "cluster_unit": "case_id",
            "bootstrap_iterations": iterations,
        }

    def _performance(
        self,
        baseline_by_case: Mapping[str, Mapping[str, Any]],
        candidate_by_case: Mapping[str, Mapping[str, Any]],
        case_ids: Sequence[str],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        latency_pairs = [
            (baseline_by_case[case_id].get("latency_ms"), candidate_by_case[case_id].get("latency_ms"))
            for case_id in case_ids
        ]
        cost_pairs = [
            (
                baseline_by_case[case_id].get("estimated_cost"),
                candidate_by_case[case_id].get("estimated_cost"),
            )
            for case_id in case_ids
        ]
        token_pairs = [
            (
                self._token_total(baseline_by_case[case_id]),
                self._token_total(candidate_by_case[case_id]),
            )
            for case_id in case_ids
        ]
        baseline_cost_modes = {
            str(baseline_by_case[case_id].get("cost_mode") or "")
            for case_id in case_ids
            if baseline_by_case[case_id].get("estimated_cost") is not None
        }
        candidate_cost_modes = {
            str(candidate_by_case[case_id].get("cost_mode") or "")
            for case_id in case_ids
            if candidate_by_case[case_id].get("estimated_cost") is not None
        }
        comparable_cost_mode = (
            len(baseline_cost_modes) == 1
            and baseline_cost_modes == candidate_cost_modes
            and "" not in baseline_cost_modes
        )
        latency_metric = self._performance_metric(
            latency_pairs, mode=mode, aggregate="p95"
        )
        cost_metric = (
            self._performance_metric(cost_pairs, mode=mode, aggregate="mean")
            if comparable_cost_mode
            else {"status": "not_available", "reason": "cost_mode_missing_or_mismatch"}
        )
        token_metric = self._performance_metric(
            token_pairs, mode=mode, aggregate="mean"
        )
        return {
            "status": (
                "available"
                if latency_metric.get("status") != "not_available"
                and cost_metric.get("status") != "not_available"
                else "not_available"
            ),
            "p95_latency_ms": latency_metric,
            "average_cost": cost_metric,
            "average_tokens": token_metric,
            "cost_mode": next(iter(baseline_cost_modes))
            if comparable_cost_mode
            else None,
        }

    @staticmethod
    def _token_total(row: Mapping[str, Any]) -> Optional[int]:
        if row.get("tokens_in") is None or row.get("tokens_out") is None:
            return None
        return int(row["tokens_in"]) + int(row["tokens_out"])

    def _performance_metric(
        self,
        pairs: Sequence[tuple[Any, Any]],
        *,
        mode: str,
        aggregate: str,
    ) -> Dict[str, Any]:
        if not pairs or any(before is None or after is None for before, after in pairs):
            return {"status": "not_available"}
        before = [float(pair[0]) for pair in pairs]
        after = [float(pair[1]) for pair in pairs]
        if aggregate == "mean":
            return self._metric(before, after, mode=mode)
        baseline_p95 = percentile(before, 0.95)
        candidate_p95 = percentile(after, 0.95)
        return {
            "status": "available",
            "baseline": baseline_p95,
            "candidate": candidate_p95,
            "delta": round(candidate_p95 - baseline_p95, 6),
            "method": "paired_case_empirical_p95",
        }

    def _safety(
        self,
        baseline_by_case: Mapping[str, Mapping[str, Any]],
        candidate_by_case: Mapping[str, Mapping[str, Any]],
        case_ids: Sequence[str],
    ) -> Dict[str, Any]:
        new_failures = []
        resolved_failures = []
        new_l3_failures = []
        new_vetoes: Dict[str, list[str]] = {}
        resolved_vetoes: Dict[str, list[str]] = {}
        for case_id in case_ids:
            baseline_row = baseline_by_case[case_id]
            candidate_row = candidate_by_case[case_id]
            before = baseline_row["hybrid"]
            after = candidate_row["hybrid"]
            baseline_failure = self._safety_failure(before)
            candidate_failure = self._safety_failure(after)
            if candidate_failure and not baseline_failure:
                new_failures.append(case_id)
            if baseline_failure and not candidate_failure:
                resolved_failures.append(case_id)
            if (
                str(candidate_row.get("risk_level") or "") == "L3"
                and bool(before["passed"])
                and not bool(after["passed"])
            ):
                new_l3_failures.append(case_id)
            added = sorted(set(after.get("vetoes") or []) - set(before.get("vetoes") or []))
            removed = sorted(set(before.get("vetoes") or []) - set(after.get("vetoes") or []))
            if added:
                new_vetoes[case_id] = added
            if removed:
                resolved_vetoes[case_id] = removed
        return {
            "new_failure_count": len(new_failures),
            "new_failure_case_ids": new_failures,
            "resolved_failure_count": len(resolved_failures),
            "resolved_failure_case_ids": resolved_failures,
            "new_l3_failure_count": len(new_l3_failures),
            "new_l3_failure_case_ids": new_l3_failures,
            "new_veto_count": sum(len(values) for values in new_vetoes.values()),
            "new_veto_case_ids": sorted(new_vetoes),
            "new_vetoes": new_vetoes,
            "resolved_veto_count": sum(len(values) for values in resolved_vetoes.values()),
            "resolved_vetoes": resolved_vetoes,
        }

    @staticmethod
    def _p0_regression(
        baseline: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> Dict[str, Any]:
        def p0_ids(report: Mapping[str, Any]) -> set[str]:
            return {
                str(row["case_id"])
                for row in (report.get("review_queue") or {}).get("cases") or []
                if row.get("priority") == "P0" and row.get("case_id")
            }

        baseline_ids = p0_ids(baseline)
        candidate_ids = p0_ids(candidate)
        new_ids = sorted(candidate_ids - baseline_ids)
        resolved_ids = sorted(baseline_ids - candidate_ids)
        return {
            "new_p0_count": len(new_ids),
            "new_p0_case_ids": new_ids,
            "resolved_p0_count": len(resolved_ids),
            "resolved_p0_case_ids": resolved_ids,
        }

    def _slice_analysis(
        self,
        baseline_by_case: Mapping[str, Mapping[str, Any]],
        candidate_by_case: Mapping[str, Mapping[str, Any]],
        case_ids: Sequence[str],
        *,
        mode: str,
        dimensions: Sequence[str],
    ) -> list[Dict[str, Any]]:
        groups: Dict[tuple[str, str], list[str]] = defaultdict(list)
        for case_id in case_ids:
            row = candidate_by_case[case_id]
            for dimension in dimensions:
                if dimension == "capability_tag":
                    values = row.get("capability_tags") or []
                else:
                    values = [str(row.get(dimension) or "unknown")]
                for value in values:
                    groups[(dimension, str(value))].append(case_id)
        minimum_reported = int(self.config["statistics"]["minimum_reported_slice_size"])
        minimum_gated = int(self.config["statistics"]["minimum_gated_slice_size"])
        output = []
        for (dimension, value), group_case_ids in sorted(groups.items()):
            count = len(group_case_ids)
            status = (
                "suppressed"
                if count < minimum_reported
                else "evaluated" if count >= minimum_gated else "descriptive_only"
            )
            row: Dict[str, Any] = {
                "dimension": dimension,
                "value": value,
                "name": f"{dimension}={value}",
                "pair_count": count,
                "status": status,
                "gated": status == "evaluated",
            }
            if status != "suppressed":
                before_pass = [
                    float(baseline_by_case[case_id]["hybrid"]["passed"])
                    for case_id in group_case_ids
                ]
                after_pass = [
                    float(candidate_by_case[case_id]["hybrid"]["passed"])
                    for case_id in group_case_ids
                ]
                before_score = [
                    float(baseline_by_case[case_id]["hybrid"]["overall_score"])
                    for case_id in group_case_ids
                ]
                after_score = [
                    float(candidate_by_case[case_id]["hybrid"]["overall_score"])
                    for case_id in group_case_ids
                ]
                pass_metric = self._metric(before_pass, after_pass, mode=mode, binary=True)
                score_metric = self._metric(before_score, after_score, mode=mode)
                row.update(
                    {
                        "pass_rate_delta": pass_metric["delta"],
                        "overall_score_delta": score_metric["delta"],
                        "pass_rate": pass_metric,
                        "overall_score": score_metric,
                    }
                )
            output.append(row)
        return output

    def _evidence(
        self,
        quality: Mapping[str, Any],
        *,
        pair_count: int,
        unresolved_count: int,
        can_compute: bool,
    ) -> Dict[str, Any]:
        if not can_compute:
            return {
                "status": "not_evaluated",
                "reasons": ["paired_comparison_not_computed"],
                "warnings": [],
                "pair_count": 0,
                "unresolved_pair_count": unresolved_count,
            }
        minimum = int(self.config["statistics"]["minimum_paired_cases"])
        if pair_count < minimum:
            return {
                "status": "insufficient_sample",
                "reasons": ["paired_case_count_below_threshold"],
                "warnings": [],
                "pair_count": pair_count,
                "minimum_pair_count": minimum,
                "unresolved_pair_count": unresolved_count,
            }
        pass_metric = quality["pass_rate"]
        interval = pass_metric["confidence_interval"]
        margin = float(
            self.config["decision_policy"]["pass_rate_noninferiority_margin"]
        )
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        p_value = float(pass_metric["two_sided_p_value"])
        alpha = float(self.config["statistics"]["alpha"])
        non_inferior = lower > -margin
        superior = lower > 0 and p_value < alpha
        inferior = upper < -margin
        if inferior:
            status = "evaluated"
            reasons = ["primary_metric_inferior"]
        elif not non_inferior:
            status = "inconclusive"
            reasons = ["noninferiority_not_established"]
        else:
            status = "evaluated"
            reasons = []
        return {
            "status": status,
            "reasons": reasons,
            "warnings": [],
            "pair_count": pair_count,
            "minimum_pair_count": minimum,
            "unresolved_pair_count": unresolved_count,
            "primary_metric": "pass_rate",
            "noninferiority_margin": margin,
            "non_inferior": non_inferior,
            "superior": superior,
            "inferior": inferior,
            "alpha": alpha,
        }

    def _release_decision(
        self,
        *,
        mode: str,
        comparability: Mapping[str, Any],
        evaluator_gate: Mapping[str, Any],
        evidence: Mapping[str, Any],
        approval_failures: Sequence[str],
        safety: Mapping[str, Any],
        quality: Mapping[str, Any],
        performance: Mapping[str, Any],
        slices: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        blocked = []
        if comparability["status"] == "not_comparable":
            blocked.extend(comparability["reasons"])
        if mode == "promotion" and evaluator_gate["status"] != "passed":
            blocked.extend(evaluator_gate["reasons"])
        blocked.extend(approval_failures)
        if blocked:
            return {
                "status": "blocked",
                "reasons": self._unique(blocked),
                "requires_human_approval": False,
            }
        if mode == "diagnostic":
            reasons = ["diagnostic_data_cannot_authorize_release"]
            if evidence["status"] != "evaluated":
                reasons.extend(evidence["reasons"])
            if safety.get("new_failure_count") or safety.get("new_veto_count"):
                reasons.append("safety_guardrail_regressed")
            return {
                "status": "diagnostic_only",
                "reasons": self._unique(reasons),
                "requires_human_approval": False,
            }

        rejection_reasons = []
        if safety.get("new_failure_count"):
            rejection_reasons.append("new_safety_failure")
        if safety.get("new_veto_count"):
            rejection_reasons.append("new_veto")
        if safety.get("new_p0_count"):
            rejection_reasons.append("new_p0_regression")
        if safety.get("new_l3_failure_count"):
            rejection_reasons.append("new_l3_pass_regression")
        pass_delta = float(quality["pass_rate"]["delta"])
        score_delta = float(quality["overall_score"]["delta"])
        if pass_delta < -float(
            self.config["decision_policy"]["pass_rate_noninferiority_margin"]
        ):
            rejection_reasons.append("pass_rate_regressed_beyond_margin")
        if score_delta < -float(
            self.config["decision_policy"]["overall_score_guardrail_margin"]
        ):
            rejection_reasons.append("overall_score_regressed")
        for row in slices:
            if row.get("status") != "evaluated":
                continue
            metric = row.get("pass_rate") or {}
            interval = metric.get("confidence_interval") or {}
            if float(interval.get("upper", 0)) < -float(
                self.config["decision_policy"]["pass_rate_noninferiority_margin"]
            ):
                rejection_reasons.append(f"gated_slice_regressed:{row['name']}")
            if float(row.get("overall_score_delta") or 0) < -float(
                self.config["decision_policy"]["overall_score_guardrail_margin"]
            ):
                rejection_reasons.append(f"gated_slice_score_regressed:{row['name']}")
        latency = performance.get("p95_latency_ms") or {}
        if latency.get("status") != "not_available" and float(latency.get("delta") or 0) > float(
            self.config["decision_policy"]["maximum_p95_latency_regression_ms"]
        ):
            rejection_reasons.append("p95_latency_regressed")
        cost = performance.get("average_cost") or {}
        if cost.get("status") != "not_available" and float(cost.get("delta") or 0) > float(
            self.config["decision_policy"]["maximum_average_cost_regression"]
        ):
            rejection_reasons.append("average_cost_regressed")
        if (
            self.config["decision_policy"]["require_performance_evidence_for_promotion"]
            and performance.get("status") != "available"
        ):
            rejection_reasons.append("performance_evidence_missing")
        if evidence["status"] == "evaluated" and evidence.get("inferior"):
            rejection_reasons.append("primary_metric_inferior")
        if rejection_reasons:
            return {
                "status": "keep_baseline",
                "reasons": self._unique(rejection_reasons),
                "requires_human_approval": False,
            }
        if evidence["status"] != "evaluated" or not evidence.get("non_inferior"):
            return {
                "status": "keep_baseline",
                "reasons": self._unique(
                    evidence.get("reasons") or ["release_evidence_inconclusive"]
                ),
                "requires_human_approval": False,
            }
        return {
            "status": "eligible_for_human_approval",
            "reasons": [
                "primary_metric_non_inferior",
                "no_new_safety_or_veto_regression",
                "quality_guardrails_passed",
            ],
            "requires_human_approval": True,
        }

    def _bad_cases(
        self,
        baseline_by_case: Mapping[str, Mapping[str, Any]],
        candidate_by_case: Mapping[str, Mapping[str, Any]],
        case_ids: Sequence[str],
        safety: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        entries = []
        new_safety = set(safety.get("new_failure_case_ids") or [])
        new_veto = set(safety.get("new_veto_case_ids") or [])
        for case_id in case_ids:
            before = baseline_by_case[case_id]
            after = candidate_by_case[case_id]
            baseline_hybrid = before["hybrid"]
            candidate_hybrid = after["hybrid"]
            score_deltas = {
                dimension: round(
                    float(candidate_hybrid["scores"][dimension])
                    - float(baseline_hybrid["scores"][dimension]),
                    4,
                )
                for dimension in self.rubric.dimension_ids
                if baseline_hybrid["scores"].get(dimension) is not None
                and candidate_hybrid["scores"].get(dimension) is not None
                and candidate_hybrid["scores"][dimension]
                < baseline_hybrid["scores"][dimension]
            }
            before_failures = set((before.get("deterministic") or {}).get("failures") or [])
            after_failures = set((after.get("deterministic") or {}).get("failures") or [])
            new_failures = sorted(after_failures - before_failures)
            reasons = list(new_failures)
            if baseline_hybrid["passed"] and not candidate_hybrid["passed"]:
                reasons.append("pass_regression")
            if score_deltas:
                reasons.append("score_regression")
            if case_id in new_safety:
                reasons.append("safety_regression")
            if case_id in new_veto:
                reasons.append("veto_regression")
            if (after.get("judge") or {}).get("status") == "error" and (
                before.get("judge") or {}
            ).get("status") != "error":
                reasons.append("judge_error")
            if not reasons:
                continue
            cause_codes = self._root_causes_for(reasons, new_failures)
            primary_cause = cause_codes[0]
            cause = self.config["root_causes"].get(primary_cause) or self.config[
                "root_causes"
            ]["score_regression"]
            priority = (
                "P0"
                if case_id in new_safety or case_id in new_veto
                else "P1" if "pass_regression" in reasons or "judge_error" in reasons else "P2"
            )
            entries.append(
                {
                    "case_id": case_id,
                    "priority": priority,
                    "failure_types": self._unique(reasons),
                    "evidence_codes": self._unique(
                        [*new_failures, *[f"dimension_delta:{key}:{value}" for key, value in score_deltas.items()]]
                    ),
                    "score_deltas": score_deltas,
                    "suspected_root_causes": cause_codes,
                    "root_cause": primary_cause,
                    "owner_module": cause["owner_module"],
                    "confidence": "high" if new_failures or priority == "P0" else "medium",
                    "status": "open",
                    "source_split": after.get("split"),
                    "scene": after.get("scene"),
                    "risk_level": after.get("risk_level"),
                }
            )
        entries.sort(key=lambda row: ({"P0": 0, "P1": 1, "P2": 2}[row["priority"]], row["case_id"]))
        return entries

    def _root_causes_for(
        self, reasons: Sequence[str], deterministic_failures: Sequence[str]
    ) -> list[str]:
        output = []
        for reason in reasons:
            if reason == "judge_error":
                output.append("judge_error")
            elif reason in {"safety_regression", "veto_regression"}:
                output.append("agent_safety_regression")
            elif reason == "score_regression":
                output.append("score_regression")
        output.extend(
            failure
            for failure in deterministic_failures
            if failure in self.config["root_causes"]
        )
        return self._unique(output or ["score_regression"])

    def _root_cause_summary(
        self, bad_cases: Sequence[Mapping[str, Any]]
    ) -> list[Dict[str, Any]]:
        grouped: Dict[str, list[str]] = defaultdict(list)
        for row in bad_cases:
            for code in row.get("suspected_root_causes") or []:
                grouped[code].append(row["case_id"])
        output = []
        for code, case_ids in grouped.items():
            details = self.config["root_causes"].get(code) or {}
            output.append(
                {
                    "root_cause": code,
                    "classification": "suspected",
                    "count": len(set(case_ids)),
                    "case_ids": sorted(set(case_ids)),
                    "owner_module": details.get("owner_module", "evaluation_operator"),
                    "action": details.get("action", "人工复核并设计独立复现实验"),
                }
            )
        output.sort(key=lambda row: (-row["count"], row["root_cause"]))
        return output

    def _recommendations(
        self, bad_cases: Sequence[Mapping[str, Any]]
    ) -> list[Dict[str, Any]]:
        summary = self._root_cause_summary(bad_cases)
        output = []
        for item in summary:
            digest = hashlib.sha256(
                f"{item['root_cause']}:{','.join(item['case_ids'])}".encode("utf-8")
            ).hexdigest()[:12]
            priority = min(
                (row["priority"] for row in bad_cases if row["case_id"] in item["case_ids"]),
                key=lambda value: {"P0": 0, "P1": 1, "P2": 2}[value],
            )
            output.append(
                {
                    "recommendation_id": f"rec-{digest}",
                    "priority": priority,
                    "status": "proposed",
                    "suspected_root_cause": item["root_cause"],
                    "owner_module": item["owner_module"],
                    "action": item["action"],
                    "source_case_ids": item["case_ids"],
                    "affected_case_count": item["count"],
                    "expected_metric": "pass_rate_and_regressed_dimensions",
                    "risk": "requires_human_review_before_any_prompt_or_model_change",
                    "validation_plan": {
                        "target_split": "regression",
                        "rerun_case_ids": item["case_ids"],
                        "acceptance": "no repeated failure and no new safety/veto regression",
                    },
                }
            )
        return output

    def _regression_candidates(
        self,
        bad_cases: Sequence[Mapping[str, Any]],
        candidate_by_case: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, list[Dict[str, Any]]]:
        policy = self.config["regression_candidate_policy"]
        allowed = set(policy["allowed_source_splits"])
        forbidden = set(policy["forbidden_source_splits"])
        proposed = []
        excluded = []
        for row in bad_cases:
            source_split = str(candidate_by_case[row["case_id"]].get("split") or "")
            if source_split in forbidden or source_split not in allowed:
                excluded.append(
                    {
                        "source_case_id": row["case_id"],
                        "source_split": source_split,
                        "reason": "source_split_not_allowed_for_regression_generation",
                    }
                )
                continue
            digest = hashlib.sha256(
                f"{row['case_id']}:{','.join(row['failure_types'])}".encode("utf-8")
            ).hexdigest()[:12]
            proposed.append(
                {
                    "candidate_id": f"reg-candidate-{digest}",
                    "source_case_id": row["case_id"],
                    "source_split": source_split,
                    "lineage": {
                        "trigger_failure_types": row["failure_types"],
                        "suspected_root_causes": row["suspected_root_causes"],
                    },
                    "target_split": "regression",
                    "status": "proposed",
                    "requires_human_approval": True,
                }
            )
        return {"proposed": proposed, "excluded": excluded}

    def _empty_analysis(self, *, pair_count: int, reason: str) -> Dict[str, Any]:
        metric = {
            "baseline": None,
            "candidate": None,
            "delta": None,
            "status": "not_evaluated",
            "reason": reason,
        }
        return {
            "quality_comparison": {
                "pass_rate": dict(metric),
                "overall_score": dict(metric),
                "dimensions": {},
            },
            "performance": {
                "status": "not_available",
                "p95_latency_ms": {"status": "not_available"},
                "average_cost": {"status": "not_available"},
                "average_tokens": {"status": "not_available"},
            },
            "safety": {
                "new_failure_count": 0,
                "new_failure_case_ids": [],
                "resolved_failure_count": 0,
                "resolved_failure_case_ids": [],
                "new_veto_count": 0,
                "new_veto_case_ids": [],
                "new_vetoes": {},
                "resolved_veto_count": 0,
                "resolved_vetoes": {},
                "new_l3_failure_count": 0,
                "new_l3_failure_case_ids": [],
                "new_p0_count": 0,
                "new_p0_case_ids": [],
                "resolved_p0_count": 0,
                "resolved_p0_case_ids": [],
            },
            "slices": [],
            "case_transitions": {
                "improved": 0,
                "regressed": 0,
                "unchanged": pair_count,
                "improved_case_ids": [],
                "regressed_case_ids": [],
            },
        }

    def _limitations(
        self,
        *,
        mode: str,
        performance: Mapping[str, Any],
        unresolved_count: int,
    ) -> list[str]:
        limitations = [
            "离线配对统计只反映当前案例分布，不替代线上灰度监控。",
            "根因是基于可见失败证据的 suspected attribution，必须人工复核。",
            "bootstrap cluster_unit=case_id；当前没有宣称 family-cluster inference。",
        ]
        if performance.get("status") != "available":
            limitations.append("延迟或成本数据不完整，性能项标记为 not_available，未按 0 处理。")
        if unresolved_count:
            limitations.append(f"有 {unresolved_count} 个未解析配对案例，未静默计入主指标。")
        if mode == "diagnostic":
            limitations.append("diagnostic 仅用于 dev/regression 迭代，不产生发布资格。")
        else:
            limitations.append("promotion 不输出 test 逐案 Bad Case、迭代建议或回归候选。")
        return limitations

    def _identity(self, report: Mapping[str, Any]) -> Dict[str, Any]:
        metadata = report.get("run_metadata") or {}
        dataset_versions = sorted(
            {
                str(row.get("dataset_version"))
                for row in report["cases"]
                if row.get("dataset_version")
            }
        )
        splits = sorted(
            {str(row.get("split")) for row in report["cases"] if row.get("split")}
        )
        claimed_dataset_versions = metadata.get("dataset_versions")
        if not isinstance(claimed_dataset_versions, list):
            claimed_dataset_versions = dataset_versions
        claimed_splits = metadata.get("splits")
        if not isinstance(claimed_splits, list):
            claimed_splits = splits
        model_metadata = {
            json.dumps(row.get("model_metadata") or {}, ensure_ascii=False, sort_keys=True)
            for row in report["cases"]
        }
        return {
            "run_id": metadata.get("run_id"),
            "variant": metadata.get("variant"),
            "current_commit": metadata.get("current_commit"),
            "result_sha256": metadata.get("result_sha256"),
            "dataset_sha256": metadata.get("dataset_sha256"),
            "case_set_sha256": metadata.get("case_set_sha256")
            or self._case_set_sha256(report["cases"]),
            "dataset_versions": sorted(str(value) for value in dataset_versions),
            "splits": sorted(str(value) for value in splits),
            "claimed_dataset_versions": sorted(
                str(value) for value in claimed_dataset_versions
            ),
            "claimed_splits": sorted(str(value) for value in claimed_splits),
            "pipeline_version": report.get("pipeline_version"),
            "rubric_version": report.get("rubric_version"),
            "prompt_version": report.get("prompt_version"),
            "judge_id": report.get("judge_id"),
            "config_sha256": report.get("config_sha256"),
            "rubric_sha256": report.get("rubric_sha256"),
            "model_snapshots": [json.loads(value) for value in sorted(model_metadata)],
        }

    @staticmethod
    def _nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _trusted_lineage_failures(
        self, report: Mapping[str, Any], *, side: str
    ) -> list[str]:
        failures = []
        if report.get("config_sha256") != self._sha256_bytes(
            DEFAULT_MACHINE_CONFIG_PATH.read_bytes()
        ):
            failures.append(f"{side}_machine_config_fingerprint_untrusted")
        if report.get("rubric_sha256") != self._sha256_bytes(self.rubric.path.read_bytes()):
            failures.append(f"{side}_rubric_fingerprint_untrusted")
        return failures

    @staticmethod
    def _resolved(row: Mapping[str, Any]) -> bool:
        hybrid = row.get("hybrid")
        return (
            isinstance(hybrid, Mapping)
            and type(hybrid.get("passed")) is bool
            and isinstance(hybrid.get("overall_score"), (int, float))
            and isinstance(hybrid.get("scores"), Mapping)
        )

    def _safety_failure(self, hybrid: Mapping[str, Any]) -> bool:
        score = (hybrid.get("scores") or {}).get("safety")
        vetoes = set(hybrid.get("vetoes") or [])
        return (score is not None and float(score) <= 1) or bool(vetoes & self.safety_vetoes)

    @staticmethod
    def _case_set_sha256(cases: Sequence[Mapping[str, Any]]) -> str:
        payload = "\n".join(sorted(str(row["case_id"]) for row in cases))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_json(value: Mapping[str, Any]) -> str:
        try:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("input report must be finite JSON data") from exc
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be a finite number")
        return normalized

    @staticmethod
    def _require_string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _require_timestamp(value: Any, name: str) -> str:
        value = EvaluationAnalysisService._require_string(value, name)
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError(f"{name} must include a timezone")
        return value

    @staticmethod
    def _unique(values: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(str(value) for value in values))
