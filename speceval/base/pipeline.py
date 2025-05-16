"""Pipeline interface for running evaluations in the SpecEval framework."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import datetime
import json

from .organization import Organization
from .parser import Specification
from .statement import Statement


@dataclass
class TestCase:
    """A single test case for evaluating compliance."""

    statement: Statement
    input_text: str
    output_text: str
    is_compliant: bool
    confidence: float
    explanation: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationResults:
    """Results of a compliance evaluation."""

    specification: Specification
    organization: Organization
    test_cases: List[TestCase]
    timestamp: datetime.datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Initialize default metadata if None is provided."""
        if self.metadata is None:
            self.metadata = {}

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        total_cases = len(self.test_cases)
        compliant_cases = sum(1 for tc in self.test_cases if tc.is_compliant)
        compliance_rate = compliant_cases / total_cases if total_cases > 0 else 0

        # Break down by statement type
        by_type = {}
        for tc in self.test_cases:
            type_value = tc.statement.type.value
            if type_value not in by_type:
                by_type[type_value] = {"total": 0, "compliant": 0}
            by_type[type_value]["total"] += 1
            if tc.is_compliant:
                by_type[type_value]["compliant"] += 1

        # Add compliance rates to by_type
        for type_stats in by_type.values():
            type_stats["compliance_rate"] = (
                type_stats["compliant"] / type_stats["total"] if type_stats["total"] > 0 else 0
            )

        # Break down by authority level
        by_authority = {}
        for tc in self.test_cases:
            auth_value = tc.statement.authority_level.value
            if auth_value not in by_authority:
                by_authority[auth_value] = {"total": 0, "compliant": 0}
            by_authority[auth_value]["total"] += 1
            if tc.is_compliant:
                by_authority[auth_value]["compliant"] += 1

        # Add compliance rates to by_authority
        for auth_stats in by_authority.values():
            auth_stats["compliance_rate"] = (
                auth_stats["compliant"] / auth_stats["total"] if auth_stats["total"] > 0 else 0
            )

        return {
            "total_cases": total_cases,
            "compliant_cases": compliant_cases,
            "compliance_rate": compliance_rate,
            "by_type": by_type,
            "by_authority": by_authority,
            "timestamp": self.timestamp.isoformat(),
            "organization": self.organization.get_info(),
            "specification": {
                "name": self.specification.name,
                "source_path": str(self.specification.source_path)
                if self.specification.source_path
                else None,
            },
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Convert results to JSON and optionally save to file.

        Args:
            path: Optional path to save the JSON file.

        Returns:
            JSON string of the results.
        """
        results_dict = {
            "summary": self.get_summary(),
            "test_cases": [
                {
                    "statement_id": tc.statement.id,
                    "input_text": tc.input_text,
                    "output_text": tc.output_text,
                    "is_compliant": tc.is_compliant,
                    "confidence": tc.confidence,
                    "explanation": tc.explanation,
                    "metadata": tc.metadata,
                }
                for tc in self.test_cases
            ],
            "metadata": self.metadata,
        }

        json_str = json.dumps(results_dict, indent=2)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    def to_html(self, path: str) -> None:
        """
        Generate an HTML report of the results and save to file.

        Args:
            path: Path to save the HTML file.
        """
        summary = self.get_summary()

        # Simple HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SpecEval Compliance Report: {summary["organization"]["name"]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .test-case {{ background-color: #fff; padding: 15px; border: 1px solid #ddd; margin-bottom: 10px; border-radius: 5px; }}
                .compliant {{ border-left: 5px solid #4CAF50; }}
                .non-compliant {{ border-left: 5px solid #F44336; }}
                pre {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; overflow: auto; }}
                .progress-bar {{
                    height: 20px;
                    background-color: #e0e0e0;
                    border-radius: 10px;
                    margin-bottom: 10px;
                }}
                .progress {{
                    height: 100%;
                    background-color: #4CAF50;
                    border-radius: 10px;
                    text-align: center;
                    line-height: 20px;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <h1>SpecEval Compliance Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Organization:</strong> {summary["organization"]["name"]}</p>
                <p><strong>Specification:</strong> {summary["specification"]["name"]}</p>
                <p><strong>Timestamp:</strong> {summary["timestamp"]}</p>

                <h3>Overall Compliance</h3>
                <div class="progress-bar">
                    <div class="progress" style="width: {summary["compliance_rate"] * 100}%;">
                        {round(summary["compliance_rate"] * 100, 1)}%
                    </div>
                </div>
                <p>{summary["compliant_cases"]} compliant out of {summary["total_cases"]} test cases</p>

                <h3>By Statement Type</h3>
                <table border="1" cellpadding="5" cellspacing="0">
                    <tr>
                        <th>Type</th>
                        <th>Compliant</th>
                        <th>Total</th>
                        <th>Rate</th>
                    </tr>
        """

        for type_name, stats in summary["by_type"].items():
            html += f"""
                    <tr>
                        <td>{type_name}</td>
                        <td>{stats["compliant"]}</td>
                        <td>{stats["total"]}</td>
                        <td>{round(stats["compliance_rate"] * 100, 1)}%</td>
                    </tr>
            """

        html += """
                </table>

                <h3>By Authority Level</h3>
                <table border="1" cellpadding="5" cellspacing="0">
                    <tr>
                        <th>Authority</th>
                        <th>Compliant</th>
                        <th>Total</th>
                        <th>Rate</th>
                    </tr>
        """

        for auth_name, stats in summary["by_authority"].items():
            html += f"""
                    <tr>
                        <td>{auth_name}</td>
                        <td>{stats["compliant"]}</td>
                        <td>{stats["total"]}</td>
                        <td>{round(stats["compliance_rate"] * 100, 1)}%</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <h2>Test Cases</h2>
        """

        for tc in self.test_cases:
            compliant_class = "compliant" if tc.is_compliant else "non-compliant"
            compliant_text = "Compliant" if tc.is_compliant else "Non-compliant"

            html += f"""
            <div class="test-case {compliant_class}">
                <h3>Statement {tc.statement.id}: {compliant_text} ({round(tc.confidence * 100)}% confidence)</h3>
                <p><strong>Statement:</strong> {tc.statement.text}</p>
                <p><strong>Type:</strong> {tc.statement.type.value}</p>
                <p><strong>Authority:</strong> {tc.statement.authority_level.value}</p>

                <h4>Input:</h4>
                <pre>{tc.input_text}</pre>

                <h4>Output:</h4>
                <pre>{tc.output_text}</pre>

                <h4>Explanation:</h4>
                <p>{tc.explanation}</p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


class Pipeline(ABC):
    """Base class for evaluation pipelines."""

    def __init__(
        self,
        specification: Specification,
        organization: Organization,
        num_test_cases: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a Pipeline object.

        Args:
            specification: The specification to evaluate against.
            organization: The organization providing models for evaluation.
            num_test_cases: Number of test cases to generate.
            metadata: Optional additional metadata.
        """
        self.specification = specification
        self.organization = organization
        self.num_test_cases = num_test_cases
        self.metadata = metadata or {}

    @abstractmethod
    def run(self) -> EvaluationResults:
        """
        Run the evaluation pipeline.

        Returns:
            EvaluationResults object with the results of the evaluation.
        """
        pass
