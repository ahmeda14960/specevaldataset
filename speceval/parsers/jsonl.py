"""Parser for extracting structured statements from JSONL-formatted specifications."""

import json
from typing import List

from ..base import SpecificationParser, Statement, StatementType, AuthorityLevel


class JsonlParser(SpecificationParser):
    """Parser for extracting structured statements from JSONL-formatted specifications.

    This parser reads JSONL files where each line contains a JSON object representing
    a statement with fields that match the Statement class attributes.
    """

    def __init__(self):
        """Initialize a new JsonlParser instance."""
        super().__init__()

    def parse(self, content: str) -> List[Statement]:
        """Parse JSONL content into a list of structured statements.

        Args:
            content: The JSONL content to parse as a string.

        Returns:
            A list of Statement objects representing the structured content.
        """
        statements = []

        # Process each non-empty line as a JSON object
        for line in content.strip().split("\n"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # Map string values to enum values
                statement_type = self._parse_statement_type(data.get("type", "guideline"))
                authority_level = self._parse_authority_level(
                    data.get("authority_level", "guideline")
                )

                # Create a Statement object
                statement = Statement(
                    id=data.get("id", ""),
                    text=data.get("text", ""),
                    type=statement_type,
                    authority_level=authority_level,
                    section=data.get("section", ""),
                    subsection=data.get("subsection"),
                    related_statements=data.get("related_statements", []),
                    metadata=data.get("metadata", {}),
                )

                statements.append(statement)
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

        return statements

    def _parse_statement_type(self, type_str: str) -> StatementType:
        """Convert a string to a StatementType enum value.

        Args:
            type_str: String representation of the statement type.

        Returns:
            Corresponding StatementType enum value.
        """
        try:
            # Try to directly map the string to an enum value
            return StatementType(type_str.lower())
        except ValueError:
            # Default to GUIDELINE if the string doesn't match any enum value
            return StatementType.GUIDELINE

    def _parse_authority_level(self, authority_str: str) -> AuthorityLevel:
        """Convert a string to an AuthorityLevel enum value.

        Args:
            authority_str: String representation of the authority level.

        Returns:
            Corresponding AuthorityLevel enum value.
        """
        try:
            # Try to directly map the string to an enum value
            return AuthorityLevel(authority_str.lower())
        except ValueError:
            # Default to GUIDELINE if the string doesn't match any enum value
            return AuthorityLevel.GUIDELINE
