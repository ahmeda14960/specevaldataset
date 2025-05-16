"""Parser interface and specification container for the SpecEval framework."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import os
import pathlib

from .statement import Statement


class SpecificationParser(ABC):
    """Abstract base class for parsing specifications into testable statements."""

    @abstractmethod
    def parse(self, content: str) -> List[Statement]:
        """
        Parse specification content into a list of testable statements.

        Args:
            content: The specification content as a string.

        Returns:
            A list of Statement objects.
        """
        pass

    @classmethod
    def from_file(cls, file_path: Union[str, pathlib.Path]) -> "Specification":
        """Create a Specification from a file.

        Args:
            file_path: Path to the specification file.

        Returns:
            A Specification object.
        """
        parser = cls()

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        statements = parser.parse(content)
        return Specification(
            name=os.path.basename(file_path),
            source_path=file_path,
            statements=statements,
        )


class Specification:
    """Represents a complete specification with parsed statements."""

    def __init__(
        self,
        name: str,
        statements: List[Statement],
        source_path: Optional[Union[str, pathlib.Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a Specification object.

        Args:
            name: The name of the specification.
            statements: List of Statement objects parsed from the specification.
            source_path: Optional path to the source file.
            metadata: Optional additional metadata about the specification.
        """
        self.name = name
        self.statements = statements
        self.source_path = source_path
        self.metadata = metadata or {}

        # Create lookup by ID for efficient access
        self.statement_lookup = {stmt.id: stmt for stmt in statements}

    def get_statement(self, statement_id: str) -> Optional[Statement]:
        """Get a statement by ID."""
        return self.statement_lookup.get(statement_id)

    def get_statements_by_type(self, statement_type: str) -> List[Statement]:
        """Get statements by type."""
        return [s for s in self.statements if s.type.value == statement_type]

    def get_statements_by_authority(self, authority_level: str) -> List[Statement]:
        """Get statements by authority level."""
        return [s for s in self.statements if s.authority_level.value == authority_level]

    def get_statements_by_section(self, section: str) -> List[Statement]:
        """Get statements by section."""
        return [s for s in self.statements if s.section == section]
