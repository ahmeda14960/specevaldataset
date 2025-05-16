"""Statement classes for representing testable statements from specifications."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class StatementType(Enum):
    """Types of statements that can be extracted from specifications."""

    PROHIBITION = "prohibition"  # Something the model must not do
    REQUIREMENT = "requirement"  # Something the model must do
    GUIDELINE = "guideline"  # Something the model should generally do
    DEFINITION = "definition"  # A definition that informs other statements

    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive lookup."""
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return None  # Let the default ValueError be raised


class AuthorityLevel(Enum):
    """Authority levels as defined in the OpenAI Model Spec."""

    PLATFORM = "platform"  # Cannot be overridden
    DEVELOPER = "developer"  # Can be overridden by platform
    USER = "user"  # Can be overridden by platform or developer
    GUIDELINE = "guideline"  # Can be implicitly overridden

    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive lookup."""
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return None  # Let the default ValueError be raised


@dataclass
class Statement:
    """
    A testable statement extracted from a specification.

    Attributes:
        id: Unique identifier for the statement
        text: The raw text of the statement
        type: Type of statement (prohibition, requirement, etc.)
        authority_level: Level of authority for the statement
        section: Section of the specification where the statement was found
        subsection: Subsection where the statement was found
        related_statements: IDs of related statements
        metadata: Additional metadata about the statement
    """

    id: str
    text: str
    type: StatementType
    authority_level: AuthorityLevel
    section: str
    subsection: Optional[str] = None
    related_statements: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values for collections if None is provided."""
        if self.related_statements is None:
            self.related_statements = []
        if self.metadata is None:
            self.metadata = {}
