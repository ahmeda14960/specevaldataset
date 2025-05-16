"""Parser for extracting structured statements from Markdown-formatted specifications."""

import re
from typing import List, Optional, Dict
from ..base import SpecificationParser, Statement, StatementType, AuthorityLevel


class MarkdownParser(SpecificationParser):
    """Parser for extracting structured statements from Markdown-formatted specifications.

    This parser identifies headings, content, and examples from markdown files and
    converts them into Statement objects. It supports hierarchical sections and
    custom authority levels specified in the markdown.
    """

    def __init__(self):
        """Initialize a new MarkdownParser instance."""
        super().__init__()

    def parse(self, content: str) -> List[Statement]:
        """Parse markdown content into a list of structured statements.

        Args:
            content: The markdown content to parse.

        Returns:
            A list of Statement objects representing the structured content.
        """
        # First, parse all statements as before
        raw_statements = self._extract_all_statements(content)

        # Then consolidate statements with the same base ID
        consolidated_statements = self._consolidate_statements(raw_statements)

        return consolidated_statements

    def _extract_all_statements(self, content: str) -> List[Statement]:
        """Extract all statement objects from the markdown content.

        This method identifies all headings and extracts content sections and examples
        that will be converted to Statement objects.

        Args:
            content: The markdown content to parse.

        Returns:
            A list of Statement objects corresponding to the content sections.
        """
        statements = []
        # Match headings with 2 to 4 '#' characters.
        # Group 1: the hashes, Group 2: heading text, Group 3: optional id, Group 4: optional authority.
        heading_pattern = re.compile(
            r"^(#{2,4})\s+([^#{]+?)\s*(?:{#([^}]+)})?(?:\s+authority=([a-z]+))?\s*$", re.MULTILINE
        )
        matches = list(heading_pattern.finditer(content))
        num_matches = len(matches)
        current_section = None
        current_subsection = None

        for i, match in enumerate(matches):
            hashes = match.group(1)
            heading_text = match.group(2).strip()
            section_id = match.group(3)
            authority_str = match.group(4)
            level = len(hashes)
            # Level 2 headings are main sections.
            if level == 2:
                current_section = heading_text
                current_subsection = None
            else:
                current_subsection = heading_text

            # Determine authority level.
            if authority_str:
                try:
                    authority_level = AuthorityLevel(authority_str)
                except ValueError:
                    authority_level = AuthorityLevel.GUIDELINE
            else:
                authority_level = AuthorityLevel.GUIDELINE

            # Extract content from this heading until the next heading.
            start = match.end()
            end = matches[i + 1].start() if i + 1 < num_matches else len(content)
            section_content = content[start:end].strip()

            # Ensure a section name is set.
            if not current_section:
                current_section = "unknown"

            section_statements = self._extract_statements_from_section(
                section_content, current_section, current_subsection, section_id, authority_level, i
            )
            statements.extend(section_statements)
        return statements

    def _extract_statements_from_section(
        self,
        section_content: str,
        section: str,
        subsection: Optional[str],
        section_id: Optional[str],
        authority_level: AuthorityLevel,
        position: int,
    ) -> List[Statement]:
        """Extract Statement objects from a specific section of the markdown.

        This method processes a section of content, identifying the main content and any example blocks,
        and converts them into Statement objects.

        Args:
            section_content: The content of the section to process.
            section: The name of the section.
            subsection: The name of the subsection, if applicable.
            section_id: The ID of the section, if specified in the markdown.
            authority_level: The authority level of the statements in this section.
            position: The position of the section in the document for ordering.

        Returns:
            A list of Statement objects representing the section content and examples.
        """
        statements = []
        # Extract example blocks starting with "**Example**:"
        example_pattern = re.compile(r"\*\*Example\*\*:.*?(?=\n\s*\n|\Z)", re.DOTALL)
        examples = []

        def replacement(match):
            examples.append(match.group(0).strip())
            return ""  # Remove example text from the main content.

        working_content = example_pattern.sub(replacement, section_content).strip()

        # Create the base ID that will be used for consolidation
        base_id = section_id or section.lower().replace(" ", "_")

        # Create one statement from the main content if it's substantive.
        if working_content and len(working_content.split()) >= 5:
            stmt_id = f"{base_id}_main"
            stmt = Statement(
                id=stmt_id,
                text=working_content,
                type=self._determine_statement_type(section),
                authority_level=authority_level,
                section=section,
                subsection=subsection,
                metadata={"base_id": base_id, "position": position, "is_main": True},
            )
            statements.append(stmt)

        # Create separate statements for each example block (if substantive).
        for idx, ex in enumerate(examples):
            if len(ex.split()) >= 10 or "```" in ex:
                ex_id = f"{base_id}_example_{idx+1}"
                ex_stmt = Statement(
                    id=ex_id,
                    text=ex,
                    type=self._determine_statement_type(section),
                    authority_level=authority_level,
                    section=section,
                    subsection=subsection,
                    metadata={"base_id": base_id, "position": position, "example_index": idx + 1},
                )
                statements.append(ex_stmt)
        return statements

    def _consolidate_statements(self, statements: List[Statement]) -> List[Statement]:
        """Consolidate statements with the same base ID into a single statement.

        This method combines related statements (such as main content and examples) that share
        a base ID into a single, consolidated Statement object.

        Args:
            statements: The list of statements to consolidate.

        Returns:
            A list of consolidated Statement objects.
        """
        # Group statements by base_id
        statements_by_base_id: Dict[str, List[Statement]] = {}

        for stmt in statements:
            base_id = stmt.metadata.get("base_id")
            if not base_id:
                # If no base_id, treat as unique
                continue

            if base_id not in statements_by_base_id:
                statements_by_base_id[base_id] = []
            statements_by_base_id[base_id].append(stmt)

        # Consolidate statements
        consolidated = []

        # Process statements that don't need consolidation (no base_id)
        for stmt in statements:
            base_id = stmt.metadata.get("base_id")
            if not base_id:
                consolidated.append(stmt)

        # Get a list of base_ids to process to avoid modifying the dictionary during iteration
        base_ids_to_process = list(statements_by_base_id.keys())

        # Process statements that need consolidation
        for base_id in base_ids_to_process:
            stmt_group = statements_by_base_id[base_id]
            if len(stmt_group) <= 1:
                # No need for consolidation
                consolidated.append(stmt_group[0])
                continue

            # Sort by position first, then by whether it's a main statement or example
            stmt_group.sort(
                key=lambda s: (
                    s.metadata.get("position", 0),
                    0 if s.metadata.get("is_main") else s.metadata.get("example_index", 1),
                )
            )

            # Take the first statement as the base
            base_stmt = stmt_group[0]

            # Combine texts from all statements in the group
            combined_text = ""
            for s in stmt_group:
                if combined_text and s.text:
                    combined_text += "\n\n"
                combined_text += s.text

            # Create new consolidated statement
            consolidated_stmt = Statement(
                id=base_id,  # Use the base_id instead of statement ID
                text=combined_text,
                type=base_stmt.type,
                authority_level=base_stmt.authority_level,
                section=base_stmt.section,
                subsection=base_stmt.subsection,
                related_statements=base_stmt.related_statements,
                metadata=base_stmt.metadata.copy(),
            )

            # Remove the processing metadata entries
            if "base_id" in consolidated_stmt.metadata:
                del consolidated_stmt.metadata["base_id"]
            if "position" in consolidated_stmt.metadata:
                del consolidated_stmt.metadata["position"]
            if "is_main" in consolidated_stmt.metadata:
                del consolidated_stmt.metadata["is_main"]
            if "example_index" in consolidated_stmt.metadata:
                del consolidated_stmt.metadata["example_index"]

            # Add to consolidated list
            consolidated.append(consolidated_stmt)

        return consolidated

    def _determine_statement_type(self, section: str) -> StatementType:
        """Determine the statement type based on the section heading text.

        This method analyzes the section heading to infer whether it represents a requirement,
        prohibition, definition, or general guideline.

        Args:
            section: The section heading text to analyze.

        Returns:
            The inferred StatementType for statements in this section.
        """
        if not section:
            return StatementType.GUIDELINE
        section_lower = section.lower()
        if "do not" in section_lower or "don't" in section_lower or "avoid" in section_lower:
            return StatementType.PROHIBITION
        elif "must" in section_lower or "required" in section_lower or "should" in section_lower:
            return StatementType.REQUIREMENT
        elif "definition" in section_lower or "meaning" in section_lower:
            return StatementType.DEFINITION
        else:
            return StatementType.GUIDELINE
