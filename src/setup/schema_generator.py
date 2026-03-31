"""Schema generator - proposes state structure from rules."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..engine.rules_index import RulesIndex
from ..llm.client import LLMClient


@dataclass
class SchemaProposal:
    """A proposed state schema with explanation."""

    description: str  # Human-readable explanation
    schema: dict  # JSON schema
    example_state: dict  # Example initial state
    tracked_elements: list[str]  # What the schema tracks


SCHEMA_PROMPT = """You are designing a state representation for a board game. Based on the rules provided, propose a JSON schema that can track all necessary game state.

GAME: {game_name}

RULES SUMMARY:
{rules_text}

Design a state schema that tracks:
1. Game phase/round information
2. Each player's resources, pieces, cards, etc.
3. Board state (spaces, areas, tracks)
4. Any accumulating resources or effects
5. Turn order and current player

Return ONLY valid JSON in this format:
{{
  "description": "Human-readable explanation of what the schema tracks",
  "tracked_elements": [
    "list of",
    "things being tracked"
  ],
  "schema": {{
    "type": "object",
    "properties": {{
      "phase": {{"type": "string"}},
      "round": {{"type": "integer"}},
      "current_player": {{"type": "string"}},
      "players": {{
        "type": "object",
        "additionalProperties": {{
          "type": "object",
          "properties": {{
            // player-specific properties
          }}
        }}
      }},
      "board": {{
        "type": "object",
        "properties": {{
          // board-specific properties
        }}
      }}
    }}
  }},
  "example_state": {{
    // A valid example initial state matching the schema
  }}
}}

Be specific to the game. Include all resources, pieces, and trackable elements mentioned in the rules."""


class SchemaGenerator:
    """Generates state schemas from game rules."""

    def __init__(self, client: LLMClient, game_name: str):
        """Initialize the schema generator.

        Args:
            client: LLM client for generation.
            game_name: Name of the game.
        """
        self.client = client
        self.game_name = game_name

    async def propose_schema(
        self,
        rules_index: RulesIndex | None = None,
        rules_text: str | None = None,
    ) -> SchemaProposal:
        """Propose a state schema based on the rules.

        Args:
            rules_index: Optional RulesIndex for comprehensive rules.
            rules_text: Optional raw rules text (used if no index).

        Returns:
            SchemaProposal with the proposed schema.
        """
        # Get rules text
        if rules_text is None and rules_index is not None:
            # Get all chunks for comprehensive understanding
            chunks = rules_index.get_all_chunks()
            rules_text = "\n\n".join(
                f"## {c.title}\n{c.content}" for c in chunks[:10]  # Limit to avoid token overflow
            )
        elif rules_text is None:
            rules_text = "(No rules provided)"

        # Generate schema proposal
        prompt = SCHEMA_PROMPT.format(
            game_name=self.game_name,
            rules_text=rules_text[:8000],  # Limit rules text
        )

        response = await self.client.complete(
            system_prompt="You are a game state designer. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=4096,
        )

        try:
            data = response.as_json()
        except json.JSONDecodeError:
            # Return a minimal default schema
            return SchemaProposal(
                description="Basic game state (schema generation failed)",
                schema={
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string"},
                        "round": {"type": "integer"},
                        "current_player": {"type": "string"},
                        "players": {"type": "object"},
                        "board": {"type": "object"},
                    },
                },
                example_state={
                    "phase": "setup",
                    "round": 1,
                    "current_player": "player1",
                    "players": {},
                    "board": {},
                },
                tracked_elements=["phase", "round", "players", "board"],
            )

        return SchemaProposal(
            description=data.get("description", ""),
            schema=data.get("schema", {}),
            example_state=data.get("example_state", {}),
            tracked_elements=data.get("tracked_elements", []),
        )

    async def refine_schema(
        self,
        current_schema: dict,
        feedback: str,
        rules_text: str | None = None,
    ) -> SchemaProposal:
        """Refine a schema based on user feedback.

        Args:
            current_schema: The current schema to refine.
            feedback: User feedback on what to change.
            rules_text: Optional rules text for context.

        Returns:
            Refined SchemaProposal.
        """
        prompt = f"""Refine this game state schema based on the feedback.

GAME: {self.game_name}

CURRENT SCHEMA:
{json.dumps(current_schema, indent=2)}

USER FEEDBACK:
{feedback}

{f"RULES CONTEXT:{chr(10)}{rules_text[:4000]}" if rules_text else ""}

Return the updated schema in the same JSON format:
{{
  "description": "Updated explanation",
  "tracked_elements": ["updated", "list"],
  "schema": {{ ... }},
  "example_state": {{ ... }}
}}"""

        response = await self.client.complete(
            system_prompt="You are a game state designer. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.5,
        )

        try:
            data = response.as_json()
            return SchemaProposal(
                description=data.get("description", ""),
                schema=data.get("schema", current_schema),
                example_state=data.get("example_state", {}),
                tracked_elements=data.get("tracked_elements", []),
            )
        except json.JSONDecodeError:
            # Return unchanged
            return SchemaProposal(
                description="Schema unchanged (refinement failed)",
                schema=current_schema,
                example_state={},
                tracked_elements=[],
            )

    def format_proposal_for_user(self, proposal: SchemaProposal) -> str:
        """Format a proposal for user review.

        Args:
            proposal: The schema proposal.

        Returns:
            Human-readable formatted string.
        """
        lines = [
            "=== Proposed State Schema ===",
            "",
            proposal.description,
            "",
            "Tracking:",
        ]

        for element in proposal.tracked_elements:
            lines.append(f"  - {element}")

        lines.extend(
            [
                "",
                "Example initial state:",
                json.dumps(proposal.example_state, indent=2),
            ]
        )

        return "\n".join(lines)


async def generate_schema_interactive(
    client: LLMClient,
    game_name: str,
    rules_index: RulesIndex | None = None,
    rules_text: str | None = None,
) -> dict:
    """Interactive schema generation with user confirmation.

    Args:
        client: LLM client.
        game_name: Name of the game.
        rules_index: Optional rules index.
        rules_text: Optional raw rules text.

    Returns:
        Confirmed schema dictionary.
    """
    generator = SchemaGenerator(client, game_name)

    print("Generating state schema proposal...")
    proposal = await generator.propose_schema(rules_index, rules_text)

    while True:
        print(generator.format_proposal_for_user(proposal))
        print()

        response = input("Accept schema? (yes/no/feedback): ").strip().lower()

        if response in ("yes", "y", ""):
            return proposal.schema

        elif response in ("no", "n"):
            feedback = input("What should be changed? ").strip()
            if feedback:
                print("Refining schema...")
                proposal = await generator.refine_schema(
                    proposal.schema,
                    feedback,
                    rules_text,
                )

        else:
            # Treat as feedback
            print("Refining schema...")
            proposal = await generator.refine_schema(
                proposal.schema,
                response,
                rules_text,
            )
