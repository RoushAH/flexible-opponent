"""Hidden state dealer - manages hidden information setup."""

import json
import random
from dataclasses import dataclass, field

from ..llm.client import LLMClient


@dataclass
class DealingResult:
    """Result of dealing hidden state."""

    ai_hidden: dict
    human_hidden: dict
    remaining_deck: list  # Cards/items not dealt
    dealing_notes: str


DEALING_PROMPT = """You are setting up hidden information for a board game.

GAME: {game_name}

RULES FOR DEALING/SETUP:
{rules_text}

HUMAN PLAYER HAS DECLARED THESE ITEMS:
{human_declared}

FULL POOL/DECK AVAILABLE:
{full_pool}

Based on the rules, determine:
1. What items should the AI player receive?
2. How many items does each player get?
3. Are there any special dealing rules (draft, choose, random)?

If the rules specify random dealing, randomly select from items NOT in the human's hand.
If the rules allow choice, select good but not overpowered items for the AI.

Return ONLY valid JSON:
{{
  "ai_items": ["list", "of", "items", "for", "ai"],
  "dealing_method": "random|draft|choice|fixed",
  "items_per_player": 7,
  "remaining_pool": ["items", "not", "dealt"],
  "notes": "Brief explanation of dealing rules applied"
}}"""


class HiddenDealer:
    """Manages dealing of hidden game information."""

    def __init__(
        self,
        client: LLMClient,
        game_name: str,
        rules_index=None,
    ):
        """Initialize the dealer.

        Args:
            client: LLM client.
            game_name: Name of the game.
            rules_index: Optional rules index for looking up dealing rules.
        """
        self.client = client
        self.game_name = game_name
        self.rules_index = rules_index

    async def deal_hidden_state(
        self,
        human_declared: dict,
        full_pool: dict | None = None,
        rules_text: str | None = None,
    ) -> DealingResult:
        """Deal hidden state to the AI based on what human declared.

        Args:
            human_declared: What the human player has (hand, bonuses, etc.)
            full_pool: Full pool of items to deal from (optional, LLM can infer).
            rules_text: Rules for dealing (optional, uses rules_index).

        Returns:
            DealingResult with AI's and human's hidden state.
        """
        # Get dealing rules if not provided
        if rules_text is None and self.rules_index is not None:
            rules_text = self.rules_index.get_relevant_rules(
                phase="setup",
                context="dealing cards hand setup starting",
                n_results=5,
            )
        elif rules_text is None:
            rules_text = "(No specific dealing rules found)"

        # Format the pool
        if full_pool is None:
            pool_text = "(Determine from rules)"
        else:
            pool_text = json.dumps(full_pool, indent=2)

        prompt = DEALING_PROMPT.format(
            game_name=self.game_name,
            rules_text=rules_text,
            human_declared=json.dumps(human_declared, indent=2),
            full_pool=pool_text,
        )

        response = await self.client.complete(
            system_prompt="You are a fair card dealer. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.7,  # Some randomness for dealing
        )

        try:
            data = response.as_json()

            # Build AI's hidden state
            ai_hidden = {
                "hand": data.get("ai_items", []),
                "dealt_by": "referee",
                "dealing_method": data.get("dealing_method", "unknown"),
            }

            return DealingResult(
                ai_hidden=ai_hidden,
                human_hidden=human_declared,
                remaining_deck=data.get("remaining_pool", []),
                dealing_notes=data.get("notes", ""),
            )

        except json.JSONDecodeError as e:
            # Fallback: empty hands - but log what went wrong
            error_preview = response.content[:200] if response.content else "empty"
            return DealingResult(
                ai_hidden={"hand": [], "dealt_by": "referee_fallback"},
                human_hidden=human_declared,
                remaining_deck=[],
                dealing_notes=f"JSON parse failed: {e}. Response: {error_preview}...",
            )

    async def deal_from_deck(
        self,
        deck: list,
        human_hand: list,
        cards_per_player: int,
    ) -> DealingResult:
        """Simple dealing: random cards from deck excluding human's hand.

        For games with straightforward random dealing.

        Args:
            deck: Full deck of cards/items.
            human_hand: Cards the human has.
            cards_per_player: How many cards each player gets.

        Returns:
            DealingResult with dealt hands.
        """
        # Remove human's cards from available pool
        available = [card for card in deck if card not in human_hand]

        # Randomly select for AI
        if len(available) >= cards_per_player:
            ai_hand = random.sample(available, cards_per_player)
        else:
            ai_hand = available.copy()

        # Update remaining
        remaining = [card for card in available if card not in ai_hand]

        return DealingResult(
            ai_hidden={
                "hand": ai_hand,
                "dealt_by": "referee",
                "dealing_method": "random",
            },
            human_hidden={"hand": human_hand},
            remaining_deck=remaining,
            dealing_notes=f"Randomly dealt {len(ai_hand)} cards to AI",
        )

    async def reveal_card(
        self,
        player: str,
        card: str,
        current_ai_hidden: dict,
        current_human_hidden: dict,
    ) -> tuple[dict, dict]:
        """Handle a card being revealed/played.

        Updates hidden state when a card moves from hidden to visible.

        Args:
            player: Who revealed the card ("ai" or human name).
            card: The card being revealed.
            current_ai_hidden: Current AI hidden state.
            current_human_hidden: Current human hidden state.

        Returns:
            Tuple of (updated_ai_hidden, updated_human_hidden).
        """
        ai_hidden = current_ai_hidden.copy()
        human_hidden = current_human_hidden.copy()

        if player == "ai":
            # Remove from AI's hand
            if "hand" in ai_hidden and card in ai_hidden["hand"]:
                ai_hidden["hand"] = [c for c in ai_hidden["hand"] if c != card]
        else:
            # Remove from human's hand
            if "hand" in human_hidden and card in human_hidden["hand"]:
                human_hidden["hand"] = [c for c in human_hidden["hand"] if c != card]

        return ai_hidden, human_hidden

    async def draw_card(
        self,
        player: str,
        remaining_deck: list,
        current_ai_hidden: dict,
        current_human_hidden: dict,
        card: str | None = None,
    ) -> tuple[dict, dict, list, str]:
        """Handle a player drawing a card.

        Args:
            player: Who is drawing ("ai" or human name).
            remaining_deck: Cards available to draw.
            current_ai_hidden: Current AI hidden state.
            current_human_hidden: Current human hidden state.
            card: Specific card drawn (if known, e.g., human declares it).

        Returns:
            Tuple of (ai_hidden, human_hidden, remaining_deck, card_drawn).
        """
        ai_hidden = current_ai_hidden.copy()
        human_hidden = current_human_hidden.copy()

        if not remaining_deck:
            return ai_hidden, human_hidden, [], ""

        # Determine which card
        if card is not None and card in remaining_deck:
            drawn = card
        else:
            drawn = random.choice(remaining_deck)

        # Update deck
        new_deck = [c for c in remaining_deck if c != drawn]

        # Add to appropriate hand
        if player == "ai":
            if "hand" not in ai_hidden:
                ai_hidden["hand"] = []
            ai_hidden["hand"].append(drawn)
        else:
            if "hand" not in human_hidden:
                human_hidden["hand"] = []
            human_hidden["hand"].append(drawn)

        return ai_hidden, human_hidden, new_deck, drawn


async def setup_hidden_state_interactive(
    client: LLMClient,
    game_name: str,
    players: list[str],
    rules_index=None,
) -> tuple[dict, dict]:
    """Interactive setup for hidden game state.

    Args:
        client: LLM client.
        game_name: Name of the game.
        players: Player names (first non-"ai" is human).
        rules_index: Optional rules index.

    Returns:
        Tuple of (ai_hidden, human_hidden) dicts.
    """
    dealer = HiddenDealer(client, game_name, rules_index)

    print("\n=== Hidden State Setup ===")
    print("The referee will manage hidden information (hands, secret bonuses, etc.)")
    print()

    # Ask human to declare their hidden items
    human_name = next((p for p in players if p != "ai"), "human")

    print(f"{human_name}, please declare your hidden items to the referee.")
    print("(The AI will NOT see this - only the referee knows both hands)")
    print()
    print("Enter your hand/hidden items as JSON (e.g., [\"Card A\", \"Card B\"]):")
    print("Or press Enter if there's no hidden state:")

    human_input = input("> ").strip()

    if human_input:
        try:
            human_items = json.loads(human_input)
            human_hidden = {"hand": human_items}
        except json.JSONDecodeError:
            # Treat as comma-separated list
            human_items = [item.strip() for item in human_input.split(",")]
            human_hidden = {"hand": human_items}
    else:
        human_hidden = {"hand": []}

    print(f"\nRecorded {len(human_hidden.get('hand', []))} hidden item(s) for {human_name}.")

    # Now deal to AI
    if human_hidden.get("hand"):
        print("\nDealing hidden items to AI based on rules...")
        result = await dealer.deal_hidden_state(human_hidden)
        print(f"  {result.dealing_notes}")
        print(f"  AI received {len(result.ai_hidden.get('hand', []))} item(s)")

        return result.ai_hidden, result.human_hidden
    else:
        return {"hand": []}, {"hand": []}
