"""
Gloss Buffer Service for SignBridge.
Stores committed stable ASL glosses, prevents consecutive duplicate recognition commits,
and provides buffer management utilities for sentence generation pipelines.
"""

from typing import List, Optional


class GlossBuffer:
    """
    Manages an ordered sequence of committed ASL gloss tokens.
    Filters consecutive duplicate commits while preserving natural word order and non-consecutive repetitions.
    """

    def __init__(self, initial_glosses: Optional[List[str]] = None):
        self._glosses: List[str] = []
        if initial_glosses:
            for g in initial_glosses:
                self.add_gloss(g)

    @staticmethod
    def clean_gloss(gloss: Optional[str]) -> str:
        """
        Sanitize and normalize a raw ASL gloss string.
        Strips whitespace, converts to uppercase, removes numeric suffixes (e.g. WATER1 -> WATER),
        removes dataset tags (e.g. THIS/IT -> THIS), and filters placeholder symbols.
        """
        if not gloss:
            return ""
        s = str(gloss).strip().upper()
        if s in ("—", "ANALYZING…", "ANALYZING..."):
            return ""
        import re
        s = re.sub(r"\d+$", "", s)
        if s.endswith("/IT"):
            s = s[:-3]
        return s.strip()

    def add_gloss(self, gloss: Optional[str]) -> bool:
        """
        Add a gloss token to the buffer.
        
        Rules:
        1. Empty or invalid glosses are ignored (returns False).
        2. Consecutive duplicate commits are rejected (e.g. WATER -> WATER -> WATER returns False for duplicates).
        3. Non-consecutive repeated glosses are accepted (e.g. I -> WANT -> WATER -> I -> WANT -> WATER).
        
        Returns:
            bool: True if the gloss was committed to the buffer, False if rejected.
        """
        cleaned = self.clean_gloss(gloss)
        if not cleaned:
            return False

        # Reject consecutive duplicate prediction commit
        if self._glosses and self._glosses[-1] == cleaned:
            return False

        self._glosses.append(cleaned)
        return True

    def get_glosses(self) -> List[str]:
        """Return a shallow copy of the current list of glosses."""
        return list(self._glosses)

    def to_string(self) -> str:
        """Return clean space-separated gloss string."""
        return " ".join(self._glosses)

    def clear(self) -> None:
        """Clear all glosses from the buffer."""
        self._glosses.clear()

    def remove_at(self, index: int) -> bool:
        """Remove gloss at specified 0-based index."""
        if 0 <= index < len(self._glosses):
            self._glosses.pop(index)
            return True
        return False

    def pop_last(self) -> Optional[str]:
        """Remove and return the most recently added gloss."""
        if self._glosses:
            return self._glosses.pop()
        return None

    def __len__(self) -> int:
        return len(self._glosses)

    def __getitem__(self, item: int) -> str:
        return self._glosses[item]

    def __repr__(self) -> str:
        return f"GlossBuffer(glosses={self._glosses})"
