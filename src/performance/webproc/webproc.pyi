"""webproc module, built in Rust 🦀."""

from typing import List

class Trie:
    """Trie data structure for eficient word storing and checking."""

    def __init__(self) -> None:
        """Creates an empty Trie."""
        ...

    def insert(self, word: str) -> None:
        """Inserts a word in the Trie."""
        ...

    def contains(self, word: str) -> bool:
        """Checks if a word is in the Trie."""
        ...

    def to_str(self) -> str:
        """Returns a string with all the words stored in the Trie, separated with \\n."""
        ...

def get_the_meat_balls(metadata_path: str, languages: List[str]) -> None:
    """
    Scraps videos from the SpreadTheSign webpage, for the specified languages, using a CSV file as reference.

    Args:
        metadata_path: Path to the metadate CSV file.
        languages: List of the language codes (e.g: ["fr.fr", "de.at"])
    """
    ...
