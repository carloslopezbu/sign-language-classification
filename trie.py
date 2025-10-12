from scrapping import _Trie


class Trie:
    def __init__(self) -> None:
        self.trie: _Trie = _Trie()

    def insert(self, word: str) -> None:
        self.trie.insert(word)

    def contains(self, word: str) -> bool:
        return self.trie.contains(word)

    def to_str(self) -> str:
        return self.trie.to_str()
