use fxhash::FxBuildHasher;
use pyo3::prelude::*;
use std::collections::HashMap;

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

#[derive(Default, Debug)]
struct _TrieNode {
    is_end_of_word: bool,
    children: FxHashMap<char, _TrieNode>,
}

#[pyclass]
#[derive(Default, Debug)]
pub struct _Trie {
    root: _TrieNode,
}

impl _Trie {
    /// Recorre el trie y acumula palabras
    fn collect_words(node: &_TrieNode, prefix: &mut String, words: &mut Vec<String>) {
        if node.is_end_of_word {
            words.push(prefix.clone());
        }

        for (ch, child) in &node.children {
            prefix.push(*ch);
            _Trie::collect_words(child, prefix, words);
            prefix.pop();
        }
    }
}

#[pymethods]
impl _Trie {
    #[new]
    pub fn new() -> Self {
        _Trie {
            root: _TrieNode::default(),
        }
    }

    pub fn insert(&mut self, word: &str) {
        let mut current_node = &mut self.root;

        for c in word.chars() {
            current_node = current_node.children.entry(c).or_default();
        }
        current_node.is_end_of_word = true;
    }

    pub fn contains(&self, word: &str) -> bool {
        let mut current_node = &self.root;

        for c in word.chars() {
            match current_node.children.get(&c) {
                Some(node) => current_node = node,
                None => return false,
            }
        }

        current_node.is_end_of_word
    }

    /// Devuelve todas las palabras en un solo string
    pub fn to_str(&self) -> String {
        let mut words = Vec::new();
        let mut prefix = String::new();
        _Trie::collect_words(&self.root, &mut prefix, &mut words);
        words.join("\n")
    }
}

type Trie = _Trie;
