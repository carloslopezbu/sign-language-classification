use pyo3::prelude::*;

mod trie;
use trie::_Trie;

mod webcrawler;

#[pymodule]
fn scrapping(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_Trie>()?;
    Ok(())
}
