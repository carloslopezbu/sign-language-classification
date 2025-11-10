use pyo3::prelude::*;

mod trie;
use trie::_Trie;

mod webcrawler;
use webcrawler::get_the_meat_balls;

#[pymodule]
fn scrapping(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_Trie>()?;
    m.add_function(wrap_pyfunction!(get_the_meat_balls, m)?)?;
    Ok(())
}
