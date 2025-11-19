use pyo3::prelude::*;

mod trie;
use trie::Trie;

mod webcrawler;
use webcrawler::get_the_meat_balls;

#[pymodule]
fn webproc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Trie>()?;
    m.add_function(wrap_pyfunction!(get_the_meat_balls, m)?)?;
    Ok(())
}
