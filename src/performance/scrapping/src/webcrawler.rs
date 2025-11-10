use csv;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::prelude::*;
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use spdlog;
use std::error::Error;
use std::time::Duration;
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
struct StSRecord {
    href: String,
    video: String,
    gloss: String,
    category: String,
}
fn extract_base_url(url: &str) -> String {
    // Buscar /word/NÚMERO/ y mantener hasta ahí
    if let Some(word_pos) = url.find("/word/") {
        let after_word = &url[word_pos + 6..];
        if let Some(slash_pos) = after_word.find('/') {
            return format!("{}/", &url[..word_pos + 6 + slash_pos]);
        }
    }
    url.to_string()
}
fn extract_record_video(
    client: &Client,
    record: &StSRecord,
    lang: &str,
) -> Result<StSRecord, Box<dyn Error + Send + Sync>> {
    const ROOT: &str = "https://spreadthesign.com";
    let href = record.href.replace("es.es", lang);
    let url = extract_base_url(&format!("{}{}", ROOT, href));
    let page = client.get(&url).send()?;
    let html = page.text()?;
    let document = Html::parse_document(&html);
    let selector = Selector::parse("video").unwrap();

    spdlog::info!("Procesando la glosa {}", record.gloss);

    let video = document
        .select(&selector)
        .next()
        .and_then(|video_elem| video_elem.value().attr("src"))
        .ok_or(format!(
            "No hay video para la glosa {}, en la lengua {}",
            record.gloss, lang
        ))?;
    Ok(StSRecord {
        href: href.to_string(),
        video: video.to_string(),
        gloss: record.gloss.clone(),
        category: record.category.clone(),
    })
}
fn get_sts_single_language(
    client: &Client,
    records: &Vec<StSRecord>,
    lang: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let results: Vec<Result<StSRecord, _>> = records
        .par_chunks(10)
        .flat_map(|chunk| {
            chunk
                .iter()
                .map(|record| extract_record_video(client, record, lang))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut writer = csv::Writer::from_path(format!("metadata.{}.csv", lang))?;
    let mut successful = 0;
    let mut failed = 0;

    for result in results {
        match result {
            Ok(res) => {
                writer.serialize(res)?;
                successful += 1;
            }
            Err(_) => {
                failed += 1;
            }
        }
    }

    writer.flush()?;

    spdlog::info!("✅ {}: {} exitosos, {} fallidos", lang, successful, failed);

    Ok(())
}
fn get_sts_multilingual(file: &str, langs: Vec<&str>) -> Result<(), Box<dyn Error + Sync + Send>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_path(file)?;
    let records: Vec<StSRecord> = reader
        .deserialize()
        .collect::<Result<Vec<StSRecord>, _>>()?;
    for lang in langs {
        spdlog::info!("Se va a procesar la lengua {}", lang);
        get_sts_single_language(&client, &records, lang)?;
    }
    Ok(())
}

#[pyfunction]
pub fn get_the_meat_balls(file: String, langs: Vec<String>) -> PyResult<()> {
    let lang_refs = langs.iter().map(|lang| lang.as_str()).collect();
    get_sts_multilingual(&file, lang_refs)
        .map_err(|e| PyRuntimeError::new_err(format!("Error: {}", e)))?;
    Ok(())
}
