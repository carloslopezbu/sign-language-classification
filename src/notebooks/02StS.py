import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" # _Procesamiento de los datos extraidos de Spread the Sign_""").center()
    return


@app.cell
def _():
    import polars as pl
    import os
    return (pl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Exploración de los datos""")
    return


@app.cell
def _():
    metadata_path: str = "./datamining/clean/StS/metadata.csv"
    return (metadata_path,)


@app.cell
def _(metadata_path: str, pl):
    df_metadata = pl.read_csv(metadata_path)
    df_metadata
    return (df_metadata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.1 Visualización""")
    return


@app.cell
def _(df_metadata, mo):
    mo.ui.data_explorer(df_metadata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Como podemos observar el campo de datos **`"grammar-category`"** esta muy desvalanceada hace la categoría nombre, la descartaremos.""")
    return


@app.cell
def _(df_metadata):
    df_without_grammar = df_metadata.drop("grammar-category")
    df_without_grammar
    return (df_without_grammar,)


@app.cell
def _(df_without_grammar, mo):
    mo.ui.data_explorer(df_without_grammar)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Vamos a ver ahora si tenemos campos duplicados""")
    return


@app.cell
def _(df_without_grammar):
    cleaned = df_without_grammar.unique(subset=["video", "category"], keep="first")

    cleaned = cleaned.filter( \
        (cleaned["category"] != "lengua-de-signos-para-principiantes") & \
        (cleaned["category"] != "lengua-de-signos-para-bebes")\
    )
    return (cleaned,)


@app.cell
def _(cleaned, mo):
    mo.ui.data_explorer(cleaned)
    return


@app.cell
def _(cleaned):
    metadata_cleaned_path: str = "./metadata.es.es.csv"
    cleaned.write_csv(metadata_cleaned_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Arreglar los ficheros de las otras lenguas""")
    return


@app.cell
def _(pl):
    def load_lang_dataframes(root: str, langs: list[str]) -> list[pl.DataFrame]:
        return [ pl.read_csv(f"{root}metadata.{lang}.csv")  for lang in langs ]
    return (load_lang_dataframes,)


@app.cell
def _(pl):
    def fix_dataframe(correct: pl.DataFrame, wrong: pl.DataFrame) -> pl.DataFrame:
        return  (
        wrong
        .drop("category")
        .join(
            correct.select(["gloss", "category"]), 
            on="gloss", 
            how="left"  # Mantiene todas las filas del DataFrame incorrecto
        )
    )
    return (fix_dataframe,)


@app.cell
def _():
    langs = ["es.ar", "en.us", "en.gb", "it.it", "pt.pt"]
    root = "./datamining/clean/StS/"
    return langs, root


@app.cell
def _(langs, load_lang_dataframes, root):
    dfs = load_lang_dataframes(root=root, langs=langs)
    return (dfs,)


@app.cell
def _(cleaned, dfs, fix_dataframe):
    fixed = [fix_dataframe(correct=cleaned, wrong=df) for df in dfs]
    return (fixed,)


@app.cell
def _(fixed, langs, root):
    for lang, df in zip(langs, fixed):
        save_path = f"{root}metadata.{lang}.fixed.csv"
        df.write_csv(save_path)
    return


@app.cell
def _(fixed, pl):
    whole = pl.concat(fixed, how="vertical")
    whole = whole.filter(
        pl.col("category").is_not_null()
        ).unique(
            subset=["video", "category"], 
            keep="first"
        )
    return (whole,)


@app.cell
def _(whole):
    whole
    return


@app.cell
def _(mo, whole):
    mo.ui.data_explorer(whole)
    return


@app.cell
def _(pl, whole):
    final = whole.with_columns(
        pl.col("category").replace(old=["oraciones", "religion"], new=["lenguaje", "arte-y-entretenimiento"])
    )
    return (final,)


@app.cell
def _(final, mo):
    mo.ui.data_explorer(final)
    return


if __name__ == "__main__":
    app.run()
