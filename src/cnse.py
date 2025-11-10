import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    import requests
    from bs4 import BeautifulSoup
    from unidecode import unidecode  # para tranformar carácteres
    import json
    return BeautifulSoup, json, requests, unidecode


@app.cell
def _():
    # Definimos algunas constantes
    cnse_autocomplete_api: str = "https://fundacioncnse-dilse.org/php/buscador-autocompletar_nuevo.php"  # url de autocompletado de la LSE
    lse_dictionary_api: str = "https://fundacioncnse.org/educa/bancolse/listado-de-signos.php#gsc.tab=0"  # url del diccionario LSE
    return cnse_autocomplete_api, lse_dictionary_api


@app.cell
def _():
    headers: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    }  # Header para hacer peticiones al servidor del diccionario LSE

    headers_with_json: dict[str, str] = (
        headers  # Header para hacer peticiones a la api de autocompletado de LSE
    )
    headers["Acept"] = "application/json"
    return headers, headers_with_json


@app.cell
def _(
    BeautifulSoup,
    headers: dict[str, str],
    json,
    lse_dictionary_api: str,
    requests,
    unidecode,
):


    rep = requests.get(
        lse_dictionary_api, headers=headers, timeout=1000
    )  # Obtenemos el html del diccionario LSE
    soup = BeautifulSoup(rep.text, "html.parser")  # Parseamos html

    ids = ["cat"] + [
        "cat" + str(i) for i in range(2, 13)
    ]  # Creamos los id's del contenido que nos interesa (cat1 no existe)

    # Recuparemos enlace y texto de las categorias almacenados en los <a/>
    refs = [
        {"href": a["href"], "text": unidecode(a.get_text()).lower()}
        for div in [soup.find("div", id=id) for id in ids]  # recorre los divs
        if div  # solo si existe
        for a in div.find_all("a", href=True)  # recorre los <a> dentro de cada div
    ]

    for r in refs:
        if "/" in r["text"]:
            masc, *female = r["text"].split("/")  # quitar palabras masculino/femenino
            r["text"] = masc

        if "(" in r["text"]:
            masc, *female = r["text"].split(
                "("
            )  # quitar especificación de tipo de palabra
            r["text"] = masc

        if "?" in r["text"]:
            r["text"] = r["text"].split("?")[1]  # quitar interrogante ?palabra?

        if "!" in r["text"]:
            r["text"] = r["text"].split("!")[1]  # quitar exclamciones !palabra!

    with open("dict.json", "w") as j:
        json.dump(refs, j, indent=3)
    return (refs,)


@app.cell
def _():
    # Extracción del autocompletado usando como prefijo solo el abecedario
    from string import ascii_uppercase as words
    return


@app.cell
def _(cnse_autocomplete_api: str, headers_with_json: dict[str, str], requests):
    # Llamadas a la api de autocompletación

    def autocomplete(query: str):
        params = {"buscar": query}
        r = requests.get(
            cnse_autocomplete_api, params=params, headers=headers_with_json, timeout=10
        )
        r.raise_for_status()
        return r.json()
    return (autocomplete,)


@app.cell
def _():
    from trie import Trie
    return (Trie,)


@app.cell
def _(Trie, autocomplete, refs, requests):
    # Llamada a la api de autocompletado con las palabras del diccionario LSE, guardadas eficientemente en un Trie

    def get_lse_words(trie: Trie):
        for ref in refs:
            try:
                text: str = ref["text"]  # ya esta en minúsculas
                splits: list[str] = text.split(" ") + [
                    text
                ]  # dividimos los espacios incluyendo la misma palabra

                for split in splits:
                    if not trie.contains(split):  # no hacemos llamadas innecesarias
                        completions = autocomplete(split)

                        if completions:
                            print(completions)
                            for comp in completions:
                                trie.insert(comp["value"])  # añadir palabras

            except requests.HTTPError:
                print(f"Ocurrio un error en la palabra {text}")
    return (get_lse_words,)


@app.cell
def _(Trie):
    # creamos el Trie
    trie: Trie = Trie()

    trie.insert("Holaa")
    trie.to_str()
    return (trie,)


@app.cell
def _(get_lse_words, trie: "Trie"):
    get_lse_words(trie)
    return


if __name__ == "__main__":
    app.run()
