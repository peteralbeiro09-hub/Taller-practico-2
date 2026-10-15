"""CLI del asistente de EcoMarket con RAG (Taller Práctico #2).

Uso:
    python app.py indexar [--estrategia estructura|parrafos|fijo]
    python app.py fragmentos [--estrategia ...] [--fuente faq.json]
    python app.py buscar "<texto>" [--k 4]
    python app.py preguntar "<texto>" [--version v1|v2] [--sin-rag] [--k 4] [--umbral 0.8] [--mostrar-prompt]
    python app.py chat
    python app.py experimento-recuperacion
    python app.py evaluar [--solo rag|cadena] [--parcial]
    python app.py probar
"""

import argparse
import json
import statistics
import sys
from datetime import datetime

from src.config import (
    CHUNK_SOLAPE, CHUNK_TAMANO, DIR_RESULTADOS, FECHA_REFERENCIA, MODELO_EMBEDDINGS, RAG_K, RAG_UMBRAL,
    ErrorConfiguracion,
)
from src.llm import ErrorLLM, RespuestaPendiente, activar_cache, completar, nombre_modelo

PENDIENTE = "⏳ Pendiente: aún no se ha obtenido del modelo (límite del endpoint gratuito)."


def cmd_probar(_args):
    print(f"Modelo: {nombre_modelo()}")
    print(completar([{"role": "user", "content": "Hola"}]))


# ---------- Fase 2: base de conocimiento ----------

def cmd_indexar(args):
    from src.indice import construir_indice
    from src.ingesta import fragmentar

    fragmentos = fragmentar(args.estrategia)
    print(f"Modelo de embeddings: {MODELO_EMBEDDINGS} (la primera vez se descarga, ~470 MB)", file=sys.stderr)
    construir_indice(fragmentos)
    conteo: dict[str, int] = {}
    for f in fragmentos:
        conteo[f.metadata["fuente"]] = conteo.get(f.metadata["fuente"], 0) + 1
    print(f"Índice creado en chroma_db/ con {len(fragmentos)} fragmentos (estrategia: {args.estrategia}):")
    for fuente, n in conteo.items():
        print(f"  {fuente:<28} {n:>3} fragmentos")


def cmd_fragmentos(args):
    from src.ingesta import fragmentar

    for f in fragmentar(args.estrategia):
        if args.fuente and f.metadata["fuente"] != args.fuente:
            continue
        print(f"--- {f.metadata['id']} | {f.metadata['seccion']} | {len(f.page_content)} caracteres ---")
        print(f.page_content, "\n")


def cmd_buscar(args):
    """Solo recuperación (embeddings + Chroma), sin LLM: gratis y sirve para calibrar k y el umbral."""
    from src import rag

    for i, (doc, score) in enumerate(rag.recuperar(args.texto, k=args.k, umbral=0), start=1):
        marca = "✔" if score >= RAG_UMBRAL else "✘ bajo el umbral"
        print(f"[{i}] sim={score:.3f} {marca}  {doc.metadata['fuente']} › {doc.metadata['seccion']}")
        print("    " + doc.page_content.replace("\n", "\n    ")[:400], "\n")


# ---------- Fase 3: preguntar y conversar ----------

def _imprimir(resultado: dict):
    print(resultado["respuesta"])
    print(f"\n   [ruta: {resultado['ruta']}]")
    for fuente in resultado.get("fuentes", []):
        print(f"   {fuente}")


def cmd_preguntar(args):
    from src import prompts, rag

    if args.mostrar_prompt:
        if args.sin_rag:
            mensajes = prompts.construir_mensajes("rag", "sin_rag", args.texto)
        else:
            pares = rag.recuperar(args.texto, args.k, args.umbral)
            mensajes = prompts.construir_mensajes("rag", args.version, args.texto,
                                                  {"contexto": rag.formatear_contexto(pares)})
        for m in mensajes:
            print(f"--- [{m['role']}] ---\n{m['content']}\n")
        return
    if args.sin_rag:
        _imprimir(rag.responder_sin_rag(args.texto))
    else:
        _imprimir(rag.responder(args.texto, args.version, args.k, args.umbral))


def cmd_chat(_args):
    from src import cadena

    print(f"Asistente virtual de EcoMarket con RAG ({nombre_modelo()}). Escribe 'salir' para terminar.\n")
    while True:
        try:
            mensaje = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if mensaje.lower() in ("salir", "exit", "quit"):
            break
        if not mensaje:
            continue
        resultado = cadena.responder(mensaje)
        print("EcoMarket: ", end="")
        _imprimir(resultado)
        print()


# ---------- Tablas Markdown ----------

def _celda(texto) -> str:
    """Hace que un texto quepa en una celda de tabla Markdown."""
    if texto is None:
        return ""
    return str(texto).strip().replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _encabezado(titulo: str, descripcion: str, con_llm: bool = True) -> list[str]:
    modelo = f"`{nombre_modelo()}` · **Temperatura:** 0 · " if con_llm else ""
    return [
        f"# {titulo}",
        "",
        f"- **LLM:** {modelo}**Embeddings:** `{MODELO_EMBEDDINGS}` · **k:** {RAG_K} · **Umbral:** {RAG_UMBRAL}"
        if con_llm else f"- **Embeddings:** `{MODELO_EMBEDDINGS}` · **Base vectorial:** ChromaDB (coseno)",
        f"- **Ejecutado:** {datetime.now():%Y-%m-%d %H:%M} · **Fecha de referencia:** {FECHA_REFERENCIA}",
        "- Salidas reales generadas por `python app.py`, sin edición manual.",
        "",
        descripcion,
        "",
    ]


def _escribir(nombre: str, lineas: list[str]):
    DIR_RESULTADOS.mkdir(exist_ok=True)
    archivo = DIR_RESULTADOS / nombre
    archivo.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Escrito results/{archivo.name}")


# ---------- Experimento de recuperación (sin LLM) ----------

def _rango(pares, evidencia: list[str]) -> int | None:
    """Posición (1..k) del primer fragmento que contiene toda la evidencia esperada, o None si no aparece."""
    for i, (doc, _) in enumerate(pares, start=1):
        if all(e.lower() in doc.page_content.lower() for e in evidencia):
            return i
    return None


def _unidades(texto: str) -> int:
    """Cuántos productos o preguntas frecuentes distintos caben en un fragmento (lo ideal es 1 o 0)."""
    return texto.count("Producto:") + texto.count("Pregunta frecuente")


def cmd_experimento_recuperacion(_args):
    """Compara estrategias de chunking y calibra el umbral midiendo solo la recuperación (gratis, sin LLM)."""
    from src.indice import construir_indice
    from src.ingesta import ESTRATEGIAS, fragmentar
    from src.contexto import casos_prueba

    casos = casos_prueba()
    info, fuera = casos["informacion"], casos["fuera_de_dominio"]
    k_max = 5
    lineas = _encabezado(
        "Experimento — Estrategias de segmentación y umbral de similitud",
        "Mide solo la **recuperación** (embeddings + ChromaDB), sin llamar al LLM. Un acierto significa que, entre "
        f"los primeros k fragmentos, hay uno que contiene **toda** la evidencia esperada del caso "
        f"(columna *Evidencia* de `data/casos_prueba.json`). Tamaño de chunk: {CHUNK_TAMANO} caracteres; "
        f"solape (solo en la estrategia por estructura): {CHUNK_SOLAPE}.",
        con_llm=False,
    )

    resumen = ["## 1. Resumen por estrategia", "",
               "| Estrategia | Fragmentos | Largo medio (car.) | Mezclan varios productos o preguntas | Acierto@1 | Acierto@3 | Acierto@5 | MRR |",
               "|---|---|---|---|---|---|---|---|"]
    detalle = ["## 2. Posición del fragmento correcto por caso", "",
               "| Caso | Consulta | Evidencia | " + " | ".join(ESTRATEGIAS) + " |",
               "|---|---|---|" + "---|" * len(ESTRATEGIAS)]
    posiciones: dict[str, list] = {}
    sims_info, sims_fuera = [], []

    for estrategia in ESTRATEGIAS:
        print(f"Indexando en memoria con la estrategia {estrategia}...", file=sys.stderr)
        fragmentos = fragmentar(estrategia)
        bd = construir_indice(fragmentos, directorio=None, coleccion=f"exp_{estrategia}")
        rangos = []
        for caso in info:
            pares = bd.similarity_search_with_relevance_scores(caso["consulta"], k=k_max)
            rangos.append(_rango(pares, caso["evidencia"]))
            if estrategia == "estructura":
                sims_info.append((caso, pares[0][1], pares[0][0]))
        if estrategia == "estructura":
            for caso in fuera:
                pares = bd.similarity_search_with_relevance_scores(caso["consulta"], k=1)
                sims_fuera.append((caso, pares[0][1], pares[0][0]))
        posiciones[estrategia] = rangos
        n = len(rangos)
        acierto = lambda k: sum(1 for r in rangos if r and r <= k) / n
        mrr = sum(1 / r for r in rangos if r) / n
        largo = statistics.mean(len(f.page_content) for f in fragmentos)
        mezclan = sum(1 for f in fragmentos if _unidades(f.page_content) > 1)
        resumen.append(f"| {estrategia} | {len(fragmentos)} | {largo:.0f} | {mezclan} | {acierto(1):.0%} | {acierto(3):.0%} | "
                       f"{acierto(5):.0%} | {mrr:.2f} |")

    for i, caso in enumerate(info):
        celdas = [str(posiciones[e][i]) if posiciones[e][i] else "✘ no está en el top 5" for e in ESTRATEGIAS]
        detalle.append(f"| {caso['id']} | {_celda(caso['consulta'])} | {_celda(' + '.join(caso['evidencia']))} | "
                       + " | ".join(celdas) + " |")

    umbral = ["## 3. Calibración del umbral (estrategia por estructura)", "",
              f"Similitud coseno del **mejor** fragmento para cada consulta. Umbral configurado: **{RAG_UMBRAL}** "
              "(`RAG_UMBRAL` en `.env`). Por debajo de él, el asistente responde que no tiene la información "
              "sin llamar al LLM.", "",
              "| Caso | Tipo | Consulta | Similitud máx. | ¿Supera el umbral? | Mejor fragmento |", "|---|---|---|---|---|---|"]
    for tipo, filas in (("En dominio", sims_info), ("Fuera de dominio", sims_fuera)):
        for caso, sim, doc in filas:
            umbral.append(f"| {caso['id']} | {tipo} | {_celda(caso['consulta'])} | {sim:.3f} | "
                          f"{'sí' if sim >= RAG_UMBRAL else 'no'} | {doc.metadata['fuente']} › {_celda(doc.metadata['seccion'])} |")
    min_info = min(s for _, s, _ in sims_info)
    max_fuera = max(s for _, s, _ in sims_fuera)
    umbral += ["", f"- Mínima similitud en dominio: **{min_info:.3f}** · Máxima fuera de dominio: **{max_fuera:.3f}**."]

    _escribir("experimento_recuperacion.md", lineas + resumen + [""] + detalle + [""] + umbral)


# ---------- Evaluación con LLM ----------

def _evaluar_rag(casos: dict) -> list[str]:
    from src import rag

    lineas = _encabezado(
        "Resultados — Sin RAG vs RAG v1 vs RAG v2",
        "Cada consulta se responde de tres formas: **sin RAG** (el LLM solo con su rol), **RAG v1** (contexto "
        "recuperado sin reglas, sin umbral) y **RAG v2** (prompt estricto con citas + umbral de similitud). "
        "La línea **Esperado** es el criterio con que se analiza cada respuesta.",
    )
    variantes = (
        ("Sin RAG", lambda c: rag.responder_sin_rag(c)),
        ("RAG v1", lambda c: rag.responder(c, "v1", umbral=0)),
        ("RAG v2", lambda c: rag.responder(c, "v2")),
    )
    for grupo, titulo in (("informacion", "Preguntas sobre EcoMarket"), ("fuera_de_dominio", "Preguntas fuera de la base de conocimiento")):
        lineas += [f"## {titulo}", ""]
        for caso in casos[grupo]:
            lineas += [f"### {caso['id']} — {caso['caso']}", "", f"**Consulta:** {caso['consulta']}", "",
                       f"**Esperado:** {caso['esperado']}", "", "| Variante | Respuesta | Ruta y fuentes |", "|---|---|---|"]
            for nombre, funcion in variantes:
                print(f"  {caso['id']} {nombre}...", file=sys.stderr)
                try:
                    r = funcion(caso["consulta"])
                    lineas.append(f"| {nombre} | {_celda(r['respuesta'])} | "
                                  f"{_celda(chr(10).join([r['ruta']] + r['fuentes']))} |")
                except RespuestaPendiente:
                    lineas.append(f"| {nombre} | {PENDIENTE} | |")
            lineas.append("")
    return lineas


def _evaluar_cadena(casos: dict) -> list[str]:
    from src import cadena

    lineas = _encabezado(
        "Resultados — Cadena completa con RAG (regresión del Taller 1 + casos nuevos)",
        "Cada consulta pasa por la cadena completa: clasificar → enrutar → responder (pedidos con búsqueda exacta, "
        "devoluciones v4 con política y catálogo recuperados por RAG, información con RAG v2) o escalar a un humano.",
    )
    lineas += ["| Caso | Consulta | Esperado | Clasificación | Ruta | Respuesta al cliente |", "|---|---|---|---|---|---|"]
    seleccion = casos["pedidos"] + casos["devoluciones"] + [c for c in casos["informacion"] if c["id"] in ("I1", "I5", "I7", "I12")] \
        + [c for c in casos["fuera_de_dominio"] if c["id"] in ("F1", "F4")]
    for caso in seleccion:
        print(f"  cadena {caso['id']}...", file=sys.stderr)
        try:
            r = cadena.responder(caso["consulta"])
        except RespuestaPendiente:
            lineas.append(f"| {caso['id']} — {_celda(caso['caso'])} | {_celda(caso['consulta'])} | "
                          f"{_celda(caso['esperado'])} | | | {PENDIENTE} |")
            continue
        clasif = json.dumps(r["clasificacion"], ensure_ascii=False) if r["clasificacion"] else "(no interpretable)"
        lineas.append(f"| {caso['id']} — {_celda(caso['caso'])} | {_celda(caso['consulta'])} | {_celda(caso['esperado'])} | "
                      f"`{_celda(clasif)}` | {_celda(r['ruta'])} | {_celda(r['respuesta'])} |")
    return lineas


def cmd_evaluar(args):
    from src.contexto import casos_prueba

    casos = casos_prueba()
    DIR_RESULTADOS.mkdir(exist_ok=True)
    activar_cache(DIR_RESULTADOS / ".cache_respuestas.json", solo_cache=args.parcial)
    trabajos = {
        "rag": ("resultados_rag.md", lambda: _evaluar_rag(casos)),
        "cadena": ("resultados_cadena.md", lambda: _evaluar_cadena(casos)),
    }
    for nombre, (archivo, trabajo) in trabajos.items():
        if args.solo and args.solo != nombre:
            continue
        print(f"Evaluando {nombre}...", file=sys.stderr)
        _escribir(archivo, trabajo())


def main():
    # La consola de Windows no usa UTF-8 por defecto; sin esto se dañan tildes y eñes.
    for flujo in (sys.stdout, sys.stderr, sys.stdin):
        flujo.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Asistente de atención al cliente de EcoMarket con RAG")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("probar", help="Envía 'Hola' al modelo configurado en .env").set_defaults(func=cmd_probar)

    p = sub.add_parser("indexar", help="Carga, segmenta, vectoriza y guarda la base de conocimiento en chroma_db/")
    p.add_argument("--estrategia", choices=("estructura", "parrafos", "fijo"), default="estructura")
    p.set_defaults(func=cmd_indexar)

    p = sub.add_parser("fragmentos", help="Muestra los fragmentos de una estrategia, sin calcular embeddings")
    p.add_argument("--estrategia", choices=("estructura", "parrafos", "fijo"), default="estructura")
    p.add_argument("--fuente", help="Filtra por archivo, p. ej. faq.json")
    p.set_defaults(func=cmd_fragmentos)

    p = sub.add_parser("buscar", help="Solo recuperación: muestra los fragmentos y su similitud (sin LLM)")
    p.add_argument("texto")
    p.add_argument("--k", type=int, default=RAG_K)
    p.set_defaults(func=cmd_buscar)

    p = sub.add_parser("preguntar", help="Responde una pregunta con la cadena RAG")
    p.add_argument("texto")
    p.add_argument("--version", choices=("v1", "v2"), default="v2")
    p.add_argument("--sin-rag", action="store_true", help="Responde sin base de conocimiento (línea base)")
    p.add_argument("--k", type=int, default=RAG_K)
    p.add_argument("--umbral", type=float, default=RAG_UMBRAL)
    p.add_argument("--mostrar-prompt", action="store_true", help="Muestra los mensajes sin llamar al modelo")
    p.set_defaults(func=cmd_preguntar)

    sub.add_parser("chat", help="Conversación con la cadena completa (Taller 1 + RAG)").set_defaults(func=cmd_chat)

    sub.add_parser("experimento-recuperacion",
                   help="Compara estrategias de chunking y calibra el umbral (sin LLM)").set_defaults(func=cmd_experimento_recuperacion)

    p = sub.add_parser("evaluar", help="Ejecuta los casos de prueba con el LLM y escribe results/")
    p.add_argument("--solo", choices=("rag", "cadena"))
    p.add_argument("--parcial", action="store_true",
                   help="No llama al modelo: escribe con las respuestas ya guardadas y marca las faltantes como pendientes")
    p.set_defaults(func=cmd_evaluar)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ErrorConfiguracion, ErrorLLM, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
