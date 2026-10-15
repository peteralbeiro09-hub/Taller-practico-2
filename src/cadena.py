"""Cadena del Taller 1 ampliada con RAG: clasificar → enrutar → responder (con datos exactos o con RAG) o escalar.

Cambios respecto al Taller 1:
- Nueva intención "informacion" (y también "otro"): se responde con la cadena RAG (src/rag.py). Antes se
  escalaba todo a un humano; ahora el asistente responde si la base de conocimiento lo permite y, si no,
  dice con honestidad que no tiene la información.
- Devoluciones: el catálogo ya no se pega completo en el prompt v4; se recuperan los productos más parecidos al
  mensaje. La política se recupera completa por metadatos (ver contexto_devolucion).
- Pedidos: se pasan al prompt solo los pedidos cuyo número menciona el cliente (búsqueda exacta, sin RAG).
"""

from src import contexto, prompts, rag
from src.config import cargar_toml
from src.llm import completar

INTENCIONES = {"pedido", "devolucion", "informacion", "queja", "otro"}

# Cuántos productos se recuperan para armar el contexto del prompt v4 de devoluciones.
K_CATALOGO = 3


def clasificar(mensaje: str) -> dict | None:
    """Pide al modelo la clasificación en JSON. Devuelve None si la respuesta no es interpretable."""
    config = cargar_toml("cadena")["clasificador"]
    mensajes = [
        {"role": "system", "content": config["system"]},
        {"role": "user", "content": config["user"].replace("{consulta}", mensaje)},
    ]
    clasificacion = prompts.extraer_json(completar(mensajes))
    if not clasificacion or clasificacion.get("intencion") not in INTENCIONES:
        return None
    return clasificacion


def _escalar(mensaje: str, clasificacion: dict | None, motivo: str) -> dict:
    datos = clasificacion or {}
    resumen = (
        f"Motivo del escalamiento: {motivo}\n"
        f"Intención: {datos.get('intencion', 'desconocida')} | "
        f"Sentimiento: {datos.get('sentimiento', 'desconocido')} | "
        f"Pedido: {datos.get('numero_seguimiento') or 'no indicado'} | "
        f"Producto: {datos.get('producto') or 'no indicado'}\n"
        f"Mensaje original: {mensaje}"
    )
    return {
        "ruta": f"humano ({motivo})",
        "clasificacion": clasificacion,
        "respuesta": cargar_toml("cadena")["escalamiento"]["cliente"],
        "resumen_agente": resumen,
        "fuentes": [],
    }


def contexto_devolucion(mensaje: str) -> tuple[dict, list[str]]:
    """Arma {politica, catalogo, pedidos} para el prompt v4 de devoluciones (recuperación híbrida).

    - Catálogo: búsqueda por similitud entre los productos (filtro tipo=producto, umbral 0 porque el
      clasificador ya decidió que es una devolución). Es la parte que crece con el negocio y la que de verdad
      hay que buscar; el producto correcto sale primero en todos los casos de prueba.
    - Política: se recuperan sus 7 secciones completas por metadatos, no por similitud. Medido en
      `python app.py experimento-recuperacion` §4: buscando las 4 secciones más parecidas solo llegaban las
      reglas necesarias en 2 de 8 casos, porque "quiero devolver el cepillo de dientes" no se parece a "Higiene
      personal: no se aceptan devoluciones" (hay que pasar por la categoría del producto). La política es corta
      (~600 tokens) y sus reglas dependen unas de otras (plazo, excepción por daño, no devolvibles).
    """
    politica = rag.fragmentos_de({"fuente": "politica_devoluciones.pdf"})
    catalogo = rag.recuperar(mensaje, k=K_CATALOGO, umbral=0, filtro={"tipo": "producto"})
    valores = {
        "politica": rag.formatear_contexto(politica),
        "catalogo": rag.formatear_contexto([d for d, _ in catalogo]),
        "pedidos": contexto.pedidos_texto(contexto.numeros_pedido(mensaje)),
    }
    fuentes = [f"politica_devoluciones.pdf completa ({len(politica)} secciones, por metadatos)"]
    fuentes += [f"{d.metadata['fuente']} › {d.metadata['seccion']} (sim. {s:.3f})" for d, s in catalogo]
    return valores, fuentes


def responder_devolucion_v4(mensaje: str) -> dict:
    """Ejecuta devoluciones v4 con contexto recuperado. El razonamiento se guarda para auditoría, nunca se muestra."""
    valores, fuentes = contexto_devolucion(mensaje)
    crudo = completar(prompts.construir_mensajes("devoluciones", "v4", mensaje, valores))
    datos = prompts.extraer_json(crudo)
    if not datos or not datos.get("respuesta_cliente"):
        return {"ok": False, "crudo": crudo, "fuentes": fuentes}
    return {
        "ok": True,
        "decision": datos.get("decision"),
        "razonamiento": datos.get("razonamiento"),
        "respuesta_cliente": datos["respuesta_cliente"],
        "fuentes": fuentes,
    }


def responder_pedido_v4(mensaje: str) -> dict:
    """Pedidos v4 del Taller 1 con búsqueda exacta: el modelo solo ve los pedidos que el cliente menciona."""
    valores = {"pedidos": contexto.pedidos_texto(contexto.numeros_pedido(mensaje))}
    respuesta = completar(prompts.construir_mensajes("pedidos", "v4", mensaje, valores))
    return {"respuesta": respuesta, "fuentes": ["pedidos.json (búsqueda exacta por número)"]}


def responder(mensaje: str) -> dict:
    """Clasifica, enruta y responde. Ante cualquier duda, escala a un agente humano."""
    clasificacion = clasificar(mensaje)
    if clasificacion is None:
        return _escalar(mensaje, None, "clasificación no interpretable")

    intencion = clasificacion["intencion"]
    if clasificacion.get("sentimiento") == "negativo":
        return _escalar(mensaje, clasificacion, "sentimiento negativo")
    if intencion == "queja":
        return _escalar(mensaje, clasificacion, "intención queja")

    if intencion == "pedido":
        r = responder_pedido_v4(mensaje)
        return {"ruta": "pedidos v4", "clasificacion": clasificacion, **r}

    if intencion == "devolucion":
        r = responder_devolucion_v4(mensaje)
        if not r["ok"]:
            return _escalar(mensaje, clasificacion, "JSON de devoluciones no interpretable")
        return {"ruta": "devoluciones v4 + RAG (productos) + política completa", "clasificacion": clasificacion,
                "respuesta": r["respuesta_cliente"], "fuentes": r["fuentes"]}

    # informacion u otro: la base de conocimiento decide si hay respuesta.
    r = rag.responder(mensaje)
    return {"clasificacion": clasificacion, **r}
