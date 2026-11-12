"""Arquitectura: renderiza los diagramas Mermaid de docs/architecture.md."""
import html
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dash_common as C  # noqa: E402

MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"

P = C.paths()
st.title("Arquitectura")

doc = P["docs"] / "architecture.md"
text = C.read_text(str(doc))
if text is None:
    C.missing("docs/architecture.md", note="Crea ese archivo con bloques ```mermaid para verlos aquí.")
else:
    blocks = C.mermaid_blocks(text)
    if not blocks:
        st.info("docs/architecture.md existe pero no contiene bloques ```mermaid.")
    for i, code in enumerate(blocks, 1):
        st.subheader(f"Diagrama {i}")
        page = f"""<!doctype html><html><body style="margin:0;background:#fff">
<pre class="mermaid">{html.escape(code)}</pre>
<script type="module">
import mermaid from '{MERMAID_JS}';
mermaid.initialize({{startOnLoad: true, securityLevel: 'strict'}});
</script></body></html>"""
        components.html(page, height=min(max(code.count("\n") * 34 + 120, 260), 900), scrolling=True)
        with st.expander(f"Código Mermaid {i}"):
            st.code(code, language="text")
    with st.expander("Documento completo"):
        st.markdown(text)
    st.caption("Los diagramas se dibujan en el navegador con mermaid 11.4.1 desde cdn.jsdelivr.net (requiere internet).")
