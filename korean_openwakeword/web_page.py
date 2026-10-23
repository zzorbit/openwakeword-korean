"""Fill an explicit, non-executable page configuration slot."""
import json


def render_page_config(template, config=None):
    marker = "<!--KOWW_PAGE_CONFIG-->"
    if template.count(marker) != 1:
        raise ValueError("Page requires one configuration slot")
    # ASCII JSON also escapes U+2028/U+2029. Escape HTML delimiters so no
    # supplied text can close the data script or introduce markup.
    data = json.dumps(config or {}, ensure_ascii=True, allow_nan=False)
    for character, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        data = data.replace(character, escaped)
    return template.replace(marker, '<script id="koww-page-config" type="application/json">' + data + "</script>")
