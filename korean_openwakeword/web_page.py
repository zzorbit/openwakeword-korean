"""Fill an explicit, non-executable page configuration slot."""
import json
from importlib.resources import files


def render_page_config(template, config=None):
    marker = "<!--KOWW_PAGE_CONFIG-->"
    if template.count(marker) != 1:
        raise ValueError("Page requires one configuration slot")
    # Standalone review serves a single HTML page. Embed the same small style
    # asset used by Studio so it needs no extra endpoint or network request.
    stylesheet = '<link rel="stylesheet" href="/audio_listening.css">'
    if stylesheet in template:
        css = (files("korean_openwakeword") / "resources" / "audio_listening.css").read_text(encoding="utf-8")
        template = template.replace(stylesheet, '<style data-koww-listening>' + css + '</style>')
    player_script = '<script src="/listening_player.js"></script>'
    if player_script in template:
        javascript = (files("korean_openwakeword") / "resources" / "listening_player.js").read_text(encoding="utf-8")
        template = template.replace(player_script, '<script>' + javascript + '</script>')
    # ASCII JSON also escapes U+2028/U+2029. Escape HTML delimiters so no
    # supplied text can close the data script or introduce markup.
    data = json.dumps(config or {}, ensure_ascii=True, allow_nan=False)
    for character, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        data = data.replace(character, escaped)
    return template.replace(marker, '<script id="koww-page-config" type="application/json">' + data + "</script>")
