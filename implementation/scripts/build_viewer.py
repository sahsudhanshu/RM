#!/usr/bin/env python
"""Inline viewer/missions.json into viewer/template.html -> viewer/index.html (single self-contained page)."""
import os
root = os.path.join(os.path.dirname(__file__), "..", "viewer")
data = open(os.path.join(root, "missions.json")).read()
html = open(os.path.join(root, "template.html")).read().replace("__DATA__", data)
open(os.path.join(root, "index.html"), "w").write(html)
print("wrote viewer/index.html", len(html) // 1024, "KB")

# offline copy: inline three.js so the page also works without internet (fonts fall back to system fonts)
import urllib.request
cache = os.path.join(root, "three.r128.min.js")
if not os.path.exists(cache):
    urllib.request.urlretrieve("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js", cache)
tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
offline = html.replace(tag, "<script>" + open(cache).read() + "</script>")
open(os.path.join(root, "index_offline.html"), "w").write(offline)
print("wrote viewer/index_offline.html", len(offline) // 1024, "KB")
