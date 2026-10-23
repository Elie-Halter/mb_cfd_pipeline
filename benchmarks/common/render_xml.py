#!/usr/bin/env python3
"""
render_xml.py -- instantiate a benchmark XML template.

Templates (`case.xml.in`) carry @KEY@ placeholders (absolute paths and run parameters) so that
nothing machine-specific is stored in the repository. Every placeholder must be given, and
every given key must be used.

Usage:
  python3 render_xml.py template.xml.in out.xml KEY=VALUE [KEY=VALUE ...]
"""
import re
import sys
import xml.etree.ElementTree as ET


def render(template, values):
    keys = set(re.findall(r"@([A-Z0-9_]+)@", template))
    missing = keys - set(values)
    unused = set(values) - keys
    if missing:
        raise SystemExit(f"render_xml: missing values for {sorted(missing)}")
    if unused:
        raise SystemExit(f"render_xml: unused keys {sorted(unused)}")
    out = template
    for k, v in values.items():
        out = out.replace(f"@{k}@", str(v))
    ET.fromstring(out)          # must stay well-formed
    return out


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    values = dict(a.split("=", 1) for a in argv[3:])
    with open(argv[1]) as f:
        text = render(f.read(), values)
    with open(argv[2], "w") as f:
        f.write(text)
    print(f"[render_xml] {argv[2]}")


if __name__ == "__main__":
    main(sys.argv)
