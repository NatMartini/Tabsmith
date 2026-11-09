"""Input records with the types a bundle expects: the first rows of a CSV as JSON lines.
Usage: records.py <csv> <input.schema.json> <out.jsonl> <limit>"""
import csv
import json
import sys

csv_path, schema_path, out_path, limit = sys.argv[1:5]
limit = int(limit)
fields = json.load(open(schema_path, encoding="utf-8"))["fields"]
tokens = {"", "NA", "null"}
with open(csv_path, newline="", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= limit:
            break
        rec = {}
        for fd in fields:
            v = row.get(fd["name"])
            if v is None or v in tokens:
                rec[fd["name"]] = None
            elif fd["kind"] == "numeric":
                x = float(v)
                rec[fd["name"]] = None if x != x else x
            elif fd.get("raw_dtype") == "bool":
                rec[fd["name"]] = v.lower() == "true"
            elif fd.get("raw_dtype") == "int":
                rec[fd["name"]] = int(v)
            else:
                rec[fd["name"]] = v
        out.write(json.dumps(rec) + "\n")
print("wrote", out_path)
