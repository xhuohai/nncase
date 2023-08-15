import argparse
import json
import pandas as pd

def json2md(json_file):
    l2 = []
    with open(json_file, 'r') as f:
      l2 = json.load(f)

    df = pd.DataFrame.from_records(l2)
    md = df.to_markdown()
    md_file = json_file.split('/')[-1].split('.')[0] + '.md'

    with open(md_file, 'w') as f:
      f.write(md)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="json2md")
    parser.add_argument("--json", help='json file', type=str)
    args = parser.parse_args()
    json2md(args.json)
