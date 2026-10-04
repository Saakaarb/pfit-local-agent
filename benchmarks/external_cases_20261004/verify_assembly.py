"""Check assembled files only. No pfit stages, LLM calls, integration or fitting."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALIASES = {'Insulin_signal': 'observable_Insulin', 'pSTAT5': 'observable_pSTAT5'}


def verify():
    hashes = json.loads((ROOT / 'sha256.json').read_text())
    for relative, expected in hashes.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    totals = {}
    for entry in json.loads((ROOT / 'data_provenance.json').read_text()):
        path = ROOT / entry['file']
        with path.open() as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == entry['columns']
            rows = list(reader)
        with (ROOT / entry['source_file']).open() as handle:
            source = {line: row for line, row in enumerate(csv.DictReader(handle, delimiter='\t'), 2)}
        assert len(rows) == entry['rows'] == len(entry['source_lines_by_output_row'])
        previous = -math.inf
        for row, lines in zip(rows, entry['source_lines_by_output_row']):
            assert all(math.isfinite(float(value)) for value in row.values())
            assert float(row['time']) > previous
            previous = float(row['time'])
            assert len(lines) == len(entry['columns']) - 1
            for column, line in zip(entry['columns'][1:], lines):
                original = source[line]
                assert row['time'] == original['time']
                assert row[column] == original['measurement']
                assert original['observableId'] == ALIASES.get(column, column)
                if column == 'Insulin_signal':
                    assert float(row[column]) > 0
                    condition = 'model1_data12' if 'insulin_10_' in path.name else 'model1_data13'
                    assert original['simulationConditionId'] == condition
                elif column == 'pSTAT5':
                    condition = 'model1_data3' if path.name == 'il13_4.csv' else 'model1_data2'
                    assert original['simulationConditionId'] == condition
                else:
                    assert original['simulationConditionId'] == 'typeIDT1_ExpID1'
        case = path.parts[-3]
        counts = totals.setdefault(case, {'records': 0, 'measurements': 0})
        counts['records'] += 1
        counts['measurements'] += len(rows) * (len(entry['columns']) - 1)
    inventory = json.loads((ROOT / 'inventory.json').read_text())
    for case in inventory['cases']:
        counts = totals[case['id']]
        assert counts['records'] == case['experiments']
        assert counts['measurements'] == case['measurements']
        inputs = ROOT / case['input_directory']
        assert set(p.name for p in inputs.iterdir()) == {'user_info.txt', *case['csv_files']}
        prompt = (inputs / 'user_info.txt').read_text()
        assert all(filename in prompt for filename in case['csv_files'])
        assert case['status'] == 'assembled_not_run'
        assert not case['included_in_model_comparison'] and not case['pfit_steps_run']
    print(json.dumps({'status': 'pass', 'checks': 'hashes, raw-data correspondence, finite values, time order, conditions, counts, input-only folders', 'cases': totals, 'pfit_steps_run': []}, indent=2))


if __name__ == '__main__':
    verify()
