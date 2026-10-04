"""Recreate only CSV inputs from the pinned source tables; never run pfit."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def measurements(model):
    with (ROOT / 'sources' / model / 'measurements.tsv').open() as handle:
        return [(line, row) for line, row in enumerate(csv.DictReader(handle, delimiter='\t'), 2)]


def prepare():
    provenance = []

    def write(case, filename, header, rows, line_numbers, model):
        path = ROOT / 'cases' / case / 'inputs' / filename
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        provenance.append(dict(file=str(path.relative_to(ROOT)), source_model=model,
            source_file=f'sources/{model}/measurements.tsv',
            source_lines_by_output_row=line_numbers, rows=len(rows), columns=header))

    model = 'Beer_MolBioSystems2014'
    selected = [(i, r) for i, r in measurements(model) if r['simulationConditionId'] == 'typeIDT1_ExpID1']
    grouped = {}
    for line, row in selected:
        group = grouped.setdefault(row['time'], {})
        assert row['observableId'] not in group
        group[row['observableId']] = (line, row['measurement'])
    rows, lines = [], []
    for time in sorted(grouped, key=float):
        g = grouped[time]
        rows.append([time, g['Bacnorm'][1], g['IndconcNormRange'][1]])
        lines.append([g['Bacnorm'][0], g['IndconcNormRange'][0]])
    write('beer_indigoidine', 'culture.csv', ['time', 'Bacnorm', 'IndconcNormRange'], rows, lines, model)

    model = 'Raia_CancerResearch2011'
    for condition, dose in [('model1_data3', 4), ('model1_data2', 20)]:
        selected = sorted([(i, r) for i, r in measurements(model)
            if r['simulationConditionId'] == condition and r['observableId'] == 'observable_pSTAT5'],
            key=lambda item: float(item[1]['time']))
        write('raia_il13', f'il13_{dose}.csv', ['time', 'pSTAT5'],
            [[r['time'], r['measurement']] for _, r in selected], [[i] for i, _ in selected], model)

    model = 'Schwen_PONE2014'
    for condition, dose in [('model1_data12', 10), ('model1_data13', 100)]:
        groups = {}
        for line, row in measurements(model):
            if row['simulationConditionId'] == condition and row['observableId'] == 'observable_Insulin':
                groups.setdefault(row['time'], []).append((line, row))
        assert all(len(rows) == 2 for rows in groups.values())
        for repeat in range(2):
            selected = [groups[t][repeat] for t in sorted(groups, key=float)]
            write('schwen_insulin', f'insulin_{dose}_r{repeat+1}.csv', ['time', 'Insulin_signal'],
                [[r['time'], r['measurement']] for _, r in selected], [[i] for i, _ in selected], model)
    (ROOT / 'data_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    prepare()
