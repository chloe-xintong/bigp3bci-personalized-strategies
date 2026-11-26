"""Consolidate report families and let existing builders update one section."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    '05_preprocessing.ipynb': ('预处理与候选检测', [
        '07_bigp3bci_expanded_candidate_pilot.ipynb',
        '05_bigp3bci_causal_preprocessing_pilot.ipynb']),
    '06_blink_review.ipynb': ('眨眼检测与人工复核', [
        '08_bigp3bci_current_blink_detection_summary.ipynb',
        '11_bigp3bci_blink_candidate_review.ipynb',
        '09_bigp3bci_reviewed_blink_false_positives.ipynb',
        '06_bigp3bci_manual_blink_threshold_validation.ipynb']),
    '10_swlda_comparison.ipynb': ('SWLDA 拒绝策略比较', [
        '16_bigp3bci_swlda_all_studyf.ipynb',
        '15_bigp3bci_swlda_full_three_sessions.ipynb',
        '14_bigp3bci_swlda_rejection_experiment.ipynb']),
}
TARGETS = {source: target for target, (_, sources) in GROUPS.items() for source in sources}


def read_sections(target):
    path = ROOT / 'notebooks' / target
    if not path.exists():
        return {}, {}
    report = nb.read(path, as_version=4)
    sections = {}
    for cell in report.cells:
        source = cell.metadata.get('report_source')
        if source:
            sections.setdefault(source, []).append(deepcopy(cell))
    return sections, report.metadata.get('source_metadata', {})


def write_report(notebook, path):
    """Drop-in nb.write replacement for a report family's existing builder."""
    path = Path(path)
    target = TARGETS.get(path.name)
    if target is None:
        return nb.write(notebook, path)
    sections, metadata = read_sections(target)
    sections[path.name] = deepcopy(notebook.cells)
    metadata[path.name] = deepcopy(notebook.metadata)
    return assemble(target, sections, metadata)


def assemble(target, sections, metadata):
    title, sources = GROUPS[target]
    cells = [nb.v4.new_markdown_cell(
        f'# {title}\n\n按功能合并的报告。首先展示最新覆盖范围，其后保留早期实验与复核记录。'
        '\n\n各章节的数据范围、阈值版本和历史输出独立，不能合并为同一个实验。'
        '重新运行时请重启 kernel，从所需章节的初始化单元开始；人工交互章节按原说明操作。'
        '\n\n更新：运行原报告 builder 会直接更新对应章节，保留其他章节的结果。')]
    if target == '06_blink_review.ipynb':
        from blink_single_review_cells import single_review_cells
        from blink_duration_report_cells import duration_report_cells
        cells.extend(single_review_cells())
        cells.extend(duration_report_cells())
    for i, source in enumerate(sources):
        if source not in sections:
            continue
        cells.append(nb.v4.new_markdown_cell(
            f'---\n\n## {"主要结果" if i == 0 else "历史实验 / 专项复核"}：{source[3:-6]}'))
        for j, original in enumerate(sections[source]):
            cell = deepcopy(original)
            cell.metadata['report_source'] = source
            cell.id = sha256(f'{source}:{j}'.encode()).hexdigest()[:16]
            if cell.cell_type == 'markdown':
                cell.source = re.sub(r'^# \d{2}\s*[\u00b7\u8def]\s*', '# ', cell.source)
                for old, new in TARGETS.items():
                    cell.source = cell.source.replace(old, new)
            cells.append(cell)
    report = nb.v4.new_notebook(cells=cells, metadata=deepcopy(metadata[sources[0]]))
    report.metadata['source_metadata'] = metadata
    nb.validate(report)
    destination = ROOT / 'notebooks' / target
    nb.write(report, destination)
    return destination


def main():
    for target, (_, sources) in GROUPS.items():
        sections, metadata = read_sections(target)
        originals = []
        for source in sources:
            path = ROOT / 'notebooks' / source
            if path.exists():
                notebook = nb.read(path, as_version=4)
                sections[source] = notebook.cells
                metadata[source] = notebook.metadata
                originals.append(path)
        if set(sections) != set(sources):
            raise ValueError(f'Missing source sections for {target}')
        # The old 05 report repeated an overview of 07; 07 is now in this file.
        sections = {source: [c for c in cells if not c.metadata.get('expanded_pilot_appendix')]
                    for source, cells in sections.items()}
        destination = assemble(target, sections, metadata)
        # Validate the serialized report before removing the superseded entry points.
        saved = nb.read(destination, as_version=4)
        nb.validate(saved)
        for source in sources:
            actual = [c for c in saved.cells if c.metadata.get('report_source') == source]
            assert len(actual) == len(sections[source])
            for before, after in zip(sections[source], actual):
                if before.cell_type == 'code':
                    assert before.source == after.source
                    assert before.outputs == after.outputs
        for path in originals:
            path.unlink()
        print(destination.name)


if __name__ == '__main__':
    main()
