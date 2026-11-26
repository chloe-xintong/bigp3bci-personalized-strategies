"""Test-set policy: every analysis reads Train EDFs only.

bigP3BCI stores recordings as <study>/<subject>/<session>/<split>/<condition>/<file>.edf.
The Test split holds the Static, Dyn and DynBigram conditions; none of them are
read by this project. Scripts call ``require_train_edf`` before opening an EDF or
trusting a cached processing record.
"""
from pathlib import Path

TRAIN_SPLIT='Train'
FORBIDDEN_PARTS=frozenset({'Test','Static','Dyn','DynBigram'})


def require_train_edf(path):
    """Return ``path`` as a Path if it is a Train EDF; raise ValueError otherwise."""
    path=Path(path)
    forbidden=FORBIDDEN_PARTS.intersection(path.parts)
    if forbidden:
        raise ValueError(f'Test-set policy: {sorted(forbidden)} data are not read ({path.name})')
    if len(path.parts)<3 or path.parts[-3]!=TRAIN_SPLIT:
        raise ValueError(f'Test-set policy: expected <session>/Train/<condition>/<file> ({path})')
    return path


def require_train_record(record):
    """Check the ``source_edf`` of a cached preprocessing record."""
    return require_train_edf(record['source_edf'])
