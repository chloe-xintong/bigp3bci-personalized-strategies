from pathlib import Path
import re
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_data_policy import require_train_edf,require_train_record

DATA=Path('data/raw/bigP3BCI-data/StudyF/F_03/SE001')
# Lines that mention a Test-split name without reading Test data.
ALLOWED={('scripts/audit_studyf_abc_strategies.py',"eeg_stats_test = sum('Test' in open(")}
LITERAL=re.compile(r"[\"'/](Test|Static|Dyn|DynBigram)[\"'/]")


class DataPolicyTests(unittest.TestCase):
    def test_train_edf_is_accepted(self):
        path=DATA/'Train'/'CBCol'/'F_03_SE001_CBCol_Train01.edf'
        self.assertEqual(require_train_edf(path),path)
        self.assertEqual(require_train_record({'source_edf':str(path)}),path)

    def test_test_split_and_conditions_are_rejected(self):
        for condition in ['Static','Dyn','DynBigram']:
            with self.subTest(condition=condition),self.assertRaises(ValueError):
                require_train_edf(DATA/'Test'/condition/'F_03_SE001_Test.edf')
        with self.assertRaises(ValueError):
            require_train_record({'source_edf':str(DATA/'Test'/'Static'/'x.edf')})

    def test_unexpected_layout_is_rejected(self):
        with self.assertRaises(ValueError):require_train_edf(DATA/'CBCol'/'x.edf')
        with self.assertRaises(ValueError):require_train_edf('x.edf')

    def test_code_does_not_name_test_split_folders(self):
        hits=[]
        for path in sorted([*ROOT.glob('scripts/*.py'),*ROOT.glob('src/*.py')]):
            relative=path.relative_to(ROOT).as_posix()
            if relative=='src/bigp3bci_data_policy.py':continue
            for number,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
                if LITERAL.search(line) and not any(relative==f and marker in line for f,marker in ALLOWED):
                    hits.append(f'{relative}:{number}: {line.strip()[:100]}')
        self.assertEqual(hits,[],'Read data through bigp3bci_data_policy; Test-split paths found:\n'+'\n'.join(hits))


if __name__=='__main__':unittest.main()
