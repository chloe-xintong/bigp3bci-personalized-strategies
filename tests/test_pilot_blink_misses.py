import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_pilot_blink_misses import union_events


class PilotMissTests(unittest.TestCase):
    def test_duplicate_clip_annotations_merge_with_all_sources(self):
        rows=[dict(edf_name='a',subject_id='s',blind_id='r1',start_seconds=1,end_seconds=2),
              dict(edf_name='a',subject_id='s',blind_id='r2',start_seconds=1.2,end_seconds=2.2)]
        events=union_events(rows)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['source_blind_ids'],['r1','r2'])
        self.assertEqual(events[0]['end_seconds'],2.2)

    def test_distinct_edfs_and_touching_events_are_not_duplicates(self):
        rows=[dict(edf_name='a',subject_id='s',blind_id='r1',start_seconds=1,end_seconds=2),
              dict(edf_name='a',subject_id='s',blind_id='r1',start_seconds=2,end_seconds=3),
              dict(edf_name='b',subject_id='s',blind_id='r2',start_seconds=1,end_seconds=2)]
        self.assertEqual(len(union_events(rows)),3)


if __name__=='__main__':unittest.main()
