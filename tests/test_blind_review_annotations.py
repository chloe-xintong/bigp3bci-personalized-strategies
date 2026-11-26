from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import os
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(Path(__file__).resolve().parents[1]))
import mne
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bigp3bci_blind_review import focused_label, save_clip, save_draft, review_panel


class BlindAnnotationTests(unittest.TestCase):
    def test_explicit_no_blink_saves_empty_annotations_but_blank_alone_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);(packet/'reviewer').mkdir()
            form=packet/'reviewer/review_form.csv'
            pd.DataFrame([dict(blind_id='R1',label='',confidence='',notes='')]).to_csv(form,index=False)
            raw=mne.io.RawArray(np.zeros((1,2049)),mne.create_info(['Fz'],256,'eeg'),verbose='ERROR')
            raw._single_review_id='R1';raw._single_review_packet=str(packet.resolve());raw._single_review_focus=4
            with self.assertRaises(ValueError):save_clip(packet,'R1',raw)
            self.assertEqual(save_clip(packet,'R1',raw,no_blink=True),'not_blink')
            row=pd.read_csv(form,keep_default_na=False).iloc[0]
            self.assertEqual(row.span_review_status,'reviewed_no_blink')
            self.assertEqual(row.label,'not_blink')
            self.assertEqual(len(mne.read_annotations(packet/'reviewer/annotations/R1-annot.fif')),0)
            self.assertTrue(pd.read_csv(packet/'reviewer/annotations/R1_spans.csv').empty)
            raw.set_annotations(mne.Annotations([3.9],[.3],['BAD_']))
            with self.assertRaises(ValueError):save_clip(packet,'R1',raw,no_blink=True)

    def test_off_focus_blink_spans_save_without_inventing_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);(packet/'reviewer').mkdir()
            form=packet/'reviewer/review_form.csv'
            pd.DataFrame([dict(blind_id='R1',label='',confidence='',notes='')]).to_csv(form,index=False)
            raw=mne.io.RawArray(np.zeros((1,2049)),mne.create_info(['Fz'],256,'eeg'),verbose='ERROR')
            raw._single_review_id='R1';raw._single_review_packet=str(packet.resolve());raw._single_review_focus=4
            raw.set_annotations(mne.Annotations([1.5,3.7,4.4],[.15,.17,.16],['BAD_']*3))
            result=save_clip(packet,'R1',raw)
            self.assertEqual(result,'spans_saved_no_focus_decision')
            self.assertEqual(len(mne.read_annotations(packet/'reviewer/annotations/R1-annot.fif')),3)
            self.assertEqual(len(pd.read_csv(packet/'reviewer/annotations/R1_spans.csv')),3)
            saved=pd.read_csv(form,keep_default_na=False).iloc[0]
            self.assertEqual(saved.label,'')
            self.assertEqual(saved.span_review_status,'spans_saved_no_focus_decision')

    def test_draft_preserves_unrecognized_spans_without_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);(packet/'reviewer').mkdir()
            form=packet/'reviewer/review_form.csv'
            pd.DataFrame([dict(blind_id='R1',label='',confidence='',notes='')]).to_csv(form,index=False)
            raw=mne.io.RawArray(np.zeros((1,2049)),mne.create_info(['Fz'],256,'eeg'),verbose='ERROR')
            raw._single_review_id='R1';raw._single_review_packet=str(packet.resolve());raw._single_review_focus=4
            raw.set_annotations(mne.Annotations([2],[.5],['my_annotation']))
            draft=save_draft(packet,'R1',raw)
            saved=mne.read_annotations(draft)
            self.assertEqual(list(saved.description),['my_annotation'])
            self.assertEqual(pd.read_csv(form,keep_default_na=False).iloc[0].label,'')

    def test_rebuilding_panel_retains_unsaved_annotations_and_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);(packet/'reviewer').mkdir()
            form=packet/'reviewer/review_form.csv'
            pd.DataFrame([dict(order=1,phase='pilot_30',blind_id='R1',reviewer_code='',label='',confidence='',notes='')]).to_csv(form,index=False)
            raw=mne.io.RawArray(np.zeros((1,2049)),mne.create_info(['Fz'],256,'eeg'),verbose='ERROR')
            raw._single_review_id='R1';raw._single_review_packet=str(packet.resolve());raw._single_review_focus=4
            with patch('IPython.display.display') as display, patch('bigp3bci_blind_review.open_clip',return_value=(raw,None)):
                state=review_panel(packet,packet)
                display.call_args.args[0].children[2].children[0].click()
                raw.set_annotations(mne.Annotations([3.9],[.3],['BAD']))
                state['_controls']['notes'].value='unsaved note'
                state['_controls']['confidence'].value='high'
                with patch('bigp3bci_blind_review.plot_clip',return_value=None) as plot:
                    display.call_args.args[0].children[2].children[0].click()
                    plot.assert_called_once_with(raw,'R1')
                restored=review_panel(packet,packet,existing_state=state)
                self.assertIs(restored['raw'],raw)
                self.assertEqual(restored['_controls']['notes'].value,'unsaved note')
                self.assertEqual(len(raw.annotations),1)
                display.call_args.args[0].children[2].children[1].click()
                saved=pd.read_csv(form,keep_default_na=False).iloc[0]
                self.assertEqual(saved.label,'true_blink')
                self.assertEqual(saved.notes,'unsaved note')

    def test_panel_open_button_passes_valid_initial_and_phase_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);(packet/'reviewer').mkdir()
            pd.DataFrame([
                dict(order=1,phase='pilot_30',blind_id='R1',reviewer_code='',label='',confidence='',notes=''),
                dict(order=2,phase='remaining_90',blind_id='R2',reviewer_code='',label='',confidence='',notes='')
            ]).to_csv(packet/'reviewer/review_form.csv',index=False)
            raw=mne.io.RawArray(np.zeros((1,2049)),mne.create_info(['Fz'],256,'eeg'),verbose='ERROR')
            with patch('IPython.display.display') as display, patch('bigp3bci_blind_review.open_clip',return_value=(raw,None)) as opened:
                state=review_panel(packet,packet)
                panel=display.call_args.args[0]
                phase,item,buttons=panel.children[:3]
                self.assertEqual(item.value,'R1')
                buttons.children[0].click()
                opened.assert_called_with(packet,packet,'R1')
                self.assertEqual(state['id'],'R1')
                phase.value='remaining_90'
                self.assertEqual(item.value,'R2')
                buttons.children[0].click()
                opened.assert_called_with(packet,packet,'R2')

    def test_bad_means_blink_only_in_new_workflow(self):
        self.assertEqual(focused_label(mne.Annotations([3.9],[.3],['BAD']),4), 'true_blink')
        self.assertEqual(focused_label(mne.Annotations([3.9],[.3],['RECOVERY']),4), 'reverse_recovery')

    def test_no_focus_or_conflicting_labels_fail(self):
        for annotations in [mne.Annotations([1],[.2],['BAD']),
                            mne.Annotations([3.8,3.9],[.4,.4],['BAD','OTHER']),
                            mne.Annotations([3.9],[.4],['BAD_channel'])]:
            with self.assertRaises(ValueError):focused_label(annotations,4)

    def test_save_reload_keeps_other_answers_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);(packet/'reviewer').mkdir()
            form=packet/'reviewer/review_form.csv'
            pd.DataFrame([dict(blind_id='R1',label='',confidence='',notes=''),
                          dict(blind_id='R2',label='uncertain',confidence='low',notes='keep')]).to_csv(form,index=False)
            raw=mne.io.RawArray(np.zeros((1,2049)),mne.create_info(['Fz'],256,'eeg'),verbose='ERROR')
            raw._single_review_id='R1';raw._single_review_packet=str(packet.resolve());raw._single_review_focus=4
            raw.set_annotations(mne.Annotations([3.9],[.3],['BAD']))
            self.assertEqual(save_clip(packet,'R1',raw,'high','test'),'true_blink')
            raw.set_annotations(mne.Annotations([3.9],[.3],['UNCERTAIN']))
            save_clip(packet,'R1',raw,'low','revised')
            table=pd.read_csv(form,keep_default_na=False).set_index('blind_id')
            self.assertEqual(table.loc['R1','label'],'uncertain')
            self.assertEqual(table.loc['R2','notes'],'keep')
            self.assertEqual(len(list((packet/'reviewer/history').iterdir())),2)
            saved=mne.read_annotations(packet/'reviewer/annotations/R1-annot.fif')
            self.assertEqual(focused_label(saved,4),'uncertain')
            with self.assertRaises(ValueError):save_clip(packet,'R2',raw)


if __name__=='__main__':unittest.main()
