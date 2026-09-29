"""CPU migration contract tests using deterministic evidence, not basketball labels.

No model training, GPU inference, Fabric calls, buffering changes, or quality tuning.
"""
from __future__ import annotations

from dataclasses import fields, replace
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nba_yolo_shot_tagger.config import RuntimeConfig, config_from_params
from nba_yolo_shot_tagger.live import FileSink, NoopSink, encode_x
from nba_yolo_shot_tagger.model import EXPECTED_FAMILIES, YoloStudentModel
from nba_yolo_shot_tagger.reviewed_core import Controller, OneEuro
from nba_yolo_shot_tagger.reviewed_runtime import ReviewedRound01Runtime, reconstruct_source_frames
from nba_yolo_shot_tagger.service import ShotFocusService
from nba_yolo_shot_tagger.trajectory import expand_to_source_frames, legal_crop_geometry
from nba_yolo_shot_tagger.types import Candidate, FrameEvidence, ModelIdentity, VideoInfo

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = str(ROOT / 'models/nba_yolo_student/runtime_manifest.json')
FAMILIES = tuple(EXPECTED_FAMILIES)


class Sink:
    def __init__(self): self.batches = []
    def publish(self, values): self.batches.append(list(values))
    @property
    def values(self): return [v for batch in self.batches for v in batch]


class Detector:
    def __init__(self, cuts=None, tail=()): self.cuts = cuts or {}; self.tail = list(tail)
    def push(self, path): return list(self.cuts.get(path, ()))
    def flush(self): answer, self.tail = self.tail, []; return answer


class Model:
    identity = ModelIdentity('fixture', 'a' * 64, 'fixture', FAMILIES)


class Video:
    def __init__(self, path='a.mp4', n=120, fps=60., width=1920, height=1080):
        self.path = path
        self.info = VideoInfo(fps, n, width, height, int(round(1000 * n / fps)))


def candidate(family='gameplay_follow', x=.72, confidence=.8):
    return Candidate(family, confidence, (x-.025, .2, x+.025, .8))


def evidence(n=120, fps=60., family='gameplay_follow', x=.72, offset=0, step=6):
    return [FrameEvidence(i, round(i/fps*1000), (candidate(family, x),))
            for i in range(offset, n, step)]


def service(mode='shot_file', detector=None, **params):
    s = ShotFocusService.__new__(ShotFocusService)
    s.config = RuntimeConfig(input_mode=mode, verify_model_sha256=False,
                             runtime_manifest_path=MANIFEST, **params)
    s.model = Model(); s.x_sink = Sink()
    s.shot_detector = detector if mode == 'segment_file' else None
    s._apply_config()
    s._shot_index = 0; s._abs_frames = 0; s._abs_ms = 0
    s._last_source_media = None; s._last_video_info = None
    s._last_abs_start = 0; s._last_abs_start_ms = 0
    s._reset_cycle(0, 0)
    return s


def payload(frame):
    return [{'class_id': FAMILIES.index(c.family), 'family': c.family,
             'confidence': c.confidence, 'bbox_xyxy_norm': list(c.bbox)} for c in frame.candidates]


def oracle(ev, info):
    r = ReviewedRound01Runtime(MANIFEST, info.width, info.height)
    times, xs, events = [], [], []
    fps = Fraction(info.fps).limit_denominator(1_000_000)
    for k, f in enumerate(ev):
        t = f.frame_index * fps.denominator / fps.numerator
        x, _, event = r.update(payload(f), t, shot_start=k == 0)
        times.append(t); xs.append(x); events.append(event)
    fps = Fraction(info.fps).limit_denominator(1_000_000)
    full = reconstruct_source_frames(times, xs, events, 0, info.frame_count,
                                     fps.numerator, fps.denominator)
    return xs, events, np.asarray(full)


class Configuration(unittest.TestCase):
    def test_manifest_default_is_local_not_container_absolute(self):
        c = config_from_params({})
        self.assertEqual(c.runtime_manifest_path, 'models/nba_yolo_student/runtime_manifest.json')
    def test_manifest_override_and_validation(self):
        self.assertEqual(config_from_params({'runtime_manifest_path':MANIFEST}).runtime_manifest_path, MANIFEST)
        for v in ('', '  ', False, 0, []):
            with self.subTest(value=v), self.assertRaises(ValueError):
                config_from_params({'runtime_manifest_path':v})
    def test_botics_io_defaults_and_modes_unchanged(self):
        c = config_from_params({})
        self.assertEqual((c.input_mode, c.family_determination_max_seconds,
                          c.min_shot_seconds, c.trajectory_commit_lag_frames), ('segment_file',5.,.75,300))
        self.assertEqual((c.live_data_stream, c.coordinate_decimals, c.output_track), ('',6,'vertical_video'))
        self.assertEqual(config_from_params({'input_mode':'shot_file'}).family_determination_max_seconds,999999.)
    def test_reviewed_evidence_defaults_preserved(self):
        c = config_from_params({})
        self.assertEqual((c.imgsz,c.inference_fps,c.batch_size,c.min_detection_confidence,
                          c.iou,c.max_detections,c.top_k,c.use_fp16), (1280,10.,8,.001,.7,100,20,False))
    def test_legacy_hold_options_still_parse(self):
        c = config_from_params({'hold_and_cut_families':'person_subject', 'min_hold_seconds':2.})
        self.assertEqual(c.hold_and_cut_families,'person_subject')
    def test_manifest_filter_is_exact(self):
        r = ReviewedRound01Runtime(MANIFEST,1920,1080)
        self.assertEqual((r.euro.min_cutoff,r.euro.beta),(0.4,1.0))
    def test_missing_runtime_manifest_fails_closed(self):
        s=service();s.config=replace(s.config,runtime_manifest_path='/missing/reviewed_manifest.json')
        with self.assertRaises(FileNotFoundError):s._consume_file(Video(),evidence())
    def test_bad_filter_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.json'; d=json.loads(Path(MANIFEST).read_text());d['filter_config']['kind']='baseline';p.write_text(json.dumps(d))
            s=service();s.config=replace(s.config,runtime_manifest_path=str(p))
            with self.assertRaises(ValueError):s._consume_file(Video(),evidence())


class RuntimeParity(unittest.TestCase):
    def check_family(self, family):
        v=Video(n=240, fps=60000/1001)
        ev=[FrameEvidence(i,round(i/v.info.fps*1000),
                          (candidate(family,.35 if i<120 else .74),)) for i in range(0,240,6)]
        s=service();shot=s._consume_file(v,ev)[0]
        xs,events,expected=oracle(ev,v.info)
        np.testing.assert_allclose(s.x_sink.values,xs,rtol=0,atol=1e-12)
        np.testing.assert_allclose(shot.x_coordinates,expected,rtol=0,atol=1e-12)
    def test_gameplay(self):self.check_family('gameplay_follow')
    def test_speaker(self):self.check_family('active_speaker')
    def test_person(self):self.check_family('person_subject')
    def test_graphics(self):self.check_family('graphic_text_lock')
    def test_static(self):self.check_family('static_composition')
    def test_split(self):self.check_family('split_screen')
    def test_safe_center(self):self.check_family('safe_center')
    def test_reference_formula_is_controller_then_one_euro_not_anchor(self):
        cfg=json.loads(Path(MANIFEST).read_text());c=Controller(cfg['controller_params'],.158203125);f=OneEuro(.4,1.)
        ev=evidence(n=120,x=.72);s=service();shot=s._consume_file(Video(),ev)[0]
        expected=[]
        for e in ev:
            t=e.frame_index/60.;base,event=c.step(t,.72,.8,'gameplay_follow')
            if event==2:f.reset()
            expected.append(f.step(t,float(np.float32(base))))
        np.testing.assert_allclose(s.x_sink.values,expected,atol=1e-12,rtol=0)
    def test_metadata_family_does_not_gate_steering(self):
        ev=evidence(n=240)
        ev=[replace(f,candidates=(candidate('active_speaker',.25,.99), candidate('gameplay_follow',.75,.4)))
            if f.frame_index>=120 else f for f in ev]
        s=service();shot=s._consume_file(Video(n=240),ev)[0]
        _,_,expected=oracle(ev,Video(n=240).info)
        np.testing.assert_allclose(shot.x_coordinates,expected,atol=1e-12,rtol=0)
        self.assertEqual(shot.focus_samples[-1].family,'active_speaker')
    def test_legacy_selection_and_smoothing_not_called(self):
        s=service()
        with patch.object(s,'_select_focus_series',side_effect=AssertionError('legacy selection used')),patch.object(s,'_smooth_from',side_effect=AssertionError('legacy smoothing used')):
            s._consume_file(Video(),evidence())
    def test_legacy_hold_flags_do_not_override_reviewed_behavior(self):
        a,b=service(),service(hold_and_cut_families='gameplay_follow',min_hold_seconds=20.)
        sa=a._consume_file(Video(),evidence())[0];sb=b._consume_file(Video(),evidence())[0]
        self.assertEqual(sa.x_coordinates,sb.x_coordinates)
    def test_per_shot_reset(self):
        s=service();a=s._consume_file(Video('a'),evidence())[0];b=s._consume_file(Video('b'),evidence())[0]
        self.assertEqual(a.x_coordinates,b.x_coordinates)
        self.assertEqual((a.start_frame,b.start_frame),(0,0))
    def test_shot_relative_clock_not_rounded_ms(self):
        v=Video(fps=60000/1001);ev=evidence(fps=v.info.fps)
        s=service();a=s._consume_file(v,ev)[0]
        _,_,expected=oracle(ev,v.info)
        np.testing.assert_allclose(a.x_coordinates,expected,atol=1e-12,rtol=0)
    def test_metadata_timestamps_preserved(self):
        v=Video(fps=59.94);ev=evidence(fps=v.info.fps)
        s=service();shot=s._consume_file(v,ev)[0]
        self.assertEqual([f.timestamp_ms for f in shot.focus_samples],[f.timestamp_ms for f in ev])
    def test_empty_detection_samples_use_reviewed_fallback(self):
        ev=evidence();ev=[replace(f,candidates=()) if f.frame_index>=30 else f for f in ev]
        v=Video();s=service();a=s._consume_file(v,ev)[0]
        _,_,expected=oracle(ev,v.info)
        np.testing.assert_allclose(a.x_coordinates,expected,atol=1e-12,rtol=0)
    def test_all_detection_samples_empty(self):
        s=service();a=s._consume_file(Video(),[replace(f,candidates=()) for f in evidence()])[0]
        self.assertEqual(set(a.x_coordinates),{.5});self.assertEqual(a.family,'safe_center')
    def test_single_evidence_sample(self):
        s=service();a=s._consume_file(Video(n=1),[evidence()[0]])[0]
        self.assertEqual(len(a.x_coordinates),1);self.assertTrue(math.isfinite(a.x_coordinates[0]))
    def test_no_samples_short_interval_preserves_center_fallback(self):
        s=service();a=s._consume_file(Video(n=2),[])[0]
        self.assertEqual(a.x_coordinates,(.5,.5))
    def test_nonzero_first_sample(self):
        v=Video();ev=evidence(offset=3);s=service();a=s._consume_file(v,ev)[0]
        _,_,expected=oracle(ev,v.info)
        np.testing.assert_allclose(a.x_coordinates,expected,atol=1e-12,rtol=0)
    def test_wrong_way_fixture_matches_reference_without_tuning(self):
        ev=evidence(n=300,x=.7)
        ev=[replace(f,candidates=(candidate('gameplay_follow',.125,.039),candidate('gameplay_follow',.72,.005)))
            if 180<=f.frame_index<192 else f for f in ev]
        v=Video(n=300);s=service();a=s._consume_file(v,ev)[0]
        _,_,expected=oracle(ev,v.info)
        np.testing.assert_allclose(a.x_coordinates,expected,atol=1e-12,rtol=0)


class StreamContracts(unittest.TestCase):
    def feed(self,s,n=6,fps=60.):
        completed=[]
        for k in range(n):completed.extend(s._consume_file(Video(str(k),fps=fps),evidence(fps=fps)))
        return completed
    def test_stream_matches_whole_shot_at_default_commit_lag(self):
        s=service('segment_file',Detector());self.feed(s)
        a=s.finalize()[0]
        w=service();b=w._consume_file(Video(n=720),evidence(n=720))[0]
        np.testing.assert_allclose(a.x_coordinates,b.x_coordinates,rtol=0,atol=1e-12)
        np.testing.assert_allclose(s.x_sink.values,w.x_sink.values,rtol=0,atol=1e-12)
    def test_stream_partition_independence_with_identical_evidence(self):
        a,b=service('segment_file',Detector()),service('segment_file',Detector())
        self.feed(a,6)
        b._consume_file(Video('0',n=240),evidence(n=240));b._consume_file(Video('1',n=480),evidence(n=480))
        aa,bb=a.finalize()[0],b.finalize()[0]
        np.testing.assert_allclose(aa.x_coordinates,bb.x_coordinates,atol=1e-12,rtol=0)
    def test_runtime_survives_segment_boundary(self):
        s=service('segment_file',Detector());self.feed(s,5)
        r=s._cycle_reviewed_runtime;self.assertIsNotNone(r);previous=s._cycle_reviewed_last_frame
        s._consume_file(Video('5'),evidence())
        self.assertIs(s._cycle_reviewed_runtime,r);self.assertGreater(s._cycle_reviewed_last_frame,previous)
    def test_all_committed_sample_frames_once(self):
        s=service('segment_file',Detector());self.feed(s,6)
        seen=list(s._cycle_indices);s.finalize()
        self.assertEqual(len(s.x_sink.values),120);self.assertEqual(len(seen),len(set(seen)))
    def test_commit_lag_not_changed(self):
        s=service('segment_file',Detector());self.feed(s,5)
        self.assertEqual(s._commit_lag,300);self.assertTrue(s._cycle_indices)
        self.assertLess(max(s._cycle_indices),s._abs_frames-s._commit_lag)
    def test_family_gate_timing_not_changed(self):
        s=service('segment_file',Detector());self.feed(s,4)
        self.assertIsNone(s.current_family);self.assertEqual(s.x_sink.values,[])
        s._consume_file(Video('4'),evidence());self.assertEqual(s.current_family,'gameplay_follow')
    def test_future_pending_evidence_does_not_change_committed_prefix(self):
        a,b=service('segment_file',Detector()),service('segment_file',Detector())
        self.feed(a,4);self.feed(b,4)
        a._consume_file(Video('4'),evidence(x=.7))
        b._consume_file(Video('4'),evidence(x=.25))
        self.assertEqual(a.x_sink.values,b.x_sink.values)
    def test_deferred_cut_resets_runtime_with_remainder(self):
        s=service('segment_file',Detector({'5':[-10]}));self.feed(s,5)
        before=s._cycle_reviewed_runtime
        shots=s._consume_file(Video('5'),evidence());self.assertEqual(len(shots),1)
        first=shots[0];last=s.finalize()[0]
        self.assertEqual((first.start_frame,first.frame_count),(-600,590))
        self.assertEqual((last.start_frame,last.frame_count),(-10,130))
        self.assertEqual(first.frame_count+last.frame_count,720)
        self.assertIsNot(s._cycle_reviewed_runtime,before)
    def test_committed_prefix_survives_late_cut(self):
        s=service('segment_file',Detector({'5':[-10]}));self.feed(s,5)
        committed=list(zip(s._cycle_indices,s._cycle_smoothed))
        self.assertTrue(committed)
        shot=s._consume_file(Video('5'),evidence())[0]
        for fi,x in committed:self.assertAlmostEqual(shot.x_coordinates[fi],x,12)
    def test_cut_gap_free_tiling(self):
        s=service('segment_file',Detector({'0':[60],'2':[-5,90]}));out=self.feed(s,3)+s.finalize()
        self.assertEqual(sum(a.frame_count for a in out),360)
    def test_no_state_bleed_between_shots_inside_segment(self):
        s=service('segment_file',Detector({'a':[120]}));ev=evidence(n=240)
        shots=s._consume_file(Video('a',n=240),ev)+s.finalize()
        self.assertEqual(len(shots),2);self.assertEqual(shots[0].x_coordinates,shots[1].x_coordinates)
    def test_micro_cut_guard_preserved(self):
        s=service('segment_file',Detector({'1':[10,12,16]}));out=self.feed(s,2)+s.finalize()
        self.assertEqual(len(out),2);self.assertEqual([a.frame_count for a in out],[130,110])
    def test_no_output_without_inputs(self):
        s=service('segment_file',Detector());self.assertEqual(s.finalize(),[]);self.assertEqual(s.x_sink.values,[])
    def test_eof_flush_once(self):
        s=service('segment_file',Detector());self.feed(s,1)
        self.assertEqual(len(s.finalize()),1);n=len(s.x_sink.values)
        self.assertEqual(s.finalize(),[]);self.assertEqual(len(s.x_sink.values),n)
    def test_live_sink_sample_cadence_not_frame_cadence(self):
        s=service();a=s._consume_file(Video(),evidence())[0]
        self.assertEqual(len(a.x_coordinates),120);self.assertEqual(len(s.x_sink.values),20)
    def test_live_fixed_point_encoding_preserved(self):
        self.assertEqual(encode_x(.613),struct.pack('<I',6130))
        s=service();s._consume_file(Video(),evidence())
        blob=b''.join(encode_x(v) for v in s.x_sink.values);self.assertEqual(len(blob),80)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'out.bin';FileSink(str(p)).publish(s.x_sink.values)
            self.assertEqual(p.read_bytes(),blob)
    def test_open_shot_geometry_change_rejected(self):
        s=service('segment_file',Detector());self.feed(s,5)
        with self.assertRaisesRegex(ValueError,'geometry/FPS'):
            s._consume_file(Video('5',width=1280,height=720),evidence())


class Reconstruction(unittest.TestCase):
    def test_speaker_jump_is_not_interpolated(self):
        a=expand_to_source_frames(sample_frame_indices=[0,6,12],sample_x=[.3,.8,.8],
             sample_events=[0,2,0],start_frame=0,end_frame=13,legal_min=.15,legal_max=.85)
        np.testing.assert_array_equal(a[:6],np.full(6,.3));self.assertEqual(a[6],.8)
    def test_negative_offsets_and_jump_at_exact_frame(self):
        a=expand_to_source_frames(sample_frame_indices=[-12,-6,0],sample_x=[.3,.8,.7],
             sample_events=[0,2,0],start_frame=-12,end_frame=1,legal_min=.15,legal_max=.85)
        self.assertEqual(len(a),13);self.assertEqual(a[6],.8);self.assertTrue(np.all(a[:6]==.3))
    def test_legacy_expansion_unchanged_when_no_events(self):
        a=expand_to_source_frames(sample_frame_indices=[0,6,12],sample_x=[.3,.8,.7],
            start_frame=-2,end_frame=15,legal_min=.15,legal_max=.85)
        np.testing.assert_array_equal(a,np.interp(np.arange(-2,15),[0,6,12],[.3,.8,.7]))
    def test_no_jump_event_matches_legacy(self):
        args=dict(sample_frame_indices=[0,6,12],sample_x=[.3,.8,.7],start_frame=0,end_frame=15,legal_min=.15,legal_max=.85)
        np.testing.assert_array_equal(expand_to_source_frames(**args),expand_to_source_frames(**args,sample_events=[0,0,0]))
    def test_event_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            expand_to_source_frames(sample_frame_indices=[0],sample_x=[.5],sample_events=[],start_frame=0,end_frame=1,legal_min=0,legal_max=1)
    def test_nan_reconstruction_rejected(self):
        with self.assertRaises(ValueError):
            expand_to_source_frames(sample_frame_indices=[0],sample_x=[float('nan')],sample_events=[0],start_frame=0,end_frame=1,legal_min=0,legal_max=1)
    def test_reversed_sample_order_rejected(self):
        with self.assertRaises(ValueError):
            expand_to_source_frames(sample_frame_indices=[6,0],sample_x=[.5,.6],sample_events=[0,0],start_frame=0,end_frame=12,legal_min=0,legal_max=1)
    def test_portrait_and_equal_aspect_hold_center(self):
        for w,h in ((1080,1920),(900,1920)):
            with self.subTest(geometry=(w,h)):
                s=service();a=s._consume_file(Video(width=w,height=h),evidence())[0]
                self.assertEqual(set(a.x_coordinates),{.5});self.assertEqual(a.crop_width_norm,1.)
    def test_custom_aspect_remains_crop_safe(self):
        for w,h,aspect in ((1920,1080,1.),(900,1920,.2)):
            with self.subTest(geometry=(w,h,aspect)):
                s=service(target_aspect_width_over_height=aspect)
                a=s._consume_file(Video(width=w,height=h),evidence(x=.95))[0]
                _,lo,hi=legal_crop_geometry(w,h,aspect)
                self.assertTrue(all(lo<=x<=hi for x in a.x_coordinates))
    def test_invalid_evidence_rejected_before_controller_state(self):
        for bad in (Candidate('bad',.8,(.1,.2,.3,.8)),candidate(confidence=float('nan')),
                    Candidate('gameplay_follow',.8,(.8,.2,.3,.8))):
            s=service();s._last_video_info=Video().info
            with self.assertRaises(ValueError):s._reviewed_samples([FrameEvidence(0,0,(bad,))])
            self.assertEqual(s._cycle_indices,[])
    def test_duplicate_and_out_of_order_samples_rejected(self):
        for frames in ([0,0],[6,0]):
            s=service();s._last_video_info=Video().info
            ev=[FrameEvidence(i,round(i/60*1000),(candidate(),)) for i in frames]
            with self.assertRaises(ValueError):s._reviewed_samples(ev)
            self.assertEqual(s._cycle_indices,[])


class ModelCompatibility(unittest.TestCase):
    def test_pre_v4_constructor_still_accepted(self):
        # Constructor verifies files but does not load Ultralytics or allocate a GPU.
        m=YoloStudentModel(model_path=str(ROOT/'models/nba_yolo_student/best.pt'),
            manifest_path=str(ROOT/'models/nba_yolo_student/model_manifest.json'),
            verify_sha256=True,device='cpu',imgsz=1280,min_confidence=.001,
            max_detections=100,use_fp16=False)
        self.assertEqual((m.iou,m.top_k),(.7,20));self.assertIsNone(m._model)
    def test_active_service_constructs_and_uses_reviewed_runtime(self):
        s=ShotFocusService(config_from_params({'input_mode':'shot_file','device':'cpu',
             'model_path':str(ROOT/'models/nba_yolo_student/best.pt'),
             'model_manifest_path':str(ROOT/'models/nba_yolo_student/model_manifest.json'),
             'runtime_manifest_path':MANIFEST}),NoopSink())
        self.assertIsNone(s.model._model)
        a=s._consume_file(Video(),evidence())[0]
        self.assertEqual(a.model.artifact_version,'round01_schemafix_v3_reviewed_oneeuro')
        self.assertNotEqual(a.x_coordinates[0],a.x_coordinates[-1])
    def test_quantize_fallback_retained(self):
        m=YoloStudentModel.__new__(YoloStudentModel)
        m.imgsz=1280;m.min_confidence=.001;m.iou=.7;m.max_detections=100;m.device='cpu';m.use_fp16=False
        class Fake:
            def __init__(self):self.calls=[]
            def predict(self,**kw):
                self.calls.append(kw)
                if 'quantize' in kw:raise SyntaxError('quantize unsupported')
                return []
        m._model=Fake();self.assertEqual(m._predict([]),[])
        self.assertEqual(len(m._model.calls),2);self.assertFalse(m._model.calls[-1]['half'])



class WireContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing botics protocol test doubles, not external services.
        import importlib.util, sys
        path=ROOT/'tests/test_common_ml_contract.py'
        spec=importlib.util.spec_from_file_location('wire_contract_fixtures',path)
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        cls.fixtures=m
    def test_real_integrated_shot_serializes_same_payload_fields(self):
        from nba_yolo_shot_tagger.contract import tags_for_analysis
        s=service();shot=s._consume_file(Video(),evidence())[0]
        tag=tags_for_analysis(shot,s.config)[0];info=tag.additional_info
        self.assertEqual((tag.source_media,tag.track,tag.start_time,tag.end_time),('a.mp4','vertical_video',0,2000))
        self.assertEqual(tag.frame_info,{'frame_idx':0})
        self.assertEqual(info['frame_count'],120);self.assertEqual(len(info['x-coordinates']),120)
        self.assertEqual(info['x_coordinate_alignment'],'source_frame')
        self.assertEqual(info['x_coordinate_units'],'normalized_source_width')
        self.assertEqual(info['focus_sample_fps'],10.)
        self.assertNotIn('focus_samples',info)
    def test_optional_focus_samples_still_work(self):
        from nba_yolo_shot_tagger.contract import tags_for_analysis
        s=service(include_focus_samples=True,emit_focus_track=True)
        shot=s._consume_file(Video(),evidence())[0];tags=tags_for_analysis(shot,s.config)
        self.assertEqual(len(tags),2);self.assertEqual(tags[1].track,'focus')
        self.assertEqual(len(tags[0].additional_info['focus_samples']),20)
    def test_negative_offsets_survive_tag_serialization(self):
        from nba_yolo_shot_tagger.contract import tags_for_analysis
        s=service('segment_file',Detector({'1':[-10]}))
        s._consume_file(Video('0'),evidence())
        shot=s._consume_file(Video('1'),evidence())[0]
        tag=tags_for_analysis(shot,s.config)[0]
        self.assertEqual(tag.source_media,'1');self.assertEqual(tag.frame_info,{'frame_idx':-120})
        self.assertLess(tag.start_time,0);self.assertEqual(tag.additional_info['frame_count'],110)
    def test_producer_progress_after_shot_output(self):
        from nba_yolo_shot_tagger.producer import NbaShotFocusProducer
        s=service();shot=s._consume_file(Video(),evidence())[0]
        with patch.object(s,'analyze_file',return_value=[shot]):
            output=list(NbaShotFocusProducer(s.config,s).produce(['a.mp4']))
        self.assertIsInstance(output[0],self.fixtures.FakeTag)
        self.assertIsInstance(output[-1],self.fixtures.FakeProgress)
        self.assertFalse(any(isinstance(v,self.fixtures.FakeError) for v in output))

if __name__=='__main__':unittest.main(verbosity=2)
