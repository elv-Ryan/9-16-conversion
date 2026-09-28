import json,tempfile,unittest
from pathlib import Path
import numpy as np
from nba_yolo_shot_tagger.reviewed_runtime import ReviewedRound01Runtime,reconstruct_source_frames
class T(unittest.TestCase):
 def manifest(self,p):
  p.write_text(json.dumps({'selected':{'kind':'filter_only'},'controller_params':{'global':{'center_fallback_timeout_seconds':1.,'max_missing_target_hold_seconds':.5},'gameplay_follow':{'acquire_confidence':.04,'maintain_confidence':0.,'occlusion_hold_seconds':2.,'pan_time_constant_seconds':.65},'person_subject':{'acquire_confidence':.08,'maintain_confidence':.02,'identity_hold_seconds':2.,'pan_time_constant_seconds':.8},'active_speaker':{'acquire_confidence':.05,'maintain_confidence':.02,'speaker_change_confirmation_observations':5,'speaker_jump_distance_x':.12,'speaker_jump_cooldown_seconds':.5}},'filter_config':{'kind':'one_euro','min_cutoff':.4,'beta':1.}}))
 def test_static(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'m.json';self.manifest(p);r=ReviewedRound01Runtime(str(p),1920,1080)
   c=[{'class_id':1,'family':'gameplay_follow','confidence':.9,'bbox_xyxy_norm':[.45,.1,.55,.9]}]
   xs=[r.update(c,i*.1,shot_start=i==0)[0] for i in range(10)];self.assertLess(np.ptp(xs),1e-8)
 def test_frame_expansion(self):
  y=reconstruct_source_frames([0,.1,.2],[.4,.5,.6],[0,0,0],0,13,60,1);self.assertEqual(len(y),13);self.assertAlmostEqual(float(y[0]),.4,5)
if __name__=='__main__':unittest.main()
