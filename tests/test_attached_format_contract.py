import sys,types,unittest
from dataclasses import dataclass
@dataclass(frozen=True)
class Tag:
 tag:str;start_time:int;end_time:int;source_media:str;track:str='';additional_info:dict|None=None;frame_info:object|None=None;vector:object|None=None
m=types.ModuleType('common_ml.tagging.messages');m.Tag=Tag;sys.modules.setdefault('common_ml',types.ModuleType('common_ml'));sys.modules.setdefault('common_ml.tagging',types.ModuleType('common_ml.tagging'));sys.modules['common_ml.tagging.messages']=m
from nba_yolo_shot_tagger.contract import tags_for_analysis
from nba_yolo_shot_tagger.config import RuntimeConfig
from nba_yolo_shot_tagger.types import ShotAnalysis,ModelIdentity
class T(unittest.TestCase):
 def test_chunks_match_reference_shape(self):
  a=ShotAnalysis('/x/iq__ABC/0.mp4','iq__ABC','s',0,10010,0,377,24000/1001,24000,1001,'24000/1001',1920,1080,'gameplay_follow',.8,.31640625,tuple([.5]*377),tuple(),ModelIdentity('m','a'*64,'v',tuple()))
  tags=tags_for_analysis(a,RuntimeConfig(verify_model_sha256=False))
  self.assertEqual([len(x.additional_info['x-coordinates']) for x in tags],[240,137]);self.assertEqual(tags[0].frame_info,{'frame_idx':0});self.assertEqual(tags[1].frame_info,{'frame_idx':240});self.assertEqual(tags[0].end_time,10010)
  self.assertEqual(set(tags[0].additional_info),{'x-coordinates','source_iq','fps','provenance'})
if __name__=='__main__':unittest.main()
