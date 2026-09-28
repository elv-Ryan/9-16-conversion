from __future__ import annotations
from typing import Iterator,List
from loguru import logger
from common_ml.tagging.messages import Error,Message,Progress
from common_ml.tagging.producer import TagMessageProducer
from .config import RuntimeConfig
from .contract import tags_for_analysis
from .service import ShotFocusService

class NbaShotFocusProducer(TagMessageProducer):
    """Long-lived Eluvio producer. Every input file is one shot by default."""
    def __init__(self,config:RuntimeConfig):self.config=config;self.service=ShotFocusService(config)
    def produce(self,files:List[str])->Iterator[Message]:
        for source_media in files:
            try:
                for analysis in self.service.analyze_file(source_media):
                    yield from tags_for_analysis(analysis,self.config)
                # Current Eluvio protocol defines progress as terminal for this file.
                yield Progress(source_media=source_media)
            except Exception as error:
                logger.opt(exception=error).error('NBA YOLO shot tagging failed for {}',source_media)
                yield Error(message=f'{type(error).__name__}: {error}',source_media=source_media)
