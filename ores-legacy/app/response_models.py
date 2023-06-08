from typing import Dict, Union, Optional, List, Any
from pydantic import BaseModel


class Score(BaseModel):
    prediction: Union[bool, str, List[str]]
    probability: Dict[str, float]


class ModelInfo(BaseModel):
    version: str


class ScoreInfo(BaseModel):
    score: Optional[Score]
    error: Optional[Any]


class WikiScores(BaseModel):
    models: Dict[str, ModelInfo]
    scores: Optional[Dict[str, Dict[str, ScoreInfo]]]


class WikiModels(BaseModel):
    models: Dict[str, ModelInfo]


class ResponseModel(BaseModel):
    arwiki: Optional[WikiScores]
    bnwiki: Optional[WikiScores]
    bswiki: Optional[WikiScores]
    cawiki: Optional[WikiScores]
    cswiki: Optional[WikiScores]
    dewiki: Optional[WikiScores]
    elwiki: Optional[WikiScores]
    enwiki: Optional[WikiScores]
    enwiktionary: Optional[WikiScores]
    eswiki: Optional[WikiScores]
    eswikibooks: Optional[WikiScores]
    eswikiquote: Optional[WikiScores]
    etwiki: Optional[WikiScores]
    euwiki: Optional[WikiScores]
    fakewiki: Optional[WikiScores]
    fawiki: Optional[WikiScores]
    fiwiki: Optional[WikiScores]
    frwiki: Optional[WikiScores]
    frwikisource: Optional[WikiScores]
    glwiki: Optional[WikiScores]
    hewiki: Optional[WikiScores]
    hiwiki: Optional[WikiScores]
    hrwiki: Optional[WikiScores]
    huwiki: Optional[WikiScores]
    hywiki: Optional[WikiScores]
    idwiki: Optional[WikiScores]
    iswiki: Optional[WikiScores]
    itwiki: Optional[WikiScores]
    jawiki: Optional[WikiScores]
    kowiki: Optional[WikiScores]
    lvwiki: Optional[WikiScores]
    nlwiki: Optional[WikiScores]
    nowiki: Optional[WikiScores]
    plwiki: Optional[WikiScores]
    ptwiki: Optional[WikiScores]
    rowiki: Optional[WikiScores]
    ruwiki: Optional[WikiScores]
    simplewiki: Optional[WikiScores]
    sqwiki: Optional[WikiScores]
    srwiki: Optional[WikiScores]
    svwiki: Optional[WikiScores]
    tawiki: Optional[WikiScores]
    testwiki: Optional[WikiScores]
    trwiki: Optional[WikiScores]
    ukwiki: Optional[WikiScores]
    viwiki: Optional[WikiScores]
    wikidatawiki: Optional[WikiScores]
    zhwiki: Optional[WikiScores]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Override the default dict method to exclude None values from the response
        If we don't do this all Optional values will be included in the response e.g.
        "enwiki": { null }
        """
        _ignored = kwargs.pop("exclude_none")
        return super().dict(*args, exclude_none=True, **kwargs)
