from enum import Enum
from typing import Any, Dict, List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from src.models.edit_check.model_server.config import settings


class Language(str, Enum):
    aa = "aa"
    ab = "ab"
    af = "af"
    ak = "ak"
    am = "am"
    ar = "ar"
    as_ = "as"
    av = "av"
    ay = "ay"
    az = "az"
    ba = "ba"
    be = "be"
    bg = "bg"
    bh = "bh"
    bi = "bi"
    bm = "bm"
    bn = "bn"
    bo = "bo"
    br = "br"
    bs = "bs"
    ca = "ca"
    ce = "ce"
    ch = "ch"
    co = "co"
    cr = "cr"
    cs = "cs"
    cu = "cu"
    cv = "cv"
    cy = "cy"
    da = "da"
    de = "de"
    dv = "dv"
    dz = "dz"
    ee = "ee"
    el = "el"
    en = "en"
    eo = "eo"
    es = "es"
    et = "et"
    eu = "eu"
    fa = "fa"
    ff = "ff"
    fi = "fi"
    fj = "fj"
    fo = "fo"
    fr = "fr"
    fy = "fy"
    ga = "ga"
    gd = "gd"
    gl = "gl"
    gn = "gn"
    gu = "gu"
    gv = "gv"
    ha = "ha"
    he = "he"
    hi = "hi"
    ho = "ho"
    hr = "hr"
    ht = "ht"
    hu = "hu"
    hy = "hy"
    hz = "hz"
    ia = "ia"
    id = "id"
    ie = "ie"
    ig = "ig"
    ii = "ii"
    ik = "ik"
    io = "io"
    is_ = "is"
    it = "it"
    iu = "iu"
    ja = "ja"
    jv = "jv"
    ka = "ka"
    kg = "kg"
    ki = "ki"
    kj = "kj"
    kk = "kk"
    kl = "kl"
    km = "km"
    kn = "kn"
    ko = "ko"
    kr = "kr"
    ks = "ks"
    ku = "ku"
    kv = "kv"
    kw = "kw"
    ky = "ky"
    la = "la"
    lb = "lb"
    lg = "lg"
    li = "li"
    ln = "ln"
    lo = "lo"
    lt = "lt"
    lu = "lu"
    lv = "lv"
    mg = "mg"
    mh = "mh"
    mi = "mi"
    mk = "mk"
    ml = "ml"
    mn = "mn"
    mr = "mr"
    ms = "ms"
    mt = "mt"
    my = "my"
    na = "na"
    nb = "nb"
    nd = "nd"
    ne = "ne"
    ng = "ng"
    nl = "nl"
    nn = "nn"
    no = "no"
    nr = "nr"
    nv = "nv"
    ny = "ny"
    oc = "oc"
    oj = "oj"
    om = "om"
    or_ = "or"
    os = "os"
    pa = "pa"
    pi = "pi"
    pl = "pl"
    ps = "ps"
    pt = "pt"
    qu = "qu"
    rm = "rm"
    rn = "rn"
    ro = "ro"
    ru = "ru"
    rw = "rw"
    sa = "sa"
    sc = "sc"
    sd = "sd"
    se = "se"
    sg = "sg"
    si = "si"
    sk = "sk"
    sl = "sl"
    sm = "sm"
    sn = "sn"
    so = "so"
    sq = "sq"
    sr = "sr"
    ss = "ss"
    st = "st"
    su = "su"
    sv = "sv"
    sw = "sw"
    ta = "ta"
    te = "te"
    tg = "tg"
    th = "th"
    ti = "ti"
    tk = "tk"
    tl = "tl"
    tn = "tn"
    to = "to"
    tr = "tr"
    ts = "ts"
    tt = "tt"
    tw = "tw"
    ty = "ty"
    ug = "ug"
    uk = "uk"
    ur = "ur"
    uz = "uz"
    ve = "ve"
    vi = "vi"
    vo = "vo"
    wa = "wa"
    wo = "wo"
    xh = "xh"
    yi = "yi"
    yo = "yo"
    za = "za"
    zh = "zh"
    zu = "zu"


class CheckType(str, Enum):
    peacock = "peacock"
    npov = "npov"
    weasel = "weasel"


class Instance(BaseModel):
    lang: Language = Field(
        ..., description="Language code of the text, e.g., 'en' for English."
    )
    check_type: CheckType = Field(
        ..., description="Type of check to perform, e.g., 'peacock'."
    )
    original_text: str = Field(..., description="The original text to be checked.")
    modified_text: str = Field(..., description="The modified text to be checked.")
    return_shap_values: bool | None = Field(
        default=False, description="Whether to return SHAP values."
    )

    @field_validator("original_text", "modified_text", mode="after")
    def text_length(cls, value):
        if len(value) > settings.max_char_length:
            raise ValueError(
                f"Text fields must be less than {settings.max_char_length} characters long"
            )
        return value

    @field_validator("modified_text", mode="after")
    def texts_must_be_different(cls, value, info):
        other_text = info.data.get("original_text")
        if other_text is not None and value == other_text:
            raise ValueError("Original text and modified text must be different")
        return value

    model_config = ConfigDict(extra="forbid")


class RequestModel(BaseModel):
    instances: List[Dict[str, Any]] = Field(
        ..., description="List of instances to be processed."
    )

    @field_validator("instances")
    def check_instances_length(cls, value):
        if len(value) == 0:
            raise ValueError("At least one instance must be provided")
        if len(value) > settings.max_batch_size:
            raise ValueError(
                f"Request length is greater than max batch size: {len(value)} > {settings.max_batch_size}"
            )
        return value

    def process_instances(self):
        responses = {"Valid": [], "Malformed": []}

        for index, instance_data in enumerate(self.instances):
            try:
                # Validate each instance fully
                instance = Instance(**instance_data)
                responses["Valid"].append(
                    {
                        "index": index,
                        "status_code": 200,
                        "instance": instance,
                    }
                )
            except ValidationError as e:
                # Collect all errors for invalid instances
                errors = []
                for error in e.errors():
                    field = ".".join(str(loc) for loc in error["loc"])
                    msg = error["msg"]
                    errors.append(f"Error in field '{field}': {msg}")
                responses["Malformed"].append(
                    {
                        "index": index,
                        "status_code": 400,
                        "errors": errors,
                    }
                )
        return responses
