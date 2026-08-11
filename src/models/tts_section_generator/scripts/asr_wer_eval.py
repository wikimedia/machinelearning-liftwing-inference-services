"""ASR round-trip WER over the TTS sections in MinIO (Phase 1, T433820 QA).

Synthesized audio -> faster-whisper transcription -> WER against the VTT
reference. The reference is the VTT text: the NORMALIZED words the voice
was asked to say (scoring against raw wiki text would count our
normalization fixes as errors).

Run inside the eval image (see Dockerfile.asreval); config via env:
TTS_GEN_S3_ENDPOINT, TTS_GEN_S3_BUCKET, AWS creds, HF_HOME for the model
cache. Prints per-section WER worst-first, the distribution, and the
worst-10 word-level diffs.

Honest-reading notes baked into the method:
* Digits->words on the hypothesis side: Whisper writes "1491" where the
  voice said "fourteen ninety one"; without this pass, number-dense
  sections show inflated WER that has nothing to do with audio quality.
* Absolute WER is NOT a certification: encyclopedic proper nouns inflate
  it (ASR mishears names it has never seen even when spoken perfectly).
  Read the distribution, the worst-N ranking, and deltas across
  generation versions; the plain-English sections approximate the ASR
  noise floor.
"""

import io
import os
import re
import statistics
import sys
import unicodedata

import boto3
import jiwer
from faster_whisper import WhisperModel

ENDPOINT = os.environ.get("TTS_GEN_S3_ENDPOINT", "http://127.0.0.1:9000")
BUCKET = os.environ.get("TTS_GEN_S3_BUCKET", "tts-artifacts")
MODEL = os.environ.get("ASR_MODEL", "medium.en")

# ── tiny digits->words (0..9999 + 4-digit year style), no extra deps ──────
_ONES = (
    "zero one two three four five six seven eight nine ten eleven "
    "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
    "nineteen"
).split()
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


def _two(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + (" " + _ONES[o] if o else "")


def _num_words(n: int) -> str:
    if n < 100:
        return _two(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + _two(r) if r else "")
    if 1000 <= n <= 9999:
        th, r = divmod(n, 100)
        if 10 <= th % 100 and r:  # year style: 1491 -> fourteen ninety one
            return _two(th) + " " + _two(r)
        th2, r2 = divmod(n, 1000)
        out = _ONES[th2] + " thousand"
        if r2 >= 100:
            out += " " + _num_words(r2)
        elif r2:
            out += " " + _two(r2)
        return out
    return str(n)


_ORD_SUFFIX = {
    1: "first",
    2: "second",
    3: "third",
    5: "fifth",
    8: "eighth",
    9: "ninth",
    12: "twelfth",
}


def _ordinal_words(n: int) -> str:
    """0..9999. Every recursive call strictly reduces n (the first version
    recursed with the SAME n for multiples of ten >= 100: "610th" hung)."""
    if n in _ORD_SUFFIX:
        return _ORD_SUFFIX[n]
    if n < 20:
        return _ONES[n] + "th"
    if n < 100:
        t, o = divmod(n, 10)
        if o == 0:
            return _TENS[t][:-1] + "ieth"
        return _TENS[t] + " " + _ordinal_words(o)
    if n % 1000 == 0:
        return _num_words(n // 1000) + " thousandth"
    if n % 100 == 0:
        return _num_words(n // 100) + " hundredth"
    return _num_words(n - n % 100) + " " + _ordinal_words(n % 100)


def _digits_to_words(text: str) -> str:
    # Whisper inverse-normalizes: "22nd", "0.09", "1998". Fold back to words
    # so the comparison is speech-vs-speech, not speech-vs-orthography.
    text = re.sub(
        r"\b(\d{1,4})(st|nd|rd|th)\b", lambda m: _ordinal_words(int(m.group(1))), text
    )
    text = re.sub(
        r"\b(\d{1,4})\.(\d+)\b",
        lambda m: _num_words(int(m.group(1)))
        + " point "
        + " ".join(_ONES[int(d)] for d in m.group(2)),
        text,
    )
    return re.sub(r"\b\d{1,4}\b", lambda m: _num_words(int(m.group(0))), text)


# Letters that do not NFKD-decompose to ASCII (the Polish l-stroke class);
# without this, reference words like "Wladyslaw" lose letters entirely and
# correct audio gets charged as errors.
_FOLD = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ß": "ss",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
        "þ": "th",
        "ð": "d",
    }
)

# Name-alias fold (T433923): pronunciation-whitelist respellings put
# non-canonical forms in the reference (the VTT carries "Vladiswav"),
# while the recognizer, hearing the name CORRECTLY, writes a canonical
# spelling ("Vladislav", sometimes the original "Jagiello"). Without
# folding, every fixed name scores as an error and improvements read as
# regressions. Both sides fold to one token per name. Pairs are the
# whitelist respellings plus recognizer spellings observed in eval
# diffs; extend the alias sets when new whitelist entries land.
_NAME_ALIASES = {
    "wladyslaw": ["vladiswav", "vladislav", "ladislaw", "wladislaw"],
    "jagiello": ["yahg yehwo", "jagielow", "jagjewo", "yag yewo"],
    "jogaila": ["yo guyla", "joe geiler", "yogelya"],
    "vytautas": ["veetowtas"],
    "wkra": ["vukra"],
    "dobrzyn": ["dob zhin", "dobzyn"],
    "bialowieza": ["byahwovyezha", "bialawieza"],
    "zloty": ["zwoty"],
    "raciaz": ["rah chonzh", "ratchanj"],
    "navahrudak": ["nava hroodak", "navahroodak"],
    "shepseskaf": ["shep sess kaf"],
    "soegijapranata": ["soo geeya prah nahta"],
    "aethelwulf": ["athel wulf"],
    "zlotoryja": ["zwo toree ya"],
    "soegija": ["soogeeya"],
}
_ALIAS_SUBS = sorted(
    ((alias, canon) for canon, aliases in _NAME_ALIASES.items() for alias in aliases),
    key=lambda x: -len(x[0]),  # longest first: multi-word aliases win
)


def _fold_name_aliases(text: str) -> str:
    for alias, canon in _ALIAS_SUBS:
        text = re.sub(rf"\b{re.escape(alias)}(?='s\b|\b)", canon, text)
    return text


# BrE reference vs whisper's AmE orthography: fold the classes that showed
# up as false errors (honour/honor, baptised/baptized).
_BRE_AME = [
    (re.compile(r"\bhonour"), "honor"),
    (re.compile(r"our\b"), "or"),
    (re.compile(r"isation\b"), "ization"),
    (re.compile(r"ised\b"), "ized"),
    (re.compile(r"ise\b"), "ize"),
]


def _fold(text: str) -> str:
    text = text.translate(_FOLD)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _norm(text: str, hypothesis: bool) -> str:
    text = _fold(text).lower()
    if hypothesis:
        text = _digits_to_words(text)
    text = text.replace("-", " ")
    for pat, repl in _BRE_AME:
        text = pat.sub(repl, text)
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _fold_name_aliases(text)


def _vtt_text(vtt: str) -> str:
    """Cue payload lines only: skip header, timings, cue ids, NOTE blocks."""
    words = []
    for line in vtt.splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith(("WEBVTT", "NOTE", "STYLE"))
            or "-->" in line
            or line.isdigit()
        ):
            continue
        words.append(re.sub(r"<[^>]+>", "", line))
    return " ".join(words)


def main() -> int:
    s3 = boto3.client("s3", endpoint_url=ENDPOINT)
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET):
        keys += [o["Key"] for o in page.get("Contents", [])]
    pairs = sorted(
        k[:-4] for k in keys if k.endswith(".mp3") and k[:-4] + ".vtt" in keys
    )
    print(f"{len(pairs)} mp3+vtt pairs in {BUCKET} at {ENDPOINT}")
    print(f"loading {MODEL} (cache: {os.environ.get('HF_HOME', '~')})")
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")

    rows, errors = [], []
    for i, base in enumerate(pairs, 1):
        try:
            vtt = s3.get_object(Bucket=BUCKET, Key=base + ".vtt")["Body"].read()
            ref = _norm(_vtt_text(vtt.decode("utf-8")), hypothesis=False)
            mp3 = s3.get_object(Bucket=BUCKET, Key=base + ".mp3")["Body"].read()
            segments, _ = model.transcribe(
                io.BytesIO(mp3), language="en", beam_size=5, vad_filter=False
            )
            hyp = _norm(" ".join(s.text for s in segments), hypothesis=True)
            wer = jiwer.wer(ref, hyp) if ref else float("nan")
            rows.append((wer, base, ref, hyp))
            print(f"  [{i}/{len(pairs)}] {wer:6.1%}  {base}", flush=True)
        except Exception as e:  # noqa: BLE001 - harness resilience
            errors.append((base, repr(e)))
            print(f"  [{i}/{len(pairs)}] ERROR   {base}: {e!r:.120}", flush=True)

    if errors:
        print(
            f"\n{len(errors)} sections errored in the harness (scored "
            f"sections unaffected):"
        )
        for base, err in errors:
            print(f"  {base}: {err[:160]}")
    rows.sort(reverse=True)
    wers = [r[0] for r in rows]
    print("\n=== WER DISTRIBUTION ===")
    print(
        f"  mean {statistics.mean(wers):.1%}   median "
        f"{statistics.median(wers):.1%}   min {min(wers):.1%}   "
        f"max {max(wers):.1%}"
    )
    print("\n=== WORST 10 (the listening queue + whitelist worklist) ===")
    for wer, base, ref, hyp in rows[:10]:
        print(f"\n{wer:6.1%}  {base}")
        out = jiwer.process_words(ref, hyp)
        # Show the first few substitution pairs: usually the mispronounced names
        subs = []
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                r = " ".join(ref.split()[chunk.ref_start_idx : chunk.ref_end_idx])
                h = " ".join(hyp.split()[chunk.hyp_start_idx : chunk.hyp_end_idx])
                subs.append(f"'{r}' -> heard '{h}'")
            if len(subs) >= 6:
                break
        for s_ in subs:
            print(f"    {s_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
