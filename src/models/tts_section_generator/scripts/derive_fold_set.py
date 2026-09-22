#!/usr/bin/env python3
"""Derive the set of Latin characters espeak cannot pronounce (T438647).

espeak falls back to spelling out a character's codepoint when its
language data has no pronunciation and no accent name for it (see the
``$accent`` flag in espeak-ng's dictionary documentation). The result is
audible nonsense and a large phoneme count: "Nguyễn" becomes 45 phonemes
reading "N G U Y letter one E C five N" where "Nguyen" is 7.

The generator folds exactly these characters to ASCII before synthesis.
Which characters they are depends on the espeak version behind
kokoro-onnx, so the set is DERIVED rather than hand-written, and this
script is how it is regenerated and re-verified.

Run it inside the tts model-server image, which is the only place with
espeak and kokoro-onnx (the generator image has neither):

    docker run --rm --user root -v "$(pwd)":/mnt \\
      --entrypoint python3 \\
      docker-registry.wikimedia.org/.../tts:<tag> /mnt/derive_fold_set.py

It prints the set as a Python literal for tts_generator/text.py, and
exits non-zero if that literal differs from what the generator currently
carries, so an espeak or kokoro upgrade shows up as a diff to review
rather than a silent change in pronunciation.
"""

import sys

from kokoro_onnx.tokenizer import Tokenizer

# espeak's spelled-out fallbacks: "letter <codepoint>", "<base> stroke".
SPELLED_OUT_MARKERS = ("lˌɛɾɚ", "stɹˈoʊk", "dˌiːstɹ")

# Latin blocks that occur in article text. Deliberately wider than the
# characters seen so far, so a new script in the corpus is caught here.
SCAN_RANGES = (
    (0x00C0, 0x024F),  # Latin-1 Supplement, Extended-A, Extended-B
    (0x0250, 0x02AF),  # IPA Extensions
    (0x1E00, 0x1EFF),  # Latin Extended Additional (Vietnamese lives here)
    (0x2C60, 0x2C7F),  # Latin Extended-C
    (0xA720, 0xA7FF),  # Latin Extended-D
)


def derive(tokenizer: Tokenizer) -> str:
    unreadable = []
    for low, high in SCAN_RANGES:
        for codepoint in range(low, high + 1):
            char = chr(codepoint)
            if not char.isalpha():
                continue
            # Phonemize in word context: a bare character is treated as a
            # letter name, which is not what we are testing for.
            phonemes = tokenizer.phonemize(f"x{char}x", lang="en-us")
            if any(marker in phonemes for marker in SPELLED_OUT_MARKERS):
                unreadable.append(char)
    return "".join(unreadable)


def main() -> int:
    derived = derive(Tokenizer())
    print(f"# {len(derived)} characters espeak cannot pronounce")
    print("_UNREADABLE_LATIN = frozenset(")
    for start in range(0, len(derived), 48):
        print(f'    "{derived[start : start + 48]}"')
    print(")")

    try:
        from tts_generator.text import _UNREADABLE_LATIN as current
    except Exception:  # generator not importable from this image
        print(
            "\n(generator not importable here; compare the literal by hand)",
            file=sys.stderr,
        )
        return 0

    missing = sorted(set(derived) - current)
    extra = sorted(current - set(derived))
    if missing or extra:
        print(
            f"\nDIFFERS from the generator's set: "
            f"{len(missing)} missing, {len(extra)} no longer needed",
            file=sys.stderr,
        )
        if missing:
            print(f"  missing: {''.join(missing)}", file=sys.stderr)
        if extra:
            print(f"  extra:   {''.join(extra)}", file=sys.stderr)
        return 1
    print(f"\nmatches the generator's set ({len(current)} characters)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
