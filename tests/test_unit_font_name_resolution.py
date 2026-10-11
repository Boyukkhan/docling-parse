#!/usr/bin/env python
"""Which of a font's names decides the typeface (PDF 32000-1, 9.6.2).

A font dictionary can carry two names. /BaseFont is the PostScript name of the
typeface. /Name is the label the resource dictionary references the font by,
and is deprecated as of PDF 1.7 -- producers put arbitrary strings there. An
arXiv stamp declaring `/BaseFont /Times-Roman /Name /arXivStAmP` matched
nothing under its label and fell through to the generic sans fallback, while
every other renderer set it in a serif.

The test compares two renders of the same /BaseFont against each other rather
than against a named face, so it holds whatever fonts the machine has
installed: adding a label must not move the result.

The other half of that fix -- dropping the Monotype "PS"/"MT" vendor tags
wherever they appear, so TimesNewRomanPS-BoldMT resolves with the same family
as TimesNewRomanPSMT instead of missing the alias on an infix "ps" -- has no
test here. Two font names that differ only in those tags also reach the width
metrics differently in the parse layer, so comparing renders cannot isolate the
face, and asserting a specific face would only hold on a machine that has it.
That change is measured on the comparison corpus instead: 13 pages improved and
none regressed.

The same label used to decide the *characters* as well: the character lookup
matched only the font's reported name (the /Name label when there is no
/FontDescriptor) against the built-in font metrics, missed Symbol.afm and
ZapfDingbats.afm, and fell back to StandardEncoding. ReportLab writes every
base-14 font with such a label (`/BaseFont /Symbol /Name /F3`, no /Encoding,
no /ToUnicode), so its `<=` came out as a pound sign, `>=` as a double dagger,
the ohm sign as `W` and its missing-glyph square as `n` (docling-parse PR
#327). The decoder now selects the built-in font exactly as the width and bbox
lookups do: the reported name first, /BaseFont when that name matches no
built-in font. The tests below pin that selection, the sources that still take
precedence over it (/ToUnicode, /Differences, a declared base encoding, a
/FontDescriptor /FontName) and the one label collision it does not resolve.
"""

from __future__ import annotations

import re
from io import BytesIO

import numpy as np
import pytest

from docling_parse.pdf_parser import DoclingPdfParser
from tests.pdf_builder import build_pdf, content_stream, render_page, stream_object
from tests.rendering_regression import coverage_ratio, region_image

TEXT_BOX = (10.0, 60.0, 290.0, 140.0)
CONTENT = "BT\n/FT 36 Tf\n20 100 Td\n(Hamburg 123) Tj\nET\n"


def _pdf(font_entries: str) -> bytes:
    return build_pdf(
        [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            "/Resources << /Font << /FT 5 0 R >> >> /Contents 4 0 R >>",
            content_stream(CONTENT),
            f"<< /Type /Font /Subtype /Type1 {font_entries} >>",
        ]
    )


def _render(font_entries: str):
    return region_image(render_page(_pdf(font_entries)), TEXT_BOX)


def test_resource_label_does_not_decide_the_typeface():
    """/Name is a label; adding one must not change the face that is drawn.

    This is the arXiv stamp: same /BaseFont, one of them additionally
    labelled, and the labelled one used to resolve somewhere else entirely.
    """
    plain_image = _render("/BaseFont /Times-Roman")
    labelled_image = _render("/BaseFont /Times-Roman /Name /arXivStAmP")

    assert coverage_ratio(plain_image) > 0.01, (
        "the fixture drew no text, so the comparison would be vacuous"
    )

    plain = np.asarray(plain_image.convert("L"), dtype=np.float64)
    labelled = np.asarray(labelled_image.convert("L"), dtype=np.float64)
    assert plain.shape == labelled.shape, (
        f"renders differ in size: {plain.shape} vs {labelled.shape}"
    )

    difference = float(np.abs(plain - labelled).mean())
    assert difference < 1.0, (
        f"a /Name label changed the rendered face (mean abs difference "
        f"{difference:.2f}); /BaseFont names the typeface, /Name does not"
    )


# --- characters: decoding through a base-14 font's built-in table ----------


def _parse_chars(font_entries: str, code: str) -> list[str]:
    pdf = build_pdf(
        [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            "/Resources << /Font << /FT 5 0 R >> >> /Contents 4 0 R >>",
            content_stream(f"BT\n/FT 12 Tf\n20 100 Td\n({code}) Tj\nET\n"),
            f"<< /Type /Font /Subtype /Type1 {font_entries} >>",
        ]
    )
    parser = DoclingPdfParser(loglevel="fatal")
    doc = parser.load(path_or_stream=BytesIO(pdf))
    _, page = next(doc.iterate_pages())
    return [cell.text for cell in page.char_cells]


@pytest.mark.parametrize(
    ("base_font", "code", "expected"),
    [
        pytest.param("/Symbol", r"\256", "→", id="symbol-arrowright"),
        pytest.param("/ZapfDingbats", "H", "★", id="dingbats-a35-star"),
    ],
)
def test_resource_label_does_not_decide_the_builtin_encoding(
    base_font: str, code: str, expected: str
):
    """/Name must not shadow /BaseFont when decoding through a built-in encoding.

    Symbol and ZapfDingbats carry no /Encoding, so the reader is obliged to
    decode through the face's built-in table (PDF 32000-1, 9.6.6.2). The
    character lookup used to try only the font's reported name against the
    base-14 metrics; a labelled dictionary reports the label (`/FT` here, the
    obsolete /Name entry), missed Symbol.afm, and fell back to
    StandardEncoding -- where Symbol's arrow at 0256 reads as the ligature
    "fi" (issue #317). (Upstream PR #327's test, ported.)
    """
    unlabelled = _parse_chars(f"/BaseFont {base_font}", code)
    labelled = _parse_chars(f"/BaseFont {base_font} /Name /FT", code)

    assert unlabelled == [expected], (
        f"{base_font} without a label decoded {unlabelled!r}, expected "
        f"[{expected!r}]; the built-in encoding is not being consulted"
    )
    assert labelled == [expected], (
        f"{base_font} with a /Name label decoded {labelled!r}, expected "
        f"[{expected!r}]; the label shadowed /BaseFont in character decoding"
    )


# The fixtures below copy ReportLab's font dictionaries byte for byte (it
# labels every font, and writes no /Encoding for Symbol and ZapfDingbats).
# Each fixture is inspected before its cells are asserted: the font
# dictionary must be exactly what the test claims, and the content stream must
# show exactly the byte codes the test claims, in that font.

LESSEQUAL = "≤"  # Symbol 0xA3 /lessequal
GREATEREQUAL = "≥"  # Symbol 0xB3 /greaterequal
OHM = "Ω"  # Symbol 0x57 /Omega; the glyph list maps Omega to U+2126
BLACK_SQUARE = "■"  # ZapfDingbats 0x6E /a73
STERLING = "£"
MULTIPLY = "×"
ALPHA = "α"
BETA = "β"

HELVETICA = "<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding{label}>>"
SYMBOL = "<</Type/Font/Subtype/Type1/BaseFont/Symbol{label}{extra}>>"
DINGBATS = "<</Type/Font/Subtype/Type1/BaseFont/ZapfDingbats{label}>>"

_TOUNICODE_A3_TO_A = (
    "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
    "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
    "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
    "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
    "1 beginbfchar\n<A3> <0041>\nendbfchar\n"
    "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
)

_FONT_DESCRIPTOR = (
    "<</Type/FontDescriptor/FontName/{name}/Flags {flags}/FontBBox[0 0 1000 1000]"
    "/ItalicAngle 0/Ascent 800/Descent -200/CapHeight 700/StemV 80>>"
)


def _literal(codes: bytes) -> str:
    return "(" + "".join(f"\\{byte:03o}" for byte in codes) + ")"


def _fixture(fonts: dict[str, str], runs: list[tuple[str, bytes]], extra=()) -> bytes:
    """One page; `runs` are shown left to right in one text object.

    `fonts` maps a resource name to its font dictionary; the dictionaries are
    objects 5.. in that order and `extra` objects follow them.
    """
    names = list(fonts)
    resources = " ".join(f"/{name} {5 + i} 0 R" for i, name in enumerate(names))
    content = "BT\n20 100 Td\n"
    for name, codes in runs:
        content += f"/{name} 12 Tf {_literal(codes)} Tj\n"
    content += "ET\n"
    return build_pdf(
        [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] "
            f"/Resources << /Font << {resources} >> >> /Contents 4 0 R >>",
            content_stream(content),
            *fonts.values(),
            *extra,
        ]
    )


_OBJECT = re.compile(rb"(\d+) 0 obj\n(.*?)\nendobj\n", re.S)
_TOKEN = re.compile(rb"\((?:\\.|[^\\)])*\)|/[^\s/<>\[\]()]+|[^\s/<>\[\]()]+")


def _unescape(literal: bytes) -> bytes:
    body, out, i = literal[1:-1], bytearray(), 0
    while i < len(body):
        if body[i : i + 1] == b"\\":
            octal = re.match(rb"[0-7]{1,3}", body[i + 1 :])
            if octal:
                out.append(int(octal.group(0), 8))
                i += 1 + len(octal.group(0))
                continue
            out += body[i + 1 : i + 2]
            i += 2
            continue
        out += body[i : i + 1]
        i += 1
    return bytes(out)


def _inspect(pdf: bytes) -> tuple[dict[str, str], list[tuple[str, int]]]:
    """The page's font dictionaries by resource name, and every shown code
    with the resource name of the font it is shown in, in content order."""
    objects = {int(m.group(1)): m.group(2) for m in _OBJECT.finditer(pdf)}
    page = objects[3].decode("latin-1")
    fonts_entry = re.search(r"/Font << (.*?) >>", page).group(1)
    fonts = {
        name: objects[int(number)].decode("latin-1")
        for name, number in re.findall(r"/(\S+) (\d+) 0 R", fonts_entry)
    }
    stream = objects[4]
    content = stream[stream.index(b"stream\n") + 7 : stream.rindex(b"\nendstream")]
    shown, current, operands = [], None, []
    for token in _TOKEN.findall(content):
        if token == b"Tf":
            current = operands[-2].decode("latin-1")[1:]
        elif token == b"Tj":
            shown += [(current, code) for code in _unescape(operands[-1])]
        if token in (b"Tf", b"Tj", b"Td", b"BT", b"ET"):
            operands = []
        else:
            operands.append(token)
    return fonts, shown


def _cells(pdf: bytes) -> list[tuple[str, str]]:
    parser = DoclingPdfParser(loglevel="fatal")
    doc = parser.load(path_or_stream=BytesIO(pdf))
    _, page = next(doc.iterate_pages())
    return [(cell.text, cell.font_name) for cell in page.char_cells]


def _decode(fonts: dict[str, str], runs: list[tuple[str, bytes]], extra=()) -> list[str]:
    pdf = _fixture(fonts, runs, extra)
    seen_fonts, shown = _inspect(pdf)
    assert seen_fonts == fonts, f"fixture fonts {seen_fonts!r}, expected {fonts!r}"
    assert shown == [(name, code) for name, codes in runs for code in codes], shown
    return [text for text, _ in _cells(pdf)]


def _scalars(texts: list[str]) -> list[str]:
    return [" ".join(f"U+{ord(ch):04X}" for ch in text) for text in texts]


@pytest.mark.parametrize("label", ["/Name/F3", ""], ids=["labelled", "unlabelled"])
def test_symbol_decodes_inequalities_and_ohm_through_its_builtin_table(label: str):
    """ReportLab's Symbol: 0xA3 lessequal, 0xB3 greaterequal, 0x57 Omega.

    Labelled, these used to come out as StandardEncoding's sterling, daggerdbl
    and W. Omega is asserted as U+2126 OHM SIGN, the glyph list's mapping --
    no Unicode normalisation is involved.
    """
    fonts = {"F3": SYMBOL.format(label=label, extra="")}
    decoded = _decode(fonts, [("F3", b"\xa3\xb3\x57")])
    assert _scalars(decoded) == ["U+2264", "U+2265", "U+2126"], decoded
    assert decoded == [LESSEQUAL, GREATEREQUAL, OHM]


def test_label_stays_the_reported_font_name():
    """Only the decoding changed: the cells still report the /Name label."""
    pdf = _fixture({"F3": SYMBOL.format(label="/Name/F3", extra="")}, [("F3", b"\xa3")])
    assert _cells(pdf) == [(LESSEQUAL, "/F3")]


@pytest.mark.parametrize("label", [("/Name/F1", "/Name/F4"), ("", "")], ids=["labelled", "unlabelled"])
def test_dingbats_missing_glyph_square_beside_a_genuine_latin_n(label: tuple[str, str]):
    """ReportLab draws a glyph it cannot encode as ZapfDingbats 0x6E (a73, a
    black square). After "10" it used to read "10n", inventing an exponent the
    page does not contain; the genuine Latin "n" next to it must stay "n"."""
    helvetica, dingbats = label
    fonts = {"F1": HELVETICA.format(label=helvetica), "F4": DINGBATS.format(label=dingbats)}
    decoded = _decode(fonts, [("F1", b"10"), ("F4", b"n"), ("F1", b"n")])
    assert _scalars(decoded) == ["U+0031", "U+0030", "U+25A0", "U+006E"], decoded
    assert "".join(decoded) == "10" + BLACK_SQUARE + "n"


@pytest.mark.parametrize(
    "font",
    [
        HELVETICA.format(label="/Name/F1"),
        HELVETICA.format(label=""),
        "<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Name/F1>>",
    ],
    ids=["winansi-labelled", "winansi-unlabelled", "builtin-standard-labelled"],
)
def test_genuine_helvetica_sterling_and_w_stay(font: str):
    """A real pound sign and a real W in a Latin font are not symbols."""
    decoded = _decode({"F1": font}, [("F1", b"\xa3W")])
    assert _scalars(decoded) == ["U+00A3", "U+0057"], decoded


@pytest.mark.parametrize("label", ["/Name/F3", ""], ids=["labelled", "unlabelled"])
def test_tounicode_entry_wins_over_the_builtin_table(label: str):
    """A valid /ToUnicode entry is the first method (9.10.2); Symbol's own
    lessequal at 0xA3 must not override it. A code the cmap does not cover
    falls back to the built-in table."""
    fonts = {"F3": SYMBOL.format(label=label, extra="/ToUnicode 6 0 R")}
    extra = [stream_object("", _TOUNICODE_A3_TO_A.encode("latin-1"))]
    decoded = _decode(fonts, [("F3", b"\xa3\xb3")], extra)
    assert decoded == ["A", GREATEREQUAL], _scalars(decoded)


@pytest.mark.parametrize("label", ["/Name/F3", ""], ids=["labelled", "unlabelled"])
def test_differences_entry_wins_over_the_builtin_table(label: str):
    """A conflicting /Differences entry names the glyph (9.6.6.1); a code it
    does not mention keeps the built-in encoding of the symbolic font."""
    fonts = {
        "F3": SYMBOL.format(
            label=label, extra="/Encoding<</Type/Encoding/Differences[163/sterling]>>"
        )
    }
    decoded = _decode(fonts, [("F3", b"\xa3\xb3")])
    assert decoded == [STERLING, GREATEREQUAL], _scalars(decoded)


@pytest.mark.parametrize("label", ["/Name/F3", ""], ids=["labelled", "unlabelled"])
def test_declared_winansi_on_symbol_keeps_multiply(label: str):
    """The declared base encoding governs where Symbol has its glyph: WinAnsi
    0xD7 is `multiply`, not Symbol's own `dotmath`; `a` and `b`, which
    Symbol lacks, take Symbol's codes (alpha, beta). A label does not change
    that any more than it changes the face."""
    fonts = {"F3": SYMBOL.format(label=label, extra="/Encoding/WinAnsiEncoding")}
    decoded = _decode(fonts, [("F3", b"a\xd7b")])
    assert decoded[1] == MULTIPLY, _scalars(decoded)
    assert decoded == [ALPHA, MULTIPLY, BETA], _scalars(decoded)


@pytest.mark.parametrize(
    ("base_font", "descriptor_name", "flags", "expected"),
    [
        pytest.param("Symbol", "Helvetica", 32, STERLING, id="fontname-helvetica-over-basefont-symbol"),
        pytest.param("Helvetica", "Symbol", 4, LESSEQUAL, id="fontname-symbol-over-basefont-helvetica"),
    ],
)
def test_font_descriptor_font_name_keeps_its_priority(
    base_font: str, descriptor_name: str, flags: int, expected: str
):
    """A /FontDescriptor /FontName that names a built-in font decides the
    table; /BaseFont is consulted only when the name matches none."""
    font = f"<</Type/Font/Subtype/Type1/BaseFont/{base_font}/Name/F3/FontDescriptor 6 0 R>>"
    extra = [_FONT_DESCRIPTOR.format(name=descriptor_name, flags=flags)]
    decoded = _decode({"F3": font}, [("F3", b"\xa3")], extra)
    assert decoded == [expected], _scalars(decoded)


def test_label_that_names_a_builtin_font_still_selects_it():
    """Documented limitation, pinned as today's behaviour.

    The selection keeps a successful name match, and a /Name label that itself
    contains a built-in font's name is such a match: `/Name /Helvetica` with
    `/BaseFont /Symbol` decodes as Helvetica, so 0xA3 is Helvetica's sterling,
    not Symbol's lessequal. Matching is a normalised substring/longest-name
    test, so other labels containing a built-in name collide the same way.
    Resolving it means choosing a real font identity for decoding; if that is
    ever done, this assertion is meant to change with it.
    """
    font = "<</Type/Font/Subtype/Type1/BaseFont/Symbol/Name/Helvetica>>"
    decoded = _decode({"F3": font}, [("F3", b"\xa3")])
    assert _scalars(decoded) == ["U+00A3"], decoded


def test_labelled_type3_font_decodes_through_its_differences():
    """Type3 has no /BaseFont to fall back to; its /Differences decide."""
    font = (
        "<</Type/Font/Subtype/Type3/Name/F3/FontBBox[0 0 1000 1000]"
        "/FontMatrix[0.001 0 0 0.001 0 0]/CharProcs 6 0 R"
        "/Encoding<</Type/Encoding/Differences[163/lessequal 179/sterling]>>"
        "/FirstChar 163/LastChar 179/Widths[500 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 500]"
        "/Resources<<>>>>"
    )
    glyph = stream_object("", b"500 0 0 0 500 700 d1\n0 0 500 700 re f\n")
    extra = ["<</lessequal 7 0 R/sterling 7 0 R>>", glyph]
    decoded = _decode({"F3": font}, [("F3", b"\xa3\xb3")], extra)
    assert decoded == [LESSEQUAL, STERLING], _scalars(decoded)
