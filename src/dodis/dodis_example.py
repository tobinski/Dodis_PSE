import argparse
import re
from pathlib import Path
from typing import NamedTuple

import spacy
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "output/dodis/model-best"

# TEI tag → spaCy NER label mapping
TAG_TO_LABEL = {
    "persname": "PER",
    "orgname": "ORG",
    "placename": "LOC",
}


class GoldEntity(NamedTuple):
    text: str
    label: str
    kb_id: str


def _local_tag_name(name: str | None) -> str:
    if not name:
        return ""
    return name.split(":")[-1].lower()


def _normalise_ref(ref: str) -> str:
    return re.sub(r"^https?://", "https://", ref.rstrip("/")).lower() if ref else ""


def extract_gold_entities(path: Path) -> list[GoldEntity]:
    with path.open(encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")
    for tag in soup.find_all(
        lambda t: _local_tag_name(getattr(t, "name", None)) in {"note", "notes"}
    ):
        tag.decompose()
    entities: list[GoldEntity] = []
    for tag in soup.find_all(True):
        local = _local_tag_name(tag.name)
        if local not in TAG_TO_LABEL:
            continue
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        kb_id = _normalise_ref(tag.get("ref", ""))
        entities.append(GoldEntity(text=text, label=TAG_TO_LABEL[local], kb_id=kb_id))
    return entities


def load_clean_text(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")
    for tag in soup.find_all(lambda t: _local_tag_name(getattr(t, "name", None)) in {"note", "notes"}):
        tag.decompose()
    for tag in soup.find_all(
        lambda t: _local_tag_name(getattr(t, "name", None)) in {"persname", "placename", "orgname"}
    ):
        tag.unwrap()
    text = " ".join(soup.stripped_strings)
    return re.sub(r"\s+", " ", text).strip()


def evaluate(gold: list[GoldEntity], doc, verbose: bool = True) -> dict:
    pred_ner: set[tuple[str, str]] = {(e.text, e.label_) for e in doc.ents}
    pred_nel: set[tuple[str, str, str]] = {
        (e.text, e.label_, _normalise_ref(e.kb_id_)) for e in doc.ents if e.kb_id_
    }
    gold_ner: set[tuple[str, str]] = {(g.text, g.label) for g in gold}
    gold_nel: set[tuple[str, str, str]] = {
        (g.text, g.label, g.kb_id) for g in gold if g.kb_id
    }

    def prf(tp, pred_total, gold_total):
        p = tp / pred_total if pred_total else 0.0
        r = tp / gold_total if gold_total else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    def print_prf(tp, pred_total, gold_total, name):
        p, r, f = prf(tp, pred_total, gold_total)
        print(f"  {name:6s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}"
              f"  (tp={tp}, pred={pred_total}, gold={gold_total})")

    if verbose:
        print("\n=== Evaluation ===")
        print("\n[NER]  (text, label) match")

    ner_tp = len(pred_ner & gold_ner)
    if verbose:
        print_prf(ner_tp, len(pred_ner), len(gold_ner), "overall")
        for label in ("PER", "ORG", "LOC"):
            p_l = {e for e in pred_ner if e[1] == label}
            g_l = {e for e in gold_ner if e[1] == label}
            print_prf(len(p_l & g_l), len(p_l), len(g_l), label)

        print("\n[NEL]  (text, label, kb_id) match")

    nel_tp = len(pred_nel & gold_nel)
    if verbose:
        print_prf(nel_tp, len(pred_nel), len(gold_nel), "overall")
        for label in ("PER", "ORG", "LOC"):
            p_l = {e for e in pred_nel if e[1] == label}
            g_l = {e for e in gold_nel if e[1] == label}
            print_prf(len(p_l & g_l), len(p_l), len(g_l), label)

        missed_ner = gold_ner - pred_ner
        missed_nel = gold_nel - pred_nel
        print(f"\n[Missed NER]  {len(missed_ner)} entities:")
        for text, label in sorted(missed_ner):
            print(f"  {label:3s}  {text!r}")
        print(f"\n[Missed NEL]  {len(missed_nel)} entities (wrong/missing kb_id):")
        for text, label, kb_id in sorted(missed_nel):
            print(f"  {label:3s}  {text!r:40s}  expected={kb_id}")

    ner_p, ner_r, ner_f = prf(ner_tp, len(pred_ner), len(gold_ner))
    nel_p, nel_r, nel_f = prf(nel_tp, len(pred_nel), len(gold_nel))
    return dict(ner_p=ner_p, ner_r=ner_r, ner_f=ner_f,
                nel_p=nel_p, nel_r=nel_r, nel_f=nel_f,
                gold=len(gold_ner), pred=len(pred_ner))


def render_inline_entities(doc) -> str:
    if not doc.ents:
        return doc.text
    parts = []
    cursor = 0
    for ent in doc.ents:
        parts.append(doc.text[cursor:ent.start_char])
        kb_id = f"|{ent.kb_id_}" if ent.kb_id_ else ""
        parts.append(f"{ent.text}[{ent.label_}{kb_id}]")
        cursor = ent.end_char
    parts.append(doc.text[cursor:])
    return "".join(parts)


def resolve_paths(inputs: list[str]) -> list[Path]:
    """Expand globs and directories into a flat list of XML files."""
    result = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            result.extend(sorted(p.rglob("*.xml")))
        else:
            expanded = sorted(Path(p.parent).glob(p.name)) if "*" in str(p) or "?" in str(p) else [p]
            result.extend(expanded)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NER + NEL on Dodis TEI XML files using a spaCy model."
    )
    parser.add_argument(
        "xml", nargs="+",
        help="Path(s) to TEI XML file(s), directory, or glob pattern (e.g. data/test/*.xml)"
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL),
        help=f"Path to the spaCy model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--no-text", action="store_true",
        help="Skip printing the inline-annotated text per document"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print an aggregated summary across all files at the end"
    )
    args = parser.parse_args()

    xml_files = resolve_paths(args.xml)
    if not xml_files:
        parser.error("No XML files found for the given input(s).")

    print(f"Loading model: {args.model}")
    nlp = spacy.load(args.model)

    totals = dict(ner_tp=0, nel_tp=0, ner_pred=0, ner_gold=0, nel_pred=0, nel_gold=0)

    for xml_path in xml_files:
        print(f"\n{'='*60}")
        print(f"File: {xml_path}")
        print('='*60)

        gold = extract_gold_entities(xml_path)
        clean_text = load_clean_text(xml_path)
        doc = nlp(clean_text)

        if not args.no_text:
            print(render_inline_entities(doc))

        verbose = not args.summary or len(xml_files) == 1
        stats = evaluate(gold, doc, verbose=verbose)

        if args.summary:
            print(f"  NER  F1={stats['ner_f']:.3f}  NEL  F1={stats['nel_f']:.3f}"
                  f"  (gold={stats['gold']}, pred={stats['pred']})")

    if args.summary and len(xml_files) > 1:
        # Re-run silently to accumulate totals
        all_gold_ner: set = set()
        all_pred_ner: set = set()
        all_gold_nel: set = set()
        all_pred_nel: set = set()
        for xml_path in xml_files:
            gold = extract_gold_entities(xml_path)
            doc = nlp(load_clean_text(xml_path))
            all_gold_ner |= {(g.text, g.label) for g in gold}
            all_pred_ner |= {(e.text, e.label_) for e in doc.ents}
            all_gold_nel |= {(g.text, g.label, g.kb_id) for g in gold if g.kb_id}
            all_pred_nel |= {(e.text, e.label_, _normalise_ref(e.kb_id_)) for e in doc.ents if e.kb_id_}

        def prf(tp, pred, gold):
            p = tp / pred if pred else 0.0
            r = tp / gold if gold else 0.0
            f = 2*p*r/(p+r) if p+r else 0.0
            return p, r, f

        ner_tp = len(all_pred_ner & all_gold_ner)
        nel_tp = len(all_pred_nel & all_gold_nel)
        np_, nr, nf = prf(ner_tp, len(all_pred_ner), len(all_gold_ner))
        lp, lr, lf = prf(nel_tp, len(all_pred_nel), len(all_gold_nel))
        print(f"\n{'='*60}")
        print(f"AGGREGATE  ({len(xml_files)} files)")
        print(f"  NER  P={np_:.3f}  R={nr:.3f}  F1={nf:.3f}"
              f"  (tp={ner_tp}, pred={len(all_pred_ner)}, gold={len(all_gold_ner)})")
        print(f"  NEL  P={lp:.3f}  R={lr:.3f}  F1={lf:.3f}"
              f"  (tp={nel_tp}, pred={len(all_pred_nel)}, gold={len(all_gold_nel)})")


if __name__ == "__main__":
    main()
