import re
import time
import json
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_USER_AGENT = "esm2quinox"


@dataclass(frozen=True)
class BlastHit:
    """A single hit from a BLAST search.

    **Attributes:**

    - `accession`: the accession number of the hit.
    - `description`: the description line of the hit.
    - `sequence`: the full protein sequence (fetched via NCBI Entrez).
    - `evalue`: the e-value of the best HSP.
    - `identity_fraction`: the fraction of identical residues in the best HSP.
    """

    accession: str
    description: str
    sequence: str
    evalue: float
    identity_fraction: float


def _request(url: str, data: dict[str, str]) -> str:
    encoded = urlencode(data).encode()
    req = Request(url, encoded, {"User-Agent": _USER_AGENT})
    with urlopen(req) as resp:
        return resp.read().decode()


def _submit_blast(
    sequence: str,
    program: str,
    database: str,
    evalue: float,
    hitlist_size: int,
) -> tuple[str, int]:
    params = {
        "CMD": "Put",
        "QUERY": sequence,
        "PROGRAM": program,
        "DATABASE": database,
        "EXPECT": str(evalue),
        "HITLIST_SIZE": str(hitlist_size),
    }
    text = _request(_BLAST_URL, params)
    rid_match = re.search(r"RID = (\S+)", text)
    rtoe_match = re.search(r"RTOE = (\d+)", text)
    if rid_match is None:
        raise RuntimeError(f"Failed to parse RID from BLAST response:\n{text[:500]}")
    rid = rid_match.group(1)
    rtoe = int(rtoe_match.group(1)) if rtoe_match is not None else 30
    return rid, rtoe


def _poll_blast(rid: str) -> dict:
    params = {
        "CMD": "Get",
        "RID": rid,
        "FORMAT_TYPE": "JSON2_S",
    }
    delay = 20
    start = time.monotonic()
    while True:
        time.sleep(delay)
        delay = min(delay + 10, 60)
        text = _request(_BLAST_URL, params)
        if "Status=WAITING" in text:
            if time.monotonic() - start > 600:
                raise TimeoutError(
                    f"BLAST search {rid} has been running for over 10 minutes."
                )
            continue
        if "Status=FAILED" in text:
            raise RuntimeError(f"BLAST search {rid} failed.")
        if "Status=UNKNOWN" in text:
            raise RuntimeError(f"BLAST search {rid} expired or does not exist.")
        # If we get here, the result should be JSON.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Might be HTML with Status=READY; try one more time to get JSON.
            if "Status=READY" in text:
                continue
            raise RuntimeError(
                f"Unexpected BLAST response:\n{text[:500]}"
            )


def _parse_hits(result: dict) -> list[tuple[str, str, float, float]]:
    # JSON2_S structure:
    # {"BlastOutput2": {"report": {"results": {"search": {"hits": [...]}}}}}
    # or sometimes: {"BlastOutput2": [{"report": ...}]}
    root = result["BlastOutput2"]
    if isinstance(root, list):
        report = root[0]["report"]
    else:
        report = root["report"]
    hits_list = report["results"]["search"]["hits"]
    parsed = []
    for hit in hits_list:
        desc = hit["description"][0]
        accession = desc["accession"]
        title = desc.get("title", "")
        best_hsp = hit["hsps"][0]
        evalue = best_hsp["evalue"]
        identity = best_hsp["identity"]
        align_len = best_hsp["align_len"]
        identity_fraction = identity / align_len if align_len > 0 else 0.0
        parsed.append((accession, title, evalue, identity_fraction))
    return parsed


def _fetch_sequences(accessions: list[str]) -> dict[str, str]:
    # Batch fetch via NCBI Entrez efetch, in chunks of 200.
    sequences: dict[str, str] = {}
    for i in range(0, len(accessions), 200):
        chunk = accessions[i : i + 200]
        params = {
            "db": "protein",
            "id": ",".join(chunk),
            "rettype": "fasta",
            "retmode": "text",
        }
        text = _request(_EFETCH_URL, params)
        current_acc = None
        current_seq: list[str] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if current_acc is not None:
                    sequences[current_acc] = "".join(current_seq)
                # Parse accession from FASTA header. Formats vary:
                # >ACC.VERSION description
                # >gi|...|ref|ACC.VERSION| description
                header = line[1:].split()[0]
                # Take the last pipe-delimited field, strip version.
                parts = header.split("|")
                raw_acc = parts[-1] if len(parts) > 1 else parts[0]
                current_acc = raw_acc.split(".")[0]
                current_seq = []
            elif current_acc is not None:
                current_seq.append(line.strip())
        if current_acc is not None:
            sequences[current_acc] = "".join(current_seq)
    return sequences


def blast(
    sequence: str,
    *,
    program: str = "blastp",
    database: str = "swissprot",
    evalue: float = 10.0,
    hitlist_size: int = 50,
) -> list[BlastHit]:
    """Searches for related protein sequences via the NCBI BLAST API.

    Submits a BLAST search, polls for results, then fetches the full sequences
    of each hit via NCBI Entrez. This can take several minutes depending on
    NCBI server load.

    **Arguments:**

    - `sequence`: the query protein sequence as a string of amino acid letters
        (e.g. `"MKWVTFISLLFLFSSAYS..."`).
    - `program`: the BLAST program to use. Defaults to `"blastp"`.
    - `database`: the BLAST database to search. Defaults to `"swissprot"`.
        Other options include `"nr"` (non-redundant, much larger and slower).
    - `evalue`: the e-value threshold. Defaults to `10.0`.
    - `hitlist_size`: the maximum number of hits to return. Defaults to `50`.

    **Returns:**

    A list of `BlastHit` objects, each containing the accession, description,
    full protein sequence, e-value, and identity fraction.

    !!! warning

        This function makes network requests to the public NCBI BLAST servers.
        Searches typically take 30 seconds to several minutes. Please be
        respectful of NCBI's usage guidelines.

    !!! example

        ```python
        import esm2quinox

        hits = esm2quinox.blast("MKWVTFISLLFLFSSAYS")
        sequences = [hit.sequence for hit in hits]
        tokens = esm2quinox.tokenise(sequences)
        ```
    """
    rid, rtoe = _submit_blast(sequence, program, database, evalue, hitlist_size)
    # Wait for the estimated time before first poll.
    time.sleep(min(rtoe, 30))
    result = _poll_blast(rid)
    parsed = _parse_hits(result)
    if len(parsed) == 0:
        return []
    accessions = [acc for acc, _, _, _ in parsed]
    sequences = _fetch_sequences(accessions)
    hits = []
    for accession, description, ev, identity_fraction in parsed:
        seq = sequences.get(accession, "")
        if len(seq) == 0:
            continue
        hits.append(
            BlastHit(
                accession=accession,
                description=description,
                sequence=seq,
                evalue=ev,
                identity_fraction=identity_fraction,
            )
        )
    return hits
