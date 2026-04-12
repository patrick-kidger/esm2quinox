import json
from unittest.mock import patch

import esm2quinox
from esm2quinox._blast import (
    _fetch_sequences,
    _parse_hits,
    _poll_blast,
    _submit_blast,
)


_MOCK_SUBMIT_RESPONSE = """
<!DOCTYPE html>
<html><body>
    RID = ABC12345
    RTOE = 10
</body></html>
"""

_MOCK_WAITING_RESPONSE = """
    Status=WAITING
"""

_MOCK_JSON_RESULT = {
    "BlastOutput2": {
        "report": {
            "program": "blastp",
            "results": {
                "search": {
                    "query_title": "Query",
                    "hits": [
                        {
                            "description": [
                                {
                                    "accession": "P12345",
                                    "title": "Some protein",
                                }
                            ],
                            "hsps": [
                                {
                                    "evalue": 1e-50,
                                    "identity": 90,
                                    "align_len": 100,
                                    "hseq": "M" * 100,
                                }
                            ],
                        },
                        {
                            "description": [
                                {
                                    "accession": "Q67890",
                                    "title": "Another protein",
                                }
                            ],
                            "hsps": [
                                {
                                    "evalue": 1e-10,
                                    "identity": 45,
                                    "align_len": 100,
                                    "hseq": "A" * 100,
                                }
                            ],
                        },
                    ],
                }
            },
        }
    }
}

_MOCK_FASTA_RESPONSE = """\
>P12345.1 Some protein
MKWVTFISLLFLFSSAYSRGVFRR
DTHKSEIAHRFKDLGEEHFK
>Q67890.2 Another protein
MAEGEITTFTALTEKFNLPPGNYKKPK
"""


def test_submit_blast():
    with patch("esm2quinox._blast._request", return_value=_MOCK_SUBMIT_RESPONSE):
        rid, rtoe = _submit_blast("MKWVTFIS", "blastp", "swissprot", 10.0, 50)
    assert rid == "ABC12345"
    assert rtoe == 10


def test_parse_hits():
    parsed = _parse_hits(_MOCK_JSON_RESULT)
    assert len(parsed) == 2
    acc0, title0, ev0, idf0 = parsed[0]
    assert acc0 == "P12345"
    assert title0 == "Some protein"
    assert ev0 == 1e-50
    assert abs(idf0 - 0.9) < 1e-9
    acc1, _, _, _ = parsed[1]
    assert acc1 == "Q67890"


def test_parse_hits_list_format():
    # JSON2 (non-_S) wraps BlastOutput2 in a list.
    result = {"BlastOutput2": [_MOCK_JSON_RESULT["BlastOutput2"]]}
    parsed = _parse_hits(result)
    assert len(parsed) == 2


def test_fetch_sequences():
    with patch("esm2quinox._blast._request", return_value=_MOCK_FASTA_RESPONSE):
        seqs = _fetch_sequences(["P12345", "Q67890"])
    assert "P12345" in seqs
    assert "Q67890" in seqs
    assert seqs["P12345"] == "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFK"
    assert seqs["Q67890"] == "MAEGEITTFTALTEKFNLPPGNYKKPK"


def test_poll_blast():
    call_count = 0

    def mock_request(url, data):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _MOCK_WAITING_RESPONSE
        return json.dumps(_MOCK_JSON_RESULT)

    with patch("esm2quinox._blast._request", side_effect=mock_request):
        with patch("time.sleep"):
            result = _poll_blast("ABC12345")
    assert "BlastOutput2" in result
    assert call_count == 2


def test_blast_end_to_end():
    submit_response = _MOCK_SUBMIT_RESPONSE
    json_response = json.dumps(_MOCK_JSON_RESULT)
    fasta_response = _MOCK_FASTA_RESPONSE

    call_count = 0

    def mock_request(url, data):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return submit_response
        if call_count == 2:
            return json_response
        return fasta_response

    with patch("esm2quinox._blast._request", side_effect=mock_request):
        with patch("time.sleep"):
            hits = esm2quinox.blast("MKWVTFIS")
    assert len(hits) == 2
    assert hits[0].accession == "P12345"
    assert hits[0].sequence == "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFK"
    assert hits[0].evalue == 1e-50
    assert abs(hits[0].identity_fraction - 0.9) < 1e-9
    assert hits[1].accession == "Q67890"

    # Check they can be tokenised.
    sequences = [hit.sequence for hit in hits]
    tokens = esm2quinox.tokenise(sequences)
    assert tokens.shape[0] == 2
