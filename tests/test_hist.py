import json

from deepofi.iex import hist


def test_list_files_filters_feed(tmp_path):
    listing = {
        "20261013": [
            {"date": "20261013", "feed": "DEEP", "version": "1.0", "link": "a", "size": "10"},
            {"date": "20261013", "feed": "TOPS", "version": "1.6", "link": "b", "size": "5"},
        ],
        "20170101": {"date": "20170101", "feed": "TOPS", "version": "1.5", "link": "c", "size": "1"},
    }
    path = tmp_path / "hist.json"
    path.write_text(json.dumps(listing))

    files = hist.list_files(url=path.as_uri())

    assert list(files) == ["20261013"]
    assert files["20261013"]["size"] == 10


def test_download_skips_complete_file(tmp_path):
    entry = {"date": "20261013", "feed": "DEEP", "version": "1.0", "link": "unused", "size": 3}
    (tmp_path / "20261013_IEXTP1_DEEP1.0.pcap.gz").write_bytes(b"abc")

    assert hist.download(entry, dest_dir=tmp_path).read_bytes() == b"abc"


def test_download_resumes_partial(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdef")
    entry = {"date": "20261013", "feed": "DEEP", "version": "1.0", "link": source.as_uri(), "size": 6}
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / "20261013_IEXTP1_DEEP1.0.pcap.gz.part").write_bytes(b"abc")

    # file:// ignores Range, so the downloader must fall back to a full rewrite.
    assert hist.download(entry, dest_dir=dest).read_bytes() == b"abcdef"
