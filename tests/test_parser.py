import gzip
import struct

import numpy as np
import pytest

from deepofi.iex.deep import (
    KIND_ASK,
    KIND_BID,
    KIND_SYSTEM_EVENT,
    KIND_TRADE,
    KIND_TRADING_STATUS,
    ParseStats,
    decode_symbol,
    encode_symbol,
    read_records,
)

# Segment from "Example Messages" in the IEX Transport Specification v1.25 (DEEP v1.0 payload).
SPEC_SEGMENT = bytes.fromhex(
    "01 00 04 80 01 00 00 00 00 00 87 42 48 00 02 00"
    "8c a6 21 00 00 00 00 00 ca c3 00 00 00 00 00 00"
    "ec 45 c2 20 96 86 6d 14"
    "26 00 54 00 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20"
    "64 00 00 00 24 1d 0f 00 00 00 00 00 96 8f 06 00 00 00 00 00"
    "1e 00 38 01 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20"
    "e4 25 00 00 24 1d 0f 00 00 00 00 00"
)
SPEC_TS = int.from_bytes(bytes.fromhex("ac 63 c0 20 96 86 6d 14"), "little")


def plu(side, flags, ts, symbol, size, price):
    return struct.pack("<cBq8sIq", b"8" if side == "B" else b"5", flags, ts, symbol.ljust(8).encode(), size, price)


def trade(ts, symbol, size, price, trade_id=1):
    return struct.pack("<cBq8sIqq", b"T", 0, ts, symbol.ljust(8).encode(), size, price, trade_id)


def trading_status(status, ts, symbol):
    return struct.pack("<cBq8s4s", b"H", ord(status), ts, symbol.ljust(8).encode(), b"    ")


def system_event(event, ts):
    return struct.pack("<cBq", b"S", ord(event), ts)


def segment(messages, first_seq, protocol=0x8004):
    payload = b"".join(struct.pack("<H", len(m)) + m for m in messages)
    header = struct.pack("<BBHIIHHqqq", 1, 0, protocol, 1, 1, len(payload), len(messages), 0, first_seq, 0)
    return header + payload


def packet(udp_payload, vlan=False):
    udp = struct.pack(">HHHH", 10378, 10378, 8 + len(udp_payload), 0) + udp_payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, bytes(4), bytes(4)) + udp
    ethernet = bytes(12) + (b"\x81\x00\x00\x01" if vlan else b"") + b"\x08\x00"
    return ethernet + ip


def pcap(packets):
    out = [struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)]
    for frame in packets:
        out.append(struct.pack("<IIII", 0, 0, len(frame), len(frame)) + frame)
    return b"".join(out)


def write(tmp_path, data, name="day.pcap"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_spec_example_segment(tmp_path):
    records = read_records(write(tmp_path, pcap([packet(SPEC_SEGMENT)])))

    assert records["kind"].tolist() == [KIND_TRADE, KIND_BID]
    assert records["ts"].tolist() == [SPEC_TS, SPEC_TS]
    assert records["price"].tolist() == [990500, 990500]  # $99.05
    assert records["size"].tolist() == [100, 9700]
    assert records["flags"][1] == 1
    assert decode_symbol(records["symbol"][0]) == "ZIEXT"


def test_symbol_roundtrip():
    assert decode_symbol(encode_symbol("AAPL")) == "AAPL"


def sample_packets(first_seq=1):
    return [
        packet(segment([system_event("R", 1)], first_seq=first_seq)),
        packet(segment([plu("B", 0, 2, "AAPL", 100, 1_500_000), plu("S", 1, 2, "AAPL", 200, 1_500_100)], first_seq=first_seq + 1)),
        packet(segment([plu("B", 1, 3, "MSFT", 300, 3_000_000), trade(4, "AAPL", 50, 1_500_100)], first_seq=first_seq + 3)),
        packet(segment([trading_status("H", 5, "AAPL")], first_seq=first_seq + 5)),
    ]


def test_symbol_filter_keeps_system_events(tmp_path):
    records = read_records(write(tmp_path, pcap(sample_packets())), symbols=["AAPL"])

    assert records["kind"].tolist() == [KIND_SYSTEM_EVENT, KIND_BID, KIND_ASK, KIND_TRADE, KIND_TRADING_STATUS]
    assert records["flags"][-1] == ord("H")
    assert {decode_symbol(s) for s in records["symbol"][1:]} == {"AAPL"}


def test_chunking_and_gzip_do_not_change_output(tmp_path):
    data = pcap([p for k in range(50) for p in sample_packets(first_seq=1 + 6 * k)])
    expected = read_records(write(tmp_path, data))
    with gzip.open(tmp_path / "day.pcap.gz", "wb") as f:
        f.write(data)

    small = read_records(write(tmp_path, data, "small.pcap"), chunk_bytes=97, out_capacity=3)
    gzipped = read_records(tmp_path / "day.pcap.gz", chunk_bytes=1000)

    assert len(expected) == 50 * 6
    np.testing.assert_array_equal(small, expected)
    np.testing.assert_array_equal(gzipped, expected)


def test_sequence_gaps_and_duplicates(tmp_path):
    packets = [
        packet(segment([plu("B", 1, 1, "AAPL", 1, 1)], first_seq=1)),
        packet(segment([plu("B", 1, 1, "AAPL", 1, 1)], first_seq=1)),  # duplicate
        packet(segment([plu("B", 1, 2, "AAPL", 2, 1), plu("B", 1, 3, "AAPL", 3, 1)], first_seq=2)),
        packet(segment([plu("B", 1, 3, "AAPL", 3, 1), plu("B", 1, 4, "AAPL", 4, 1)], first_seq=3)),  # half seen
        packet(segment([plu("B", 1, 9, "AAPL", 9, 1)], first_seq=9)),  # 4 messages missing
    ]
    stats = ParseStats()
    records = read_records(write(tmp_path, pcap(packets)), stats=stats)

    assert records["size"].tolist() == [1, 2, 3, 4, 9]
    assert stats.duplicate_packets == 1
    assert stats.gap_messages == 4


def test_unknown_grown_and_foreign_messages(tmp_path):
    grown = plu("B", 1, 1, "AAPL", 10, 1) + b"\x00" * 6
    packets = [
        packet(segment([b"Zxyz", grown], first_seq=1), vlan=True),
        packet(segment([plu("B", 1, 2, "AAPL", 20, 1)], first_seq=1, protocol=0x8003)),
    ]
    stats = ParseStats()
    records = read_records(write(tmp_path, pcap(packets)), stats=stats)

    assert records["size"].tolist() == [10]
    assert stats.counts_by_type() == {"Z": 1, "8": 1}
    assert stats.other_packets == 1


def test_rejects_non_pcap(tmp_path):
    with pytest.raises(ValueError):
        read_records(write(tmp_path, b"\x00" * 64))
