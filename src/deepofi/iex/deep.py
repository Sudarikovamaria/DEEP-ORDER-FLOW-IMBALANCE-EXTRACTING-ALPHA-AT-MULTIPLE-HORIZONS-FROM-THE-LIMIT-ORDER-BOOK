"""Streaming parser for IEX DEEP v1.0 captures (IEX-TP v1 transport).

Layouts follow the IEX DEEP Specification v1.08 and IEX Transport Specification v1.25.
HIST files are pcapng (IEX merges several captures with mergecap); classic pcap is also read.
All binary fields are little endian. Prices stay as raw int64 fixed-point (1e-4 dollars).
"""
import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numba
import numpy as np

DEEP_PROTOCOL_ID = 0x8004
PRICE_SCALE = 10_000

KIND_BID = 0             # '8' Price Level Update, buy side
KIND_ASK = 1             # '5' Price Level Update, sell side
KIND_TRADE = 2           # 'T' Trade Report
KIND_TRADE_BREAK = 3     # 'B' Trade Break
KIND_TRADING_STATUS = 4  # 'H' Trading Status, status byte in `flags`
KIND_SECURITY_EVENT = 5  # 'E' Security Event, event byte in `flags`
KIND_SYSTEM_EVENT = 6    # 'S' System Event, event byte in `flags`, symbol = 0
KIND_OPERATIONAL_HALT = 7  # 'O' Operational Halt Status, status byte in `flags`

# For price level updates `flags` is the event flag: 1 means the atomic book update is complete.
RECORD_DTYPE = np.dtype([
    ("ts", "<i8"),
    ("price", "<i8"),
    ("symbol", "<u8"),
    ("size", "<u4"),
    ("kind", "u1"),
    ("flags", "u1"),
])

_PCAP_HEADER_LEN = 24
_PCAP_MAGICS = (0xA1B2C3D4, 0xA1B23C4D)  # microsecond and nanosecond resolution
_PCAPNG_MAGIC = 0x0A0D0D0A               # Section Header Block type
_LINKTYPE_ETHERNET = 1
_LINKTYPE_RAW = (101, 228)

_STATUS_NEED_DATA = 0
_STATUS_OUT_FULL = 1
_STATUS_ERROR = 2

# Indices into the int64 `state` array shared with the numba scanners.
_ST_NEXT_SEQ = 0
_ST_PACKETS = 1
_ST_GAP_MESSAGES = 2
_ST_DUPLICATE_PACKETS = 3
_ST_OTHER_PACKETS = 4
_ST_LINKTYPE = 5


def encode_symbol(symbol):
    return int(np.frombuffer(symbol.ljust(8).encode("ascii"), dtype="<u8")[0])


def decode_symbol(value):
    return int(value).to_bytes(8, "little").decode("ascii").rstrip()


@dataclass
class ParseStats:
    message_counts: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))
    state: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))
    truncated_bytes: int = 0

    @property
    def packets(self):
        return int(self.state[_ST_PACKETS])

    @property
    def gap_messages(self):
        return int(self.state[_ST_GAP_MESSAGES])

    @property
    def duplicate_packets(self):
        return int(self.state[_ST_DUPLICATE_PACKETS])

    @property
    def other_packets(self):
        return int(self.state[_ST_OTHER_PACKETS])

    def counts_by_type(self):
        return {chr(code): int(count) for code, count in enumerate(self.message_counts) if count}


@numba.njit(cache=True)
def _u16(buf, i):
    return np.uint32(buf[i]) | (np.uint32(buf[i + 1]) << 8)


@numba.njit(cache=True)
def _u32(buf, i):
    return np.uint32(buf[i]) | (np.uint32(buf[i + 1]) << 8) | (np.uint32(buf[i + 2]) << 16) | (np.uint32(buf[i + 3]) << 24)


@numba.njit(cache=True)
def _u64(buf, i):
    value = np.uint64(0)
    for k in range(8):
        value |= np.uint64(buf[i + k]) << np.uint64(8 * k)
    return value


@numba.njit(cache=True)
def _process_packet(buf, packet, packet_end, out, n_out, symbols, counts, state):
    """Decode one captured frame. Returns (n_out, False) if `out` has no room for its messages."""
    ip = packet
    if state[_ST_LINKTYPE] == 1:
        ethertype = (np.uint32(buf[packet + 12]) << 8) | np.uint32(buf[packet + 13])
        ip = packet + 14
        if ethertype == 0x8100:
            ethertype = (np.uint32(buf[packet + 16]) << 8) | np.uint32(buf[packet + 17])
            ip = packet + 18
        if ethertype != 0x0800:
            state[_ST_OTHER_PACKETS] += 1
            return n_out, True

    if ip + 20 > packet_end or (buf[ip] >> 4) != 4 or buf[ip + 9] != 17:
        state[_ST_OTHER_PACKETS] += 1
        return n_out, True

    tp = ip + (buf[ip] & 0x0F) * 4 + 8
    if tp + 40 > packet_end or _u16(buf, tp + 2) != 0x8004:
        state[_ST_OTHER_PACKETS] += 1
        return n_out, True

    message_count = _u16(buf, tp + 14)
    if n_out + message_count > out.shape[0]:
        return n_out, False

    state[_ST_PACKETS] += 1
    first_seq = np.int64(_u64(buf, tp + 24))
    next_seq = state[_ST_NEXT_SEQ]
    skip = 0
    if next_seq > 0 and first_seq < next_seq:
        if message_count == 0 or first_seq + message_count <= next_seq:
            if message_count > 0:
                state[_ST_DUPLICATE_PACKETS] += 1
            return n_out, True
        skip = next_seq - first_seq
    if next_seq > 0 and first_seq > next_seq:
        state[_ST_GAP_MESSAGES] += first_seq - next_seq
    state[_ST_NEXT_SEQ] = max(next_seq, first_seq + message_count)

    m = tp + 40
    for index in range(message_count):
        if m + 2 > packet_end:
            break
        length = _u16(buf, m)
        data = m + 2
        m = data + length
        if length == 0 or m > packet_end:
            break
        if index < skip:
            continue

        message_type = buf[data]
        counts[message_type] += 1

        if message_type == 0x53:  # 'S' System Event
            if length < 10:
                continue
            out[n_out].kind = 6
            out[n_out].flags = buf[data + 1]
            out[n_out].ts = np.int64(_u64(buf, data + 2))
            out[n_out].symbol = 0
            out[n_out].price = 0
            out[n_out].size = 0
            n_out += 1
            continue

        if message_type == 0x38 or message_type == 0x35:
            kind, min_length = (0 if message_type == 0x38 else 1), 30
        elif message_type == 0x54:
            kind, min_length = 2, 38
        elif message_type == 0x42:
            kind, min_length = 3, 38
        elif message_type == 0x48:
            kind, min_length = 4, 22
        elif message_type == 0x45:
            kind, min_length = 5, 18
        elif message_type == 0x4F:
            kind, min_length = 7, 18
        else:
            continue
        if length < min_length:
            continue

        symbol = _u64(buf, data + 10)
        if symbols.shape[0] > 0:
            j = np.searchsorted(symbols, symbol)
            if j == symbols.shape[0] or symbols[j] != symbol:
                continue

        out[n_out].kind = kind
        out[n_out].flags = buf[data + 1]
        out[n_out].ts = np.int64(_u64(buf, data + 2))
        out[n_out].symbol = symbol
        if kind <= 3:
            out[n_out].size = _u32(buf, data + 18)
            out[n_out].price = np.int64(_u64(buf, data + 22))
        else:
            out[n_out].size = 0
            out[n_out].price = 0
        n_out += 1

    return n_out, True


@numba.njit(cache=True)
def _scan_pcap(buf, end, pos, out, n_out, symbols, counts, state):
    while pos + 16 <= end:
        incl_len = _u32(buf, pos + 8)
        if pos + 16 + incl_len > end:
            return pos, n_out, _STATUS_NEED_DATA
        n_out, done = _process_packet(buf, pos + 16, pos + 16 + incl_len, out, n_out, symbols, counts, state)
        if not done:
            return pos, n_out, _STATUS_OUT_FULL
        pos += 16 + incl_len
    return pos, n_out, _STATUS_NEED_DATA


@numba.njit(cache=True)
def _scan_pcapng(buf, end, pos, out, n_out, symbols, counts, state):
    while pos + 12 <= end:
        block_type = _u32(buf, pos)
        block_len = _u32(buf, pos + 4)
        if block_len < 12 or block_len % 4 != 0:
            return pos, n_out, _STATUS_ERROR
        if pos + block_len > end:
            return pos, n_out, _STATUS_NEED_DATA

        if block_type == 0x0A0D0D0A:  # Section Header Block: only little-endian sections
            if _u32(buf, pos + 8) != 0x1A2B3C4D:
                return pos, n_out, _STATUS_ERROR
        elif block_type == 1:  # Interface Description Block
            state[_ST_LINKTYPE] = _u16(buf, pos + 8)
        elif block_type == 6:  # Enhanced Packet Block
            captured = _u32(buf, pos + 20)
            n_out, done = _process_packet(buf, pos + 28, pos + 28 + captured, out, n_out, symbols, counts, state)
            if not done:
                return pos, n_out, _STATUS_OUT_FULL
        elif block_type == 3:  # Simple Packet Block
            captured = min(_u32(buf, pos + 8), block_len - 16)
            n_out, done = _process_packet(buf, pos + 12, pos + 12 + captured, out, n_out, symbols, counts, state)
            if not done:
                return pos, n_out, _STATUS_OUT_FULL
        pos += block_len
    return pos, n_out, _STATUS_NEED_DATA


def open_source(source):
    """Open a capture path (.gz detected by suffix or magic), or wrap an already opened binary stream."""
    if isinstance(source, (str, Path)):
        stream = open(source, "rb")
    else:
        stream = source
    head = stream.peek(2)[:2] if hasattr(stream, "peek") else b""
    if head == b"\x1f\x8b" or (isinstance(source, (str, Path)) and str(source).endswith(".gz")):
        return gzip.GzipFile(fileobj=stream, mode="rb")
    return stream


def iter_records(source, symbols=None, chunk_bytes=1 << 26, out_capacity=1 << 20, stats=None):
    """Yield arrays of RECORD_DTYPE in feed order from a pcap or pcapng capture.

    `symbols` restricts symbol messages to the given tickers (System Events are always kept).
    Pass a ParseStats to collect message counts by type, sequence gaps and duplicates.
    """
    stats = stats if stats is not None else ParseStats()
    symbol_filter = np.array(sorted(encode_symbol(s) for s in symbols or ()), dtype=np.uint64)
    stream = open_source(source)

    try:
        buf = np.empty(chunk_bytes + (1 << 17), dtype=np.uint8)
        view = memoryview(buf)
        end = stream.readinto(view[:_PCAP_HEADER_LEN])
        if end < _PCAP_HEADER_LEN:
            raise ValueError("File is too short to be a capture")
        magic = int.from_bytes(buf[:4].tobytes(), "little")

        if magic == _PCAPNG_MAGIC:
            scan = _scan_pcapng
        elif magic in _PCAP_MAGICS:
            scan = _scan_pcap
            stats.state[_ST_LINKTYPE] = int.from_bytes(buf[20:24].tobytes(), "little") & 0x0FFFFFFF
            end = 0
        else:
            raise ValueError(f"Unsupported capture magic {magic:#x} (expected little-endian pcap or pcapng)")

        out = np.empty(out_capacity, dtype=RECORD_DTYPE)
        n_out = 0
        eof = False

        while True:
            if not eof:
                read = stream.readinto(view[end:end + chunk_bytes])
                eof = read == 0
                end += read

            pos, n_out, status = scan(buf, end, 0, out, n_out, symbol_filter, stats.message_counts, stats.state)
            while status == _STATUS_OUT_FULL:
                yield out[:n_out].copy()
                n_out = 0
                pos, n_out, status = scan(buf, end, pos, out, n_out, symbol_filter, stats.message_counts, stats.state)
            if status == _STATUS_ERROR:
                raise ValueError("Malformed or big-endian pcapng block")

            linktype = stats.state[_ST_LINKTYPE]
            if linktype and linktype != _LINKTYPE_ETHERNET and linktype not in _LINKTYPE_RAW:
                raise ValueError(f"Unsupported link type {linktype}")

            buf[:end - pos] = buf[pos:end]
            end -= pos
            if eof:
                stats.truncated_bytes = end
                break

        if n_out:
            yield out[:n_out].copy()
    finally:
        stream.close()


def read_records(source, symbols=None, stats=None, **kwargs):
    chunks = list(iter_records(source, symbols=symbols, stats=stats, **kwargs))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=RECORD_DTYPE)
