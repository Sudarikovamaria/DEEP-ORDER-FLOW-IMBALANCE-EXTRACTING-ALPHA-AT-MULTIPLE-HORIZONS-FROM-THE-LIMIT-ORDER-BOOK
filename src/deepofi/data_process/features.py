"""OF, LOB and OFI inputs (Section 2.1 of the paper, HANDOFF §5).

Level i is the i-th best price on a side, not a fixed price. An empty level (NaN price, size 0)
compares as -inf on the bid side and +inf on the ask side, so a level appearing counts as a price
improvement and a level disappearing as a price worsening. The first row of a day has no
predecessor and gets zero order flow.
"""
import numpy as np

LEVELS = 10


def book_arrays(book, levels=LEVELS):
    def columns(side, field):
        return book[[f"{side}_{field}_{i}" for i in range(1, levels + 1)]].to_numpy(dtype=np.float64)

    return columns("bid", "px"), columns("bid", "sz"), columns("ask", "px"), columns("ask", "sz")


def side_order_flow(px, sz, side, variant="paper"):
    """Formulas (2) and (3) of the paper. variant="cont" uses -v(t-1) instead of -v(t) when the
    level moves away, as in Cont, Kukanov, Stoikov (2014)."""
    if variant not in ("paper", "cont"):
        raise ValueError(f"Unknown order flow variant {variant!r}")
    empty = -np.inf if side == "bid" else np.inf
    px = np.where(np.isnan(px), empty, px)
    prev_px, prev_sz = px[:-1], sz[:-1]
    cur_px, cur_sz = px[1:], sz[1:]

    # Both formulas reduce to: a better price adds the new queue, a worse price removes a queue.
    improved = cur_px > prev_px if side == "bid" else cur_px < prev_px
    worsened = cur_px < prev_px if side == "bid" else cur_px > prev_px
    departed = cur_sz if variant == "paper" else prev_sz

    flow = np.where(improved, cur_sz, cur_sz - prev_sz)
    flow = np.where(worsened, -departed, flow)
    return np.vstack([np.zeros((1, px.shape[1])), flow])


def order_flow(book, variant="paper", levels=LEVELS):
    """OF = [bOF_1..bOF_L, aOF_1..aOF_L], shape T x 2L."""
    bid_px, bid_sz, ask_px, ask_sz = book_arrays(book, levels)
    bid = side_order_flow(bid_px, bid_sz, "bid", variant)
    ask = side_order_flow(ask_px, ask_sz, "ask", variant)
    return np.hstack([bid, ask]).astype(np.float32)


def order_flow_imbalance(of):
    levels = of.shape[1] // 2
    return (of[:, :levels] - of[:, levels:]).astype(np.float32)


def limit_order_book(book, levels=LEVELS):
    """LOB states in the order of formula (1): a_1, va_1, b_1, vb_1, ..., shape T x 4L.

    Empty levels take the price of the deepest non-empty level on the same side (size stays 0);
    a fully empty side takes its last known prices.
    """
    bid_px, bid_sz, ask_px, ask_sz = book_arrays(book, levels)
    filled = []
    for px in (ask_px, bid_px):
        px = px.copy()
        for i in range(1, levels):
            px[:, i] = np.where(np.isnan(px[:, i]), px[:, i - 1], px[:, i])
        rows = np.where(~np.isnan(px[:, 0]), np.arange(len(px)), 0)
        np.maximum.accumulate(rows, out=rows)
        px = px[rows]
        filled.append(np.nan_to_num(px, nan=0.0))
    ask_px, bid_px = filled

    lob = np.empty((len(bid_px), 4 * levels))
    lob[:, 0::4], lob[:, 1::4], lob[:, 2::4], lob[:, 3::4] = ask_px, ask_sz, bid_px, bid_sz
    return lob.astype(np.float32)
