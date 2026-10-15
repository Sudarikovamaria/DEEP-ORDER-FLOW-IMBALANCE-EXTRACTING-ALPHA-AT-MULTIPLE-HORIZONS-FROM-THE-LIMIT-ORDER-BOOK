import numpy as np
import pandas as pd
import pytest

from deepofi.data_process.features import limit_order_book, order_flow, order_flow_imbalance, side_order_flow


def level1(px, sz):
    return np.array(px, dtype=float)[:, None], np.array(sz, dtype=float)[:, None]


@pytest.mark.parametrize("variant, expected", [("paper", [0, 50, 30, -80]), ("cont", [0, 50, 30, -30])])
def test_bid_order_flow_handoff_example(variant, expected):
    # HANDOFF §5 table: same price, price up, price down.
    px, sz = level1([10.00, 10.00, 10.01, 10.00], [100, 150, 30, 80])
    assert side_order_flow(px, sz, "bid", variant)[:, 0].tolist() == expected


@pytest.mark.parametrize("variant, expected", [("paper", [0, 50, 30, -80]), ("cont", [0, 50, 30, -30])])
def test_ask_order_flow_mirror_example(variant, expected):
    # Mirror of the bid case: same price, ask down (improves), ask up (moves away).
    px, sz = level1([10.02, 10.02, 10.01, 10.02], [100, 150, 30, 80])
    assert side_order_flow(px, sz, "ask", variant)[:, 0].tolist() == expected


def test_empty_levels():
    px, sz = level1([np.nan, 10.00, np.nan, np.nan], [0, 70, 0, 0])
    assert side_order_flow(px, sz, "bid", "paper")[:, 0].tolist() == [0, 70, 0, 0]
    assert side_order_flow(px, sz, "bid", "cont")[:, 0].tolist() == [0, 70, -70, 0]
    assert side_order_flow(px, sz, "ask", "cont")[:, 0].tolist() == [0, 70, -70, 0]


def book_frame(bid_px, bid_sz, ask_px, ask_sz, levels=2):
    columns = {}
    for side, px, sz in (("bid", bid_px, bid_sz), ("ask", ask_px, ask_sz)):
        for i in range(levels):
            columns[f"{side}_px_{i + 1}"] = [row[i] for row in px]
            columns[f"{side}_sz_{i + 1}"] = [row[i] for row in sz]
    return pd.DataFrame(columns)


def test_of_ofi_and_lob_layout():
    book = book_frame(
        bid_px=[[10.00, 9.99], [10.00, 9.99]], bid_sz=[[100, 200], [120, 200]],
        ask_px=[[10.01, np.nan], [10.01, 10.02]], ask_sz=[[300, 0], [250, 40]],
    )
    of = order_flow(book, levels=2)
    assert of.shape == (2, 4)
    assert of[1].tolist() == [20, 0, -50, 40]  # bOF_1, bOF_2, aOF_1, aOF_2
    assert order_flow_imbalance(of)[1].tolist() == [70, -40]

    lob = limit_order_book(book, levels=2)
    assert lob.shape == (2, 8)
    np.testing.assert_allclose(lob[0], [10.01, 300, 10.00, 100, 10.01, 0, 9.99, 200], rtol=1e-6)
    assert not np.isnan(lob).any()
