from __future__ import annotations

from csc869.typst_report import _figure, _stat, _t, _table


def test_escape_typst_markup():
    assert _t("a_b") == "a\\_b"
    assert _t("x > y") == "x \\> y"
    assert _t("results/cv_tidy.csv") == "results/cv\\_tidy.csv"


def test_table_is_booktabs_style():
    out = _table(["Model", "F1"], [["Tree", "0.6615"]], size="8pt")
    assert "table.header" in out
    assert "table.hline" in out
    assert "[Tree]" in out
    assert "[0.6615]" in out
    assert "size: 8pt" in out
    # No colour theme: the table must not introduce fills or rgb() values.
    assert "fill:" not in out
    assert "rgb(" not in out


def test_table_highlight_is_bold_not_coloured():
    out = _table(["Model", "F1"], [["Tree", "0.6615"]], highlight={0})
    assert "[*Tree*]" in out
    assert "[*0.6615*]" in out
    assert "fill:" not in out


def test_figure_and_stat_are_code_expressions():
    figure = _figure("roc.png", "figures", "ROC curves")
    assert figure.startswith("figure(")
    assert 'image("figures/roc.png"' in figure
    stat = _stat("75", "CV fits")
    assert stat.startswith("block(")
    assert "rgb(" not in stat
