"""Charts for the README. Static PNGs, kept deliberately plain."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _fig():
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def profit_chart(bets, best_bets, stake, edge, path):
    fig, ax = _fig()
    ax.axhline(0, color=AXIS, linewidth=1)
    for b, colour, label in [(bets, BLUE, "Closing odds"), (best_bets, ORANGE, "Best price (optimistic)")]:
        if len(b):
            ax.plot(b["date"], b["cum_profit"], color=colour, linewidth=2, label=label)
            v = b['cum_profit'].iloc[-1]
            ax.annotate(f"{'−' if v < 0 else '+'}£{abs(v):,.0f}", (b["date"].iloc[-1], b["cum_profit"].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylabel("Cumulative profit (£)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)
    _style(ax, "Value-bet strategy: cumulative profit",
           f"£{stake:.0f} flat stake when model EV > {edge:.0%}, out-of-sample seasons only")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def calibration_chart(cal_model, cal_market, path):
    fig, ax = _fig()
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    ax.text(0.80, 0.83, "perfect calibration", rotation=29, fontsize=8.5, color=MUTED)
    for t, colour, label in [(cal_market, ORANGE, "Market"), (cal_model, BLUE, "Model")]:
        ax.plot(t["predicted"], t["observed"], color=colour, linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    _style(ax, "Calibration: when we say 30%, does it happen 30% of the time?",
           "All home/draw/away probabilities pooled into 10% buckets, test seasons")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def longshot_chart(fl, path):
    fig, ax = _fig()
    roi = fl["roi"] * 100
    lo, hi = fl["roi_ci_low"] * 100, fl["roi_ci_high"] * 100
    bars = ax.bar(fl.index, roi, color=BLUE, width=0.6)
    ax.errorbar(range(len(fl)), roi, yerr=[roi - lo, hi - roi], fmt="none", ecolor=INK2,
                elinewidth=1, capsize=3)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(range(len(fl)))
    ax.set_xticklabels([f"{b}\n{v:+.1f}%\n(n={int(n):,})" for b, v, n in zip(fl.index, roi, fl["outcomes"])])
    ax.margins(y=0.15)
    ax.set_xlabel("Decimal odds band (left = favourites, right = longshots); small n = noisy")
    ax.set_ylabel("Return per £1 bet (%)")
    _style(ax, "Return from backing every outcome, by odds band",
           "All matches, closing market-average odds. Whiskers: 95% interval (match-level bootstrap)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
