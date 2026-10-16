import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="NPN1石 アクティブLPF設計", layout="wide")

st.title("NPNトランジスタ1石 アクティブ ローパスフィルタ設計")
st.markdown(
    "エミッタフォロワ(ゲイン≒1)を使ったアクティブLPFです。"
    "次数を **1次 / 2次 / 3次** から選べます。"
    "2次はSallen-Key型、3次は **低インピーダンスRC(1次) → Sallen-Key(2次)** のカスケード構成です。"
)

VT = 0.02585  # 熱電圧 [V] (常温)
E24 = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
]


# ---------------------------------------------------------------- ユーティリティ
def nearest_e24(x):
    if x is None or x <= 0:
        return x
    dec = math.floor(math.log10(x))
    best = None
    for d in (dec - 1, dec, dec + 1):
        for m in E24:
            v = m * (10 ** d)
            if best is None or abs(math.log(v / x)) < abs(math.log(best / x)):
                best = v
    return best


def fmt_r(r):
    if r is None:
        return "—"
    if r >= 1e6:
        return f"{r / 1e6:.3g} MΩ"
    if r >= 1e3:
        return f"{r / 1e3:.3g} kΩ"
    return f"{r:.3g} Ω"


def fmt_c(c):
    if c is None:
        return "—"
    if c >= 1e-6:
        return f"{c * 1e6:.3g} µF"
    if c >= 1e-9:
        return f"{c * 1e9:.3g} nF"
    return f"{c * 1e12:.3g} pF"


def fmt_spice(val):
    if val is None:
        return "0"
    if val >= 1e6:
        return f"{val / 1e6:.4g}Meg"
    if val >= 1e3:
        return f"{val / 1e3:.4g}k"
    if val >= 1:
        return f"{val:.4g}"
    if val >= 1e-3:
        return f"{val * 1e3:.4g}m"
    if val >= 1e-6:
        return f"{val * 1e6:.4g}u"
    if val >= 1e-9:
        return f"{val * 1e9:.4g}n"
    return f"{val * 1e12:.4g}p"


# ---------------------------------------------------------------- フィルタ設計
def design_sections(kind, fc, ripple, order):
    """フィルタ特性から各セクションの (f0, Q) を返す。Q=None は1次セクション。
    3次の戻り値順序: [(2次SKのf0,Q), (1次RCのf0,None)]"""
    if kind == "バターワース":
        if order == 1:
            return [(fc, None)]
        if order == 2:
            return [(fc, 1 / math.sqrt(2))]
        return [(fc, 1.0), (fc, None)]

    if kind == "ベッセル":
        if order == 1:
            return [(fc, None)]
        if order == 2:
            q = 1 / math.sqrt(3)
            k = 1 / q**2 - 2
            u = (-k + math.sqrt(k * k + 4)) / 2
            return [(fc / math.sqrt(u), q)]
        # 反転ベッセル θ3(s)=s^3+6s^2+15s+15 の根から -3dB正規化
        return [(1.4526 * fc, 0.6910), (1.3268 * fc, None)]

    # チェビシェフ
    eps = math.sqrt(10 ** (ripple / 10) - 1)
    a = math.asinh(1 / eps) / order
    if order == 1:
        return [(fc * math.sinh(a), None)]
    if order == 2:
        sigma = math.sinh(a) * math.sin(math.pi / 4)
        omega = math.cosh(a) * math.cos(math.pi / 4)
        w0n = math.hypot(sigma, omega)
        return [(fc * w0n, w0n / (2 * sigma))]
    sigma = math.sinh(a) * math.sin(math.pi / 6)
    omega = math.cosh(a) * math.cos(math.pi / 6)
    w0n = math.hypot(sigma, omega)
    Q = w0n / (2 * sigma)
    f0_1st = fc * math.sinh(a)
    return [(fc * w0n, Q), (f0_1st, None)]


def ideal_H(f, sections, kind, ripple, order):
    H = np.ones_like(f, dtype=complex)
    for f0, Q in sections:
        x = f / f0
        if Q is None:
            H *= 1 / (1 + 1j * x)
        else:
            H *= 1 / (1 - x**2 + 1j * x / Q)
    # 偶数次チェビシェフの基準化（ピークを0dB、DCを -ripple [dB] に合わせる場合）
    # ※Sallen-Key実回路はDCゲイン≒1(0dB)でピークが持ち上がる挙動をするため、
    # 実回路との比較しやすさを優先してDC=0dB基準のままにしています。
    return H


# ---------------------------------------------------------------- 実回路(節点解析)
def actual_H_order1(f, r1, c2, r_e_ac, re_, rb_par, r_in, rl, cin=None, cout=None):
    s = 2j * np.pi * f
    # 出力負荷インピーダンス (Cout考慮)
    z_load = rl + (1 / (s * cout) if cout is not None else 0)
    z_e = 1 / (1 / r_e_ac + 1 / z_load)
    k = z_e / (z_e + re_)
    rb = 1 / (1 / rb_par + 1 / r_in)

    # 入力側インピーダンス (Cin考慮)
    z_r1 = r1 + (1 / (s * cin) if cin is not None else 0)

    # ベース電圧 Vb / Vin
    h_vb = 1 / (1 + z_r1 * (1 / rb + s * c2))
    # エミッタ電圧 Ve = k * Vb, Vout = Ve * (rl / z_load)
    h_out = rl / z_load if cout is not None else 1.0
    return h_vb * k * h_out


def actual_H_order2(f, r1, r2, c1, c2, r_e_ac, re_, rb_par, r_in, rl, cin=None, cout=None):
    s = 2j * np.pi * f
    z_load = rl + (1 / (s * cout) if cout is not None else 0)
    z_e = 1 / (1 / r_e_ac + 1 / z_load)
    k = z_e / (z_e + re_)
    rb = 1 / (1 / rb_par + 1 / r_in)

    z_r1 = r1 + (1 / (s * cin) if cin is not None else 0)

    a = 1 + r2 / rb + s * r2 * c2
    vin_over_vb = a + z_r1 * ((a - 1) / r2 + s * c1 * (a - k))
    h_out = rl / z_load if cout is not None else 1.0
    return (k / vin_over_vb) * h_out


def actual_H_order3(f, r1, r2, c1, c2, r3, c3, r_e_ac, re_, rb_par, r_in, rl, cin=None, cout=None):
    """3次: RC(1次) → Sallen-Key(2次) → エミッタフォロワ"""
    s = 2j * np.pi * f
    z_load = rl + (1 / (s * cout) if cout is not None else 0)
    z_e = 1 / (1 / r_e_ac + 1 / z_load)
    k = z_e / (z_e + re_)
    rb = 1 / (1 / rb_par + 1 / r_in)

    z_r3 = r3 + (1 / (s * cin) if cin is not None else 0)

    d = z_r3 / r1
    a = 1 + r2 / rb + s * r2 * c2
    c = a + r1 * ((a - 1) / r2 + s * c1 * (a - k))
    b = 1 + d + s * z_r3 * c3
    vin_over_vb = c * b - a * d
    h_out = rl / z_load if cout is not None else 1.0
    return (k / vin_over_vb) * h_out


# ---------------------------------------------------------------- 回路図SVG
def circuit_svg(p, order):
    els = []

    def line(x1, y1, x2, y2, w=2):
        els.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="{w}"/>')

    def dot(x, y):
        els.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="#222" stroke="none"/>')

    def text(x, y, s, anchor="middle", size=13, bold=False):
        wt = "bold" if bold else "normal"
        els.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{wt}" fill="#222" stroke="none">{s}</text>'
        )

    def res_h(x1, x2, y):
        els.append(f'<rect x="{x1}" y="{y - 9}" width="{x2 - x1}" height="18" fill="#fff"/>')

    def res_v(x, y1, y2):
        els.append(f'<rect x="{x - 9}" y="{y1}" width="18" height="{y2 - y1}" fill="#fff"/>')

    def cap_h(x, y):
        line(x - 5, y - 15, x - 5, y + 15, 3)
        line(x + 5, y - 15, x + 5, y + 15, 3)

    def cap_v(x, y):
        line(x - 15, y - 5, x + 15, y - 5, 3)
        line(x - 15, y + 5, x + 15, y + 5, 3)

    def ground(x, y):
        line(x, y, x, y + 12)
        line(x - 15, y + 12, x + 15, y + 12)
        line(x - 9, y + 18, x + 9, y + 18)
        line(x - 3, y + 24, x + 3, y + 24)

    def vcc(x, y):
        line(x - 14, y, x + 14, y)
        text(x, y - 8, "+Vcc")

    W = 860 if order == 3 else 760

    # --- 入力段 ---
    els.append('<circle cx="25" cy="200" r="4" fill="#fff"/>')
    text(25, 185, "Vin")
    line(29, 200, 80, 200)
    cap_h(85, 200)
    line(90, 200, 130, 200)
    text(85, 175, "Cin")
    text(85, 228, p["Cin"])

    if order == 3:
        # --- 1次RC (低インピーダンス) ---
        res_h(130, 190, 200)
        line(130, 200, 190, 200)
        els.append('<rect x="130" y="191" width="60" height="18" fill="none"/>')
        text(160, 180, "R3")
        text(160, 226, p["R3"])

        line(190, 200, 220, 200)
        dot(220, 200)
        # C3: ノードX → GND
        line(220, 200, 220, 265)
        cap_v(220, 270)
        line(220, 275, 220, 340)
        text(203, 268, "C3", anchor="end", size=14, bold=True)
        text(203, 285, p["C3"], anchor="end", size=14, bold=True)

        # R1 (SK入力)
        line(220, 200, 250, 200)
        res_h(250, 310, 200)
        line(250, 200, 310, 200)
        els.append('<rect x="250" y="191" width="60" height="18" fill="none"/>')
        text(280, 180, "R1")
        text(280, 226, p["R1"])

        # ノードA
        line(310, 200, 370, 200)
        dot(350, 200)

        # R2
        res_h(370, 430, 200)
        line(370, 200, 430, 200)
        els.append('<rect x="370" y="191" width="60" height="18" fill="none"/>')
        text(400, 180, "R2")
        text(400, 226, p["R2"])

        # ノードB + C2
        line(430, 200, 575, 200)
        dot(465, 200)
        line(465, 200, 465, 265)
        cap_v(465, 270)
        line(465, 275, 465, 340)
        text(448, 268, "C2", anchor="end", size=14, bold=True)
        text(448, 285, p["C2"], anchor="end", size=14, bold=True)

        # バイアス分圧
        dot(515, 200)
        vcc(515, 110)
        line(515, 110, 515, 125)
        res_v(515, 125, 165)
        els.append('<rect x="506" y="125" width="18" height="40" fill="none"/>')
        line(515, 165, 515, 200)
        text(533, 142, "RB1", anchor="start")
        text(533, 158, p["RB"], anchor="start")
        line(515, 200, 515, 235)
        res_v(515, 235, 285)
        els.append('<rect x="506" y="235" width="18" height="50" fill="none"/>')
        line(515, 285, 515, 340)
        text(533, 258, "RB2", anchor="start")
        text(533, 274, p["RB"], anchor="start")

        # C1 正帰還 (ノードA → 出力)
        line(350, 200, 350, 55)
        line(350, 55, 500, 55)
        cap_h(505, 55)
        line(510, 55, 660, 55)
        line(660, 55, 660, 260)
        dot(660, 260)
        text(505, 30, f"C1 = {p['C1']}", size=14, bold=True)

        # トランジスタ
        els.append('<circle cx="602" cy="200" r="40" fill="none"/>')
        line(575, 175, 575, 225, 4)
        line(575, 190, 620, 165)
        line(620, 165, 620, 110)
        vcc(620, 110)
        line(575, 210, 620, 235)
        els.append('<polygon points="613.3,231.3 602.3,230.3 606.7,222.5" fill="#222"/>')
        line(620, 235, 620, 285)
        text(672, 205, "Q1", anchor="start", size=14, bold=True)

        # エミッタ抵抗
        dot(620, 260)
        res_v(620, 285, 335)
        els.append('<rect x="611" y="285" width="18" height="50" fill="none"/>')
        line(620, 335, 620, 340)
        text(638, 307, "RE", anchor="start")
        text(638, 323, p["RE"], anchor="start")

        # 出力 (エミッタフォロワ直結 = 低インピーダンス)
        line(620, 260, 700, 260)
        cap_h(705, 260)
        line(710, 260, 820, 260)
        dot(800, 260)
        line(800, 260, 828, 260)
        els.append('<circle cx="832" cy="260" r="4" fill="#fff"/>')
        text(832, 245, "Vout")
        text(705, 238, "Cout")
        text(705, 300, p["Cout"])
        line(800, 260, 800, 285)
        res_v(800, 285, 335)
        els.append('<rect x="791" y="285" width="18" height="50" fill="none"/>')
        line(800, 335, 800, 340)
        text(818, 307, "RL", anchor="start")
        text(818, 323, p["RL"], anchor="start")

        # GNDレール
        line(220, 340, 800, 340)
        ground(550, 340)

    else:
        # 1次 / 2次
        res_h(130, 190, 200)
        els.append('<rect x="130" y="191" width="60" height="18" fill="none"/>')
        line(130, 200, 190, 200)
        text(160, 180, "R" if order == 1 else "R1")
        text(160, 226, p["R1"])

        if order == 1:
            line(190, 200, 455, 200)
            dot(345, 200)
            dot(395, 200)
        else:
            line(190, 200, 250, 200)
            dot(230, 200)
            res_h(250, 310, 200)
            els.append('<rect x="250" y="191" width="60" height="18" fill="none"/>')
            line(250, 200, 310, 200)
            text(280, 180, "R2")
            text(280, 226, p["R2"])
            line(310, 200, 455, 200)
            dot(345, 200)
            dot(395, 200)

            line(230, 200, 230, 55)
            line(230, 55, 380, 55)
            cap_h(385, 55)
            line(390, 55, 540, 55)
            line(540, 55, 540, 260)
            text(385, 30, f"C1 = {p['C1']}", size=14, bold=True)

        # C2
        line(345, 200, 345, 265)
        cap_v(345, 270)
        line(345, 275, 345, 340)
        text(328, 268, "C2", anchor="end", size=14, bold=True)
        text(328, 285, p["C2"], anchor="end", size=14, bold=True)

        # バイアス
        vcc(395, 110)
        line(395, 110, 395, 125)
        res_v(395, 125, 165)
        els.append('<rect x="386" y="125" width="18" height="40" fill="none"/>')
        line(395, 165, 395, 200)
        text(413, 142, "RB1", anchor="start")
        text(413, 158, p["RB"], anchor="start")
        line(395, 200, 395, 235)
        res_v(395, 235, 285)
        els.append('<rect x="386" y="235" width="18" height="50" fill="none"/>')
        line(395, 285, 395, 340)
        text(413, 258, "RB2", anchor="start")
        text(413, 274, p["RB"], anchor="start")

        # トランジスタ
        els.append('<circle cx="482" cy="200" r="40" fill="none"/>')
        line(455, 175, 455, 225, 4)
        line(455, 190, 500, 165)
        line(500, 165, 500, 110)
        vcc(500, 110)
        line(455, 210, 500, 235)
        els.append('<polygon points="493.3,231.3 482.3,230.3 486.7,222.5" fill="#222"/>')
        line(500, 235, 500, 285)
        text(552, 205, "Q1", anchor="start", size=14, bold=True)

        dot(500, 260)
        res_v(500, 285, 335)
        els.append('<rect x="491" y="285" width="18" height="50" fill="none"/>')
        line(500, 335, 500, 340)
        text(518, 307, "RE", anchor="start")
        text(518, 323, p["RE"], anchor="start")

        line(500, 260, 580, 260)
        if order == 2:
            dot(540, 260)
        cap_h(585, 260)
        line(590, 260, 680, 260)
        dot(680, 260)
        line(680, 260, 700, 260)
        els.append('<circle cx="704" cy="260" r="4" fill="#fff"/>')
        text(704, 245, "Vout")
        text(585, 238, "Cout")
        text(585, 300, p["Cout"])
        line(680, 260, 680, 285)
        res_v(680, 285, 335)
        els.append('<rect x="671" y="285" width="18" height="50" fill="none"/>')
        line(680, 335, 680, 340)
        text(698, 307, "RL", anchor="start")
        text(698, 323, p["RL"], anchor="start")
        line(345, 340, 680, 340)
        ground(520, 340)

    return (
        f'<svg viewBox="0 0 {W} 385" width="100%" height="100%" '
        'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
        'font-family="Noto Sans JP, Meiryo, sans-serif" stroke="#222" fill="none" '
        'stroke-linecap="round" style="background:#fff;border-radius:8px">'
        + "".join(els)
        + "</svg>"
    )


# ---------------------------------------------------------------- サイドバー入力
with st.sidebar:
    st.header("フィルタ特性")
    order = st.selectbox("次数", [1, 2, 3], index=1)
    kind = st.selectbox("特性", ["バターワース", "ベッセル", "チェビシェフ"])
    ripple = 0.5
    if kind == "チェビシェフ":
        ripple = st.selectbox("リップル [dB]", [0.1, 0.5, 1.0, 2.0, 3.0], index=1)
    fc_label = "リップル帯域端 fc [Hz]" if kind == "チェビシェフ" else "-3dB周波数 fc [Hz]"
    fc = st.number_input(
        fc_label, min_value=1.0, max_value=1e7, value=1000.0, step=100.0, format="%.1f"
    )
    R = st.number_input(
        "R [Ω]" if order == 1 else "R1 = R2 [Ω]",
        min_value=100.0,
        value=10000.0,
        step=1000.0,
    )

    if order == 3:
        r3_div = st.slider(
            "1次RCインピーダンス比 (R / R3)",
            min_value=2,
            max_value=50,
            value=10,
            help="R3 = R / この比率 とすることで1次RC段を低インピーダンス化し、2次Sallen-Key段との干渉を減らします。",
        )
    else:
        r3_div = 10

    st.header("トランジスタ・バイアス")
    vcc_v = st.number_input("電源電圧 Vcc [V]", min_value=3.0, value=12.0, step=1.0)
    ie_ma = st.number_input("エミッタ電流 IE [mA]", min_value=0.05, value=1.0, step=0.1)
    hfe = st.number_input("hFE", min_value=20, value=150, step=10)
    ft_mhz = st.number_input("fT [MHz]", min_value=1.0, value=200.0, step=10.0)
    rl = st.number_input("負荷抵抗 RL [Ω]", min_value=100.0, value=100000.0, step=10000.0)

    st.header("オプション")
    if order >= 2:
        correct_k = st.checkbox("Kが1未満である分をC1/C2比で補正", value=True)
    else:
        correct_k = False

    if order == 3:
        compensate_r1 = st.checkbox(
            "R1からR3の直列分を差し引く (R1 = R - R3)",
            value=False,
            help="DC〜低域において1次RCのR3がR1と直列に見える影響を簡易的に補正します。",
        )
    else:
        compensate_r1 = False

    use_e24 = st.checkbox("素子値をE24標準値に丸めてシミュレーション", value=True)
    include_coupling = st.checkbox("結合コンデンサ (Cin, Cout) の低域特性を含める", value=False)


# ---------------------------------------------------------------- 設計計算
sections = design_sections(kind, fc, ripple, order)
f0_max = max(s[0] for s in sections)
f0_min = min(s[0] for s in sections)

# バイアス(ベース電位 ≒ Vcc/2)
ve = vcc_v / 2 - 0.65
if ve <= 0:
    st.error("Vcc が小さすぎます(Vcc/2 − 0.65V > 0 となるようにしてください)。")
    st.stop()

RE_calc = ve / (ie_ma * 1e-3)
RE = nearest_e24(RE_calc)
ie = ve / RE
re_ = VT / ie
r_ac = RE * rl / (RE + rl)
K = r_ac / (r_ac + re_)
r_in = hfe * (re_ + r_ac)
rb_calc = 2 * r_in / 10
rb_each = nearest_e24(rb_calc)
rb_par = rb_each / 2
rb_tot = 1 / (1 / rb_par + 1 / r_in)

# --- 次数別の部品定数 ---
C1 = C2 = C3 = None
R3 = None
R1_val = R
R2_val = R

if order == 1:
    f0_1 = sections[0][0]
    C2 = 1 / (2 * math.pi * f0_1 * R)

elif order == 2:
    f0_2, Q_2 = sections[0]
    if correct_k and (1 - K) > 1e-9:
        disc = 1 / Q_2**2 - 8 * (1 - K)
        if disc < 0:
            st.warning("Kが小さすぎて補正解がありません。補正なし(K=1想定)で計算します。")
            ratio = 2 * Q_2
        else:
            ratio = (1 / Q_2 - math.sqrt(disc)) / (2 * (1 - K))
    else:
        ratio = 2 * Q_2
    C2 = 1 / (2 * math.pi * f0_2 * R * ratio)
    C1 = ratio**2 * C2

else:  # order == 3
    (f0_2, Q_2), (f0_1, _) = sections
    if correct_k and (1 - K) > 1e-9:
        disc = 1 / Q_2**2 - 8 * (1 - K)
        if disc < 0:
            st.warning("Kが小さすぎて補正解がありません。補正なし(K=1想定)で計算します。")
            ratio = 2 * Q_2
        else:
            ratio = (1 / Q_2 - math.sqrt(disc)) / (2 * (1 - K))
    else:
        ratio = 2 * Q_2
    C2 = 1 / (2 * math.pi * f0_2 * R * ratio)
    C1 = ratio**2 * C2

    # 1次RCを低インピーダンス化
    R3 = R / r3_div
    C3 = 1 / (2 * math.pi * f0_1 * R3)
    if compensate_r1 and R > R3:
        R1_val = R - R3

# E24丸め処理
R1s = nearest_e24(R1_val) if use_e24 else R1_val
R2s = nearest_e24(R2_val) if use_e24 else R2_val
C1s = nearest_e24(C1) if (C1 is not None and use_e24) else C1
C2s = nearest_e24(C2) if use_e24 else C2
C3s = nearest_e24(C3) if (C3 is not None and use_e24) else C3
R3s = nearest_e24(R3) if (R3 is not None and use_e24) else R3

# Cin / Cout (HPF f0/100)
f_hp = f0_min / 100
if order == 3:
    Cin_calc = 1 / (2 * math.pi * f_hp * (R3 + R1_val + R2_val + rb_tot))
elif order == 2:
    Cin_calc = 1 / (2 * math.pi * f_hp * (R1_val + R2_val + rb_tot))
else:
    Cin_calc = 1 / (2 * math.pi * f_hp * (R1_val + rb_tot))
Cout_calc = 1 / (2 * math.pi * f_hp * rl)

Cin = nearest_e24(Cin_calc)
Cout = nearest_e24(Cout_calc)

if f0_max > ft_mhz * 1e6 / 100:
    st.warning("f₀ が fT の1/100を超えています。トランジスタの高周波特性の影響が無視できません。")


# ---------------------------------------------------------------- 結果表示
st.subheader("設計結果")

if order == 1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("f₀ (1次)", f"{sections[0][0]:,.1f} Hz")
    m2.metric("次数", "1次")
    m3.metric("エミッタフォロワのゲインK", f"{K:.4f}")
    m4.metric("C2 (計算値)", fmt_c(C2))
elif order == 2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("f₀ (2次)", f"{sections[0][0]:,.1f} Hz")
    m2.metric("Q", f"{sections[0][1]:.3f}")
    m3.metric("C1/C2", f"{C1 / C2:.2f}")
    m4.metric("エミッタフォロワのゲインK", f"{K:.4f}")
else:
    (f0_2, Q_2), (f0_1, _) = sections
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("f₀ (2次SK)", f"{f0_2:,.1f} Hz")
    m2.metric("Q (2次SK)", f"{Q_2:.3f}")
    m3.metric("f₀ (1次RC)", f"{f0_1:,.1f} Hz")
    m4.metric("R3 / R", f"1/{r3_div}")

# 部品表データ作成
rows = []
if order == 1:
    rows.append({"部品": "R", "計算値": fmt_r(R1_val), "E24値": fmt_r(nearest_e24(R1_val))})
    rows.append({"部品": "C2", "計算値": fmt_c(C2), "E24値": fmt_c(nearest_e24(C2))})
else:
    if compensate_r1 and order == 3:
        rows.append({"部品": "R1 (補正後)", "計算値": fmt_r(R1_val), "E24値": fmt_r(nearest_e24(R1_val))})
        rows.append({"部品": "R2", "計算値": fmt_r(R2_val), "E24値": fmt_r(nearest_e24(R2_val))})
    else:
        rows.append({"部品": "R1 = R2", "計算値": fmt_r(R), "E24値": fmt_r(nearest_e24(R))})
    rows.append({"部品": "C1 (帰還側)", "計算値": fmt_c(C1), "E24値": fmt_c(nearest_e24(C1))})
    rows.append({"部品": "C2 (GND側)", "計算値": fmt_c(C2), "E24値": fmt_c(nearest_e24(C2))})

if order == 3:
    rows.append(
        {
            "部品": f"R3 (1次RC, =R/{r3_div})",
            "計算値": fmt_r(R3),
            "E24値": fmt_r(nearest_e24(R3)),
        }
    )
    rows.append({"部品": "C3 (1次RC)", "計算値": fmt_c(C3), "E24値": fmt_c(nearest_e24(C3))})

rows += [
    {"部品": "RE", "計算値": fmt_r(RE_calc), "E24値": fmt_r(RE)},
    {"部品": "RB1 = RB2", "計算値": fmt_r(rb_calc), "E24値": fmt_r(rb_each)},
    {"部品": "Cin (f₀/100でHPF)", "計算値": fmt_c(Cin_calc), "E24値": fmt_c(Cin)},
    {"部品": "Cout (f₀/100でHPF)", "計算値": fmt_c(Cout_calc), "E24値": fmt_c(Cout)},
]

# 回路図パラメータ
p_svg = {
    "R1": fmt_r(R1s),
    "R2": fmt_r(R2s),
    "C1": fmt_c(C1s),
    "C2": fmt_c(C2s),
    "C3": fmt_c(C3s),
    "R3": fmt_r(R3s),
    "RE": fmt_r(RE),
    "RB": fmt_r(rb_each),
    "RL": fmt_r(rl),
    "Cin": fmt_c(Cin),
    "Cout": fmt_c(Cout),
}
svg = circuit_svg(p_svg, order)
svg_page = (
    '<meta charset="utf-8">'
    "<style>html,body{margin:0;height:100%;background:transparent;}</style>"
    f"{svg}"
)

col_circuit, col_parts = st.columns([3, 2], gap="large")
with col_circuit:
    st.subheader("回路図")
    components.html(svg_page, height=440, scrolling=False)
with col_parts:
    st.subheader("部品表")
    st.dataframe(rows, hide_index=True, width="stretch", height=310)
    st.caption("容量・抵抗はE24標準値。Cin/Coutは直流カット用で、f₀/100をHPF遮断周波数として設定。")
    if order == 3:
        st.info(
            f"3次は **入力側1次RC(R3=R/{r3_div})** を低インピーダンス化し、"
            "SK(2次)段の前段に配置。出力はエミッタフォロワ直結のため低インピーダンス。"
        )


# ---------------------------------------------------------------- 周波数特性・群遅延
st.subheader("周波数特性シミュレーション")

f_start = f0_min / 500 if include_coupling else f0_min / 100
f = np.logspace(math.log10(f_start), math.log10(f0_max * 100), 800)
h_id = ideal_H(f, sections, kind, ripple, order)

cin_arg = Cin if include_coupling else None
cout_arg = Cout if include_coupling else None

if order == 1:
    h_ac = actual_H_order1(f, R1s, C2s, RE, re_, rb_par, r_in, rl, cin_arg, cout_arg)
    h_ac_dc = actual_H_order1(np.array([f0_min / 100]), R1s, C2s, RE, re_, rb_par, r_in, rl, None, None)[0]
elif order == 2:
    h_ac = actual_H_order2(f, R1s, R2s, C1s, C2s, RE, re_, rb_par, r_in, rl, cin_arg, cout_arg)
    h_ac_dc = actual_H_order2(np.array([f0_min / 100]), R1s, R2s, C1s, C2s, RE, re_, rb_par, r_in, rl, None, None)[0]
else:
    h_ac = actual_H_order3(f, R1s, R2s, C1s, C2s, R3s, C3s, RE, re_, rb_par, r_in, rl, cin_arg, cout_arg)
    h_ac_dc = actual_H_order3(np.array([f0_min / 100]), R1s, R2s, C1s, C2s, R3s, C3s, RE, re_, rb_par, r_in, rl, None, None)[0]

ref_db = 20 * np.log10(abs(h_ac_dc))
mag_ac = 20 * np.log10(np.abs(h_ac))
mag_id = 20 * np.log10(np.abs(h_id))

# 位相 [deg] と 群遅延 [ms]
phase_id = np.unwrap(np.angle(h_id)) * (180 / np.pi)
phase_ac = np.unwrap(np.angle(h_ac)) * (180 / np.pi)

w = 2 * np.pi * f
gd_id = -np.gradient(np.unwrap(np.angle(h_id)), w) * 1e3  # [ms]
gd_ac = -np.gradient(np.unwrap(np.angle(h_ac)), w) * 1e3  # [ms]

# LPF側の -3dB 周波数を安全に探索
valid_idx = np.where(mag_ac >= ref_db - 3)[0]
f_3db = f[valid_idx.max()] if len(valid_idx) > 0 else float("nan")

col_chart, col_stats = st.columns([3, 1], gap="large")
with col_stats:
    st.metric("通過帯域基準ゲイン", f"{ref_db:.2f} dB")
    st.metric("ピーキング(対基準)", f"{mag_ac.max() - ref_db:+.2f} dB")
    st.metric("-3dB周波数 (実回路)", f"{f_3db:,.1f} Hz" if not math.isnan(f_3db) else "—")
    # f=fc における群遅延
    idx_fc = int(np.argmin(np.abs(f - fc)))
    st.metric("群遅延 (@fc)", f"{gd_ac[idx_fc]:.3f} ms")

with col_chart:
    tab_gain, tab_gd, tab_phase = st.tabs(["ゲイン [dB]", "群遅延 [ms]", "位相 [deg]"])

    rule = (
        alt.Chart(pd.DataFrame({"freq": [fc]}))
        .mark_rule(strokeDash=[4, 4], color="red")
        .encode(x="freq:Q")
    )

    with tab_gain:
        df_gain = pd.concat(
            [
                pd.DataFrame({"freq": f, "val": mag_id, "series": f"理想 ({order}次)"}),
                pd.DataFrame({"freq": f, "val": mag_ac, "series": "実回路 (バイアス・負荷込み)"}),
            ]
        )
        ymax = float(max(mag_id.max(), mag_ac.max())) + 3
        chart_gain = (
            alt.Chart(df_gain)
            .mark_line()
            .encode(
                x=alt.X("freq:Q", scale=alt.Scale(type="log"), title="周波数 [Hz]"),
                y=alt.Y("val:Q", scale=alt.Scale(domain=[-60, ymax]), title="ゲイン [dB]"),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top", labelLimit=400)),
                strokeDash=alt.StrokeDash("series:N", legend=None),
            )
        )
        st.altair_chart((chart_gain + rule).properties(height=360), width="stretch")
        st.caption("赤の破線が入力した fc。実回路はKとバイアス抵抗による負荷、素子値のE24丸めを含む節点解析です。")

    with tab_gd:
        # 群遅延は fc * 10 程度までが見やすいためフィルタリング
        mask_gd = (f >= f0_min / 20) & (f <= f0_max * 10)
        df_gd = pd.concat(
            [
                pd.DataFrame({"freq": f[mask_gd], "val": gd_id[mask_gd], "series": f"理想 ({order}次)"}),
                pd.DataFrame({"freq": f[mask_gd], "val": gd_ac[mask_gd], "series": "実回路"}),
            ]
        )
        chart_gd = (
            alt.Chart(df_gd)
            .mark_line()
            .encode(
                x=alt.X("freq:Q", scale=alt.Scale(type="log"), title="周波数 [Hz]"),
                y=alt.Y("val:Q", title="群遅延 τg [ms]"),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top", labelLimit=400)),
                strokeDash=alt.StrokeDash("series:N", legend=None),
            )
        )
        st.altair_chart((chart_gd + rule).properties(height=360), width="stretch")
        st.caption("群遅延 τg = -dφ/dω。ベッセル特性を選ぶと通過帯域内で平坦になり、CWやパルス波形のリンギングを抑えられます。")

    with tab_phase:
        df_ph = pd.concat(
            [
                pd.DataFrame({"freq": f, "val": phase_id, "series": f"理想 ({order}次)"}),
                pd.DataFrame({"freq": f, "val": phase_ac, "series": "実回路"}),
            ]
        )
        chart_ph = (
            alt.Chart(df_ph)
            .mark_line()
            .encode(
                x=alt.X("freq:Q", scale=alt.Scale(type="log"), title="周波数 [Hz]"),
                y=alt.Y("val:Q", title="位相 [deg]"),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top", labelLimit=400)),
                strokeDash=alt.StrokeDash("series:N", legend=None),
            )
        )
        st.altair_chart((chart_ph + rule).properties(height=360), width="stretch")


# ---------------------------------------------------------------- SPICEネットリスト出力
with st.expander("LTspice シミュレーション用ネットリスト (.cir)"):
    spice_lines = [
        f"* NPN 1-Transistor Active LPF ({order}-Order {kind}, fc={fc}Hz)",
        f"VCC vcc 0 {vcc_v}",
        "VIN in 0 AC 1",
        f"Cin in n_in {fmt_spice(Cin)}",
    ]
    if order == 1:
        spice_lines += [
            f"R1 n_in b {fmt_spice(R1s)}",
            f"C2 b 0 {fmt_spice(C2s)}",
        ]
    elif order == 2:
        spice_lines += [
            f"R1 n_in n_a {fmt_spice(R1s)}",
            f"R2 n_a b {fmt_spice(R2s)}",
            f"C1 n_a e {fmt_spice(C1s)}",
            f"C2 b 0 {fmt_spice(C2s)}",
        ]
    else:
        spice_lines += [
            f"R3 n_in n_x {fmt_spice(R3s)}",
            f"C3 n_x 0 {fmt_spice(C3s)}",
            f"R1 n_x n_a {fmt_spice(R1s)}",
            f"R2 n_a b {fmt_spice(R2s)}",
            f"C1 n_a e {fmt_spice(C1s)}",
            f"C2 b 0 {fmt_spice(C2s)}",
        ]

    spice_lines += [
        f"RB1 vcc b {fmt_spice(rb_each)}",
        f"RB2 b 0 {fmt_spice(rb_each)}",
        "Q1 vcc b e NPN_MOD",
        f"RE e 0 {fmt_spice(RE)}",
        f"Cout e out {fmt_spice(Cout)}",
        f"RL out 0 {fmt_spice(rl)}",
        f".model NPN_MOD NPN(Is=1e-14 Bf={hfe} Tf={1/(2*math.pi*ft_mhz*1e6):.3e})",
        f".ac dec 100 {max(1, int(f0_min/500))} {int(f0_max*100)}",
        ".end",
    ]
    st.code("\n".join(spice_lines), language="text")


# ---------------------------------------------------------------- 設計式
with st.expander("設計式と注意点"):
    st.markdown(
        f"""
**次数と伝達関数**
- 1次: $H(s) = \\frac{{1}}{{1 + s/\\omega_0}}$
- 2次: $H(s) = \\frac{{1}}{{1 + s/(Q\\omega_0) + (s/\\omega_0)^2}}$
- 3次: 1次RC × 2次SK のカスケード

**2次 Sallen-Key (ゲイン $K=1$)**
$$H(s) = \\frac{{1}}{{1 + s C_2(R_1+R_2) + s^2 R_1 R_2 C_1 C_2}}$$
- $R_1=R_2=R$ のとき $C_1/C_2 = 4Q^2$, $C_2 = \\frac{{1}}{{4\\pi f_0 R Q}}$, $C_1 = \\frac{{Q}}{{\\pi f_0 R}}$
- $K < 1$ のとき: $\\frac{{1}}{{Q}} = \\frac{{2}}{{r}} + (1-K)r \\quad (r=\\sqrt{{C_1/C_2}})$ を解いて補正

**3次の構成（低インピーダンス1次RC前置き）**
Vin → Cin → R3(低Z) → ノードX → R1 → ノードA → R2 → ノードB → ベース
- $R_3 = R / {r3_div}$ として入力段を低インピーダンス化
- エミッタフォロワ出力がそのままフィルタ出力 → 出力インピーダンスが低い
- 1次RCの負荷はSK段の入力インピーダンス($\\approx 2R + r_b$)で、$R_3 \\ll 2R$ なので分離近似が成立
- $C_3 = \\frac{{1}}{{2\\pi f_{{01}} R_3}}$

**注意**
- 全体のインピーダンスレベルは $R$ で決まります。さらに下げたい場合は $R$ を小さくしてください（$C$ は大きくなります）。
- ベース側のバイアス抵抗が $R_1+R_2$ と同程度だと、通過帯域ゲインが下がります。
- ベース電流による直流バイアス分圧のずれは無視しています。実機では $V_B$・$I_E$ を測って調整してください。
"""
    )