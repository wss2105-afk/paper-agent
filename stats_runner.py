"""통계 분석 실행 모듈 — '데이터 분석 설계' 모드에서 실제 통계를 계산한다.

각 run_* 함수는 dict를 반환한다:
  {
    "tables": {표이름: DataFrame, ...},   # 앱에서 st.dataframe으로 표시
    "summary": str,                       # 결과 요약 텍스트 (Claude 해석 프롬프트에도 사용)
  }
입력 오류(변수 부족, 집단 수 불일치 등)는 ValueError로 올리고, 앱에서 st.error로 표시한다.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── 공통 유틸 ─────────────────────────────────────────────────

def _clean(df, cols):
    """분석 대상 열만 골라 결측 행 제거. 사용된 N을 함께 반환."""
    sub = df[cols].dropna()
    if len(sub) < 3:
        raise ValueError(f"결측 제거 후 남은 사례가 {len(sub)}개뿐이에요. 데이터를 확인해주세요.")
    return sub, len(sub)


def _p_str(p):
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def _numeric_check(df, cols):
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"수치형이 아닌 변수가 있어요: {', '.join(bad)}")


# ── 1. 독립표본 t검정 ─────────────────────────────────────────

def run_ttest_ind(df, dv, group_col):
    _numeric_check(df, [dv])
    sub, n = _clean(df, [dv, group_col])
    levels = sub[group_col].dropna().unique()
    if len(levels) != 2:
        raise ValueError(
            f"집단변수 '{group_col}'의 수준이 {len(levels)}개예요. "
            "독립표본 t검정은 정확히 2개 집단이 필요해요. (3개 이상이면 ANOVA를 사용하세요)"
        )
    g1 = sub[sub[group_col] == levels[0]][dv]
    g2 = sub[sub[group_col] == levels[1]][dv]

    lev_stat, lev_p = stats.levene(g1, g2)
    equal_var = lev_p >= 0.05
    t, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
    df_used = (len(g1) + len(g2) - 2) if equal_var else None

    # Cohen's d (pooled SD)
    pooled_sd = np.sqrt(((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2)
                        / (len(g1) + len(g2) - 2))
    d = (g1.mean() - g2.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    desc = pd.DataFrame({
        "집단": [str(levels[0]), str(levels[1])],
        "N": [len(g1), len(g2)],
        "평균": [round(g1.mean(), 3), round(g2.mean(), 3)],
        "표준편차": [round(g1.std(), 3), round(g2.std(), 3)],
    })
    result = pd.DataFrame({
        "t": [round(t, 3)], "p": [round(p, 4)],
        "자유도": [df_used if df_used else "Welch 보정"],
        "Cohen's d": [round(d, 3)],
        "Levene p (등분산)": [round(lev_p, 4)],
    })
    summary = (
        f"독립표본 t검정: 종속변수={dv}, 집단변수={group_col} (N={n})\n"
        f"- Levene 등분산 검정 p {_p_str(lev_p)} → {'등분산 가정 충족(Student t)' if equal_var else '등분산 위배(Welch t 적용)'}\n"
        f"- {levels[0]} (n={len(g1)}): M={g1.mean():.3f}, SD={g1.std():.3f}\n"
        f"- {levels[1]} (n={len(g2)}): M={g2.mean():.3f}, SD={g2.std():.3f}\n"
        f"- t={t:.3f}, p {_p_str(p)}, Cohen's d={d:.3f}\n"
        f"- {'통계적으로 유의함 (p < .05)' if p < 0.05 else '통계적으로 유의하지 않음 (p ≥ .05)'}"
    )
    return {"tables": {"기술통계": desc, "t검정 결과": result}, "summary": summary}


# ── 2. 대응표본 t검정 ─────────────────────────────────────────

def run_ttest_rel(df, var1, var2):
    _numeric_check(df, [var1, var2])
    sub, n = _clean(df, [var1, var2])
    t, p = stats.ttest_rel(sub[var1], sub[var2])
    diff = sub[var1] - sub[var2]
    d = diff.mean() / diff.std() if diff.std() > 0 else np.nan  # Cohen's d for paired

    desc = pd.DataFrame({
        "변수": [var1, var2],
        "N": [n, n],
        "평균": [round(sub[var1].mean(), 3), round(sub[var2].mean(), 3)],
        "표준편차": [round(sub[var1].std(), 3), round(sub[var2].std(), 3)],
    })
    result = pd.DataFrame({
        "t": [round(t, 3)], "자유도": [n - 1], "p": [round(p, 4)],
        "평균차": [round(diff.mean(), 3)], "Cohen's d": [round(d, 3)],
    })
    summary = (
        f"대응표본 t검정: {var1} vs {var2} (N={n})\n"
        f"- {var1}: M={sub[var1].mean():.3f}, SD={sub[var1].std():.3f}\n"
        f"- {var2}: M={sub[var2].mean():.3f}, SD={sub[var2].std():.3f}\n"
        f"- 평균차={diff.mean():.3f}, t({n-1})={t:.3f}, p {_p_str(p)}, Cohen's d={d:.3f}\n"
        f"- {'통계적으로 유의함 (p < .05)' if p < 0.05 else '통계적으로 유의하지 않음 (p ≥ .05)'}"
    )
    return {"tables": {"기술통계": desc, "t검정 결과": result}, "summary": summary}


# ── 3. 일원분산분석 (ANOVA) + Tukey 사후검정 ──────────────────

def run_anova(df, dv, group_col):
    _numeric_check(df, [dv])
    sub, n = _clean(df, [dv, group_col])
    levels = sub[group_col].unique()
    if len(levels) < 3:
        raise ValueError(
            f"집단변수 '{group_col}'의 수준이 {len(levels)}개예요. "
            "ANOVA는 3개 이상 집단에 사용해요. (2개면 t검정을 사용하세요)"
        )
    groups = [sub[sub[group_col] == lv][dv] for lv in levels]
    f, p = stats.f_oneway(*groups)

    # eta squared
    grand = sub[dv].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((sub[dv] - grand) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    desc = pd.DataFrame({
        "집단": [str(lv) for lv in levels],
        "N": [len(g) for g in groups],
        "평균": [round(g.mean(), 3) for g in groups],
        "표준편차": [round(g.std(), 3) for g in groups],
    })
    result = pd.DataFrame({
        "F": [round(f, 3)],
        "df(집단간, 집단내)": [f"({len(levels)-1}, {n-len(levels)})"],
        "p": [round(p, 4)], "eta²": [round(eta_sq, 3)],
    })

    tables = {"기술통계": desc, "ANOVA 결과": result}
    posthoc_txt = ""
    if p < 0.05:
        tk = stats.tukey_hsd(*groups)
        rows = []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                pv = tk.pvalue[i][j]
                rows.append({
                    "비교": f"{levels[i]} vs {levels[j]}",
                    "평균차": round(groups[i].mean() - groups[j].mean(), 3),
                    "p": round(pv, 4),
                    "유의(p<.05)": "★" if pv < 0.05 else "",
                })
        posthoc = pd.DataFrame(rows)
        tables["Tukey HSD 사후검정"] = posthoc
        sig_pairs = [r["비교"] for r in rows if r["p"] < 0.05]
        posthoc_txt = f"\n- Tukey HSD 사후검정에서 유의한 쌍: {', '.join(sig_pairs) if sig_pairs else '없음'}"

    means_txt = "; ".join(f"{lv}: M={g.mean():.3f}, SD={g.std():.3f} (n={len(g)})"
                          for lv, g in zip(levels, groups))
    summary = (
        f"일원분산분석: 종속변수={dv}, 집단변수={group_col} (N={n}, 집단 {len(levels)}개)\n"
        f"- 집단별: {means_txt}\n"
        f"- F({len(levels)-1}, {n-len(levels)})={f:.3f}, p {_p_str(p)}, eta²={eta_sq:.3f}\n"
        f"- {'통계적으로 유의함 (p < .05)' if p < 0.05 else '통계적으로 유의하지 않음 (p ≥ .05)'}"
        f"{posthoc_txt}"
    )
    return {"tables": tables, "summary": summary}


# ── 4. 상관분석 ───────────────────────────────────────────────

def run_correlation(df, cols, method="pearson"):
    if len(cols) < 2:
        raise ValueError("상관분석에는 변수 2개 이상이 필요해요.")
    _numeric_check(df, cols)
    sub, n = _clean(df, cols)

    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    k = len(cols)
    r_mat = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((k, k)), index=cols, columns=cols)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            r, p = corr_fn(sub[cols[i]], sub[cols[j]])
            r_mat.iloc[i, j] = r_mat.iloc[j, i] = round(r, 3)
            p_mat.iloc[i, j] = p_mat.iloc[j, i] = round(p, 4)
            pairs.append((cols[i], cols[j], r, p))

    sig = [f"{a}–{b}: r={r:.3f}, p {_p_str(p)}" for a, b, r, p in pairs if p < 0.05]
    method_kr = "Pearson" if method == "pearson" else "Spearman"
    summary = (
        f"{method_kr} 상관분석 (N={n}, 변수 {k}개: {', '.join(cols)})\n"
        + "\n".join(f"- {a}–{b}: r={r:.3f}, p {_p_str(p)}" for a, b, r, p in pairs)
        + f"\n유의한 상관(p<.05): {len(sig)}쌍 / 전체 {len(pairs)}쌍"
    )
    return {"tables": {"상관계수 행렬": r_mat, "p값 행렬": p_mat}, "summary": summary}


# ── 5. 카이제곱 검정 ──────────────────────────────────────────

def run_chisquare(df, var1, var2):
    sub, n = _clean(df, [var1, var2])
    ct = pd.crosstab(sub[var1], sub[var2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("두 변수 모두 2개 이상의 범주가 필요해요.")
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    low_expected = (expected < 5).sum() / expected.size * 100

    result = pd.DataFrame({
        "χ²": [round(chi2, 3)], "자유도": [dof], "p": [round(p, 4)],
        "Cramér's V": [round(cramers_v, 3)],
    })
    warn = f"\n- ⚠️ 기대빈도 5 미만 셀이 {low_expected:.0f}%예요. 20%를 넘으면 결과 해석에 주의하세요." if low_expected > 20 else ""
    summary = (
        f"카이제곱 독립성 검정: {var1} × {var2} (N={n})\n"
        f"- χ²({dof})={chi2:.3f}, p {_p_str(p)}, Cramér's V={cramers_v:.3f}\n"
        f"- {'두 변수 간 연관이 통계적으로 유의함 (p < .05)' if p < 0.05 else '유의한 연관 없음 (p ≥ .05)'}"
        f"{warn}"
    )
    return {"tables": {"교차표": ct, "카이제곱 결과": result}, "summary": summary}


# ── 6. 신뢰도 분석 (Cronbach's α) ─────────────────────────────

def run_cronbach(df, cols):
    if len(cols) < 2:
        raise ValueError("신뢰도 분석에는 문항 2개 이상이 필요해요.")
    _numeric_check(df, cols)
    sub, n = _clean(df, cols)

    def alpha(frame):
        k = frame.shape[1]
        item_var = frame.var(axis=0, ddof=1).sum()
        total_var = frame.sum(axis=1).var(ddof=1)
        return k / (k - 1) * (1 - item_var / total_var) if total_var > 0 else np.nan

    a = alpha(sub)
    rows = []
    total = sub.sum(axis=1)
    for c in cols:
        rest = total - sub[c]
        r_it = sub[c].corr(rest)
        a_del = alpha(sub.drop(columns=c)) if len(cols) > 2 else np.nan
        rows.append({
            "문항": c,
            "수정된 문항-전체 상관": round(r_it, 3),
            "문항 제거 시 α": round(a_del, 3) if not np.isnan(a_del) else "-",
        })
    item_tbl = pd.DataFrame(rows)
    result = pd.DataFrame({"Cronbach's α": [round(a, 3)], "문항 수": [len(cols)], "N": [n]})

    level = "우수 (≥ .9)" if a >= 0.9 else "양호 (≥ .8)" if a >= 0.8 else "수용 가능 (≥ .7)" if a >= 0.7 else "낮음 (< .7)"
    summary = (
        f"신뢰도 분석: 문항 {len(cols)}개 ({', '.join(cols)}), N={n}\n"
        f"- Cronbach's α = {a:.3f} → {level}\n"
        + "\n".join(f"- {r['문항']}: 문항-전체 상관={r['수정된 문항-전체 상관']}, 제거 시 α={r['문항 제거 시 α']}"
                    for r in rows)
    )
    return {"tables": {"신뢰도": result, "문항 분석": item_tbl}, "summary": summary}


# ── 7. 중다회귀분석 (OLS) ─────────────────────────────────────

def run_regression(df, dv, ivs):
    import statsmodels.api as sm
    if not ivs:
        raise ValueError("독립변수를 1개 이상 선택해주세요.")
    _numeric_check(df, [dv] + list(ivs))
    sub, n = _clean(df, [dv] + list(ivs))

    X = sm.add_constant(sub[list(ivs)])
    model = sm.OLS(sub[dv], X).fit()

    # 표준화 계수(β): z-score 회귀
    zsub = (sub - sub.mean()) / sub.std()
    zmodel = sm.OLS(zsub[dv], sm.add_constant(zsub[list(ivs)])).fit()

    rows = []
    for name in ["const"] + list(ivs):
        rows.append({
            "변수": "(상수)" if name == "const" else name,
            "B": round(model.params[name], 3),
            "SE": round(model.bse[name], 3),
            "β": "-" if name == "const" else round(zmodel.params[name], 3),
            "t": round(model.tvalues[name], 3),
            "p": round(model.pvalues[name], 4),
        })
    coef_tbl = pd.DataFrame(rows)
    fit_tbl = pd.DataFrame({
        "R²": [round(model.rsquared, 3)],
        "수정 R²": [round(model.rsquared_adj, 3)],
        "F": [round(model.fvalue, 3)],
        "F의 p": [round(model.f_pvalue, 4)],
        "N": [n],
    })

    sig_ivs = [r["변수"] for r in rows[1:] if r["p"] < 0.05]
    summary = (
        f"중다회귀분석: 종속변수={dv}, 독립변수={', '.join(ivs)} (N={n})\n"
        f"- 모형: R²={model.rsquared:.3f}, 수정 R²={model.rsquared_adj:.3f}, "
        f"F({int(model.df_model)}, {int(model.df_resid)})={model.fvalue:.3f}, p {_p_str(model.f_pvalue)}\n"
        + "\n".join(
            f"- {r['변수']}: B={r['B']}, β={r['β']}, t={r['t']}, p {_p_str(r['p'])}"
            for r in rows[1:])
        + f"\n- 유의한 예측변수(p<.05): {', '.join(sig_ivs) if sig_ivs else '없음'}"
    )
    return {"tables": {"회귀계수": coef_tbl, "모형 적합도": fit_tbl}, "summary": summary}


# ── 8. 다층모형 (HLM / 선형혼합모형) ──────────────────────────

def run_hlm(df, dv, ivs, group_col):
    import statsmodels.formula.api as smf
    _numeric_check(df, [dv] + list(ivs))
    sub, n = _clean(df, [dv] + list(ivs) + [group_col])
    n_groups = sub[group_col].nunique()
    if n_groups < 5:
        raise ValueError(
            f"집단(상위수준) 수가 {n_groups}개뿐이에요. HLM은 보통 집단이 충분히 많아야 해요 (권장 10개 이상)."
        )

    # 컬럼명에 공백/특수문자가 있으면 formula가 깨지므로 임시 이름으로 치환
    rename = {c: f"v{i}" for i, c in enumerate([dv] + list(ivs) + [group_col])}
    inv = {v: k for k, v in rename.items()}
    d2 = sub.rename(columns=rename)
    dv_r = rename[dv]
    ivs_r = [rename[c] for c in ivs]
    grp_r = rename[group_col]

    # 무조건모형(null model) → ICC
    null = smf.mixedlm(f"{dv_r} ~ 1", d2, groups=d2[grp_r]).fit(reml=True)
    var_between = float(null.cov_re.iloc[0, 0])
    var_within = float(null.scale)
    icc = var_between / (var_between + var_within)

    # 연구모형 (임의절편 + 고정효과)
    formula = f"{dv_r} ~ " + " + ".join(ivs_r)
    model = smf.mixedlm(formula, d2, groups=d2[grp_r]).fit(reml=True)

    rows = []
    for name in model.params.index:
        if name == "Group Var":
            continue
        disp = "(절편)" if name == "Intercept" else inv.get(name, name)
        rows.append({
            "고정효과": disp,
            "계수": round(model.params[name], 3),
            "SE": round(model.bse[name], 3),
            "z": round(model.tvalues[name], 3),
            "p": round(model.pvalues[name], 4),
        })
    coef_tbl = pd.DataFrame(rows)
    var_tbl = pd.DataFrame({
        "집단 간 분산(절편)": [round(float(model.cov_re.iloc[0, 0]), 3)],
        "집단 내 분산(잔차)": [round(float(model.scale), 3)],
        "ICC(무조건모형)": [round(icc, 3)],
        "집단 수": [n_groups], "N": [n],
    })

    sig = [r["고정효과"] for r in rows if r["고정효과"] != "(절편)" and r["p"] < 0.05]
    summary = (
        f"다층모형(HLM, 임의절편): 종속변수={dv}, 독립변수={', '.join(ivs)}, "
        f"상위수준={group_col} (집단 {n_groups}개, N={n})\n"
        f"- ICC={icc:.3f} → 전체 분산의 {icc*100:.1f}%가 집단 간 차이. "
        f"{'다층모형 사용이 정당함 (ICC > .05)' if icc > 0.05 else 'ICC가 낮아 일반 회귀도 고려 가능'}\n"
        + "\n".join(
            f"- {r['고정효과']}: 계수={r['계수']}, z={r['z']}, p {_p_str(r['p'])}"
            for r in rows if r["고정효과"] != "(절편)")
        + f"\n- 유의한 고정효과(p<.05): {', '.join(sig) if sig else '없음'}"
    )
    return {"tables": {"고정효과": coef_tbl, "분산성분": var_tbl}, "summary": summary}


# ── 9. 잠재계층분석 (LCA) ─────────────────────────────────────

def run_lca(df, cols, n_classes=0, max_classes=5, measurement="continuous"):
    """잠재계층분석(StepMix). n_classes=0이면 BIC 최소 모형 자동 선택.
    measurement: 'continuous'(리커트 평균 등 연속형) 또는 'categorical'(범주형)."""
    from stepmix.stepmix import StepMix

    if len(cols) < 2:
        raise ValueError("잠재계층분석에는 지표 변수 2개 이상이 필요해요.")
    sub, n = _clean(df, cols)
    if n < 50:
        raise ValueError(f"사례 수가 {n}개예요. LCA는 최소 50 사례 이상을 권장해요.")

    if measurement == "continuous":
        _numeric_check(df, cols)
        X = sub[cols].to_numpy(dtype=float)
    else:
        X = np.column_stack([pd.factorize(sub[c])[0] for c in cols]).astype(float)

    def make_model(k):
        kw = dict(n_components=k, measurement=measurement,
                  random_state=42, n_init=3, verbose=0)
        try:
            return StepMix(progress_bar=0, **kw)
        except TypeError:
            return StepMix(**kw)

    rows, models = [], {}
    for k in range(1, max_classes + 1):
        m = make_model(k)
        m.fit(X)
        if k == 1:
            ent = "-"
        else:
            pp = np.clip(m.predict_proba(X), 1e-12, 1)
            ent = round(1 - (-(pp * np.log(pp)).sum()) / (n * np.log(k)), 3)
        rows.append({"계층 수": k, "AIC": round(m.aic(X), 1),
                     "BIC": round(m.bic(X), 1), "Entropy": ent})
        models[k] = m

    comp = pd.DataFrame(rows)
    best_k = int(comp.loc[comp["BIC"].idxmin(), "계층 수"])
    k_sel = int(n_classes) if n_classes and n_classes >= 2 else best_k

    tables = {"모형 비교 (계층 수 결정)": comp}
    if k_sel == 1:
        summary = (
            f"잠재계층분석(LCA): 지표 {len(cols)}개, N={n}, 탐색 범위 1~{max_classes}계층\n"
            f"- BIC 기준 최적 모형이 1계층이에요. 데이터에서 뚜렷한 잠재계층이 "
            f"구분되지 않는다는 의미로, 표본이 동질적일 가능성이 커요.\n"
            + "\n".join(f"- {r['계층 수']}계층: AIC={r['AIC']}, BIC={r['BIC']}, Entropy={r['Entropy']}" for r in rows)
        )
        return {"tables": tables, "summary": summary}

    model = models[k_sel]
    labels = model.predict(X) + 1
    assigned = sub.copy()
    assigned["_계층"] = labels

    prop = pd.DataFrame({
        "계층": [f"계층 {c}" for c in sorted(set(labels))],
        "N": [int((labels == c).sum()) for c in sorted(set(labels))],
        "비율(%)": [round((labels == c).mean() * 100, 1) for c in sorted(set(labels))],
    })
    tables["계층 크기"] = prop

    if measurement == "continuous":
        profile = assigned.groupby("_계층")[list(cols)].mean().round(3)
        profile.index = [f"계층 {i}" for i in profile.index]
        tables["계층별 프로필 (지표 평균)"] = profile
        prof_txt = "\n".join(
            f"- {idx}: " + ", ".join(f"{c}={profile.loc[idx, c]}" for c in cols)
            for idx in profile.index)
    else:
        rows_p = []
        for c_lab in sorted(set(labels)):
            grp = assigned[assigned["_계층"] == c_lab]
            row = {"계층": f"계층 {c_lab}"}
            for col in cols:
                top = grp[col].mode().iloc[0]
                share = (grp[col] == top).mean() * 100
                row[col] = f"{top} ({share:.0f}%)"
            rows_p.append(row)
        profile = pd.DataFrame(rows_p).set_index("계층")
        tables["계층별 프로필 (최빈 응답과 비율)"] = profile
        prof_txt = profile.to_string()

    ent_sel = next(r["Entropy"] for r in rows if r["계층 수"] == k_sel)
    summary = (
        f"잠재계층분석(LCA): 지표 {len(cols)}개 ({', '.join(cols)}), N={n}\n"
        f"- 모형 비교(1~{max_classes}계층): "
        + "; ".join(f"{r['계층 수']}계층 BIC={r['BIC']}" for r in rows) + "\n"
        f"- 선택 모형: {k_sel}계층 "
        f"({'BIC 최소 자동 선택' if not n_classes else '사용자 지정'}"
        f"{', BIC 최적은 ' + str(best_k) + '계층' if n_classes and k_sel != best_k else ''}), "
        f"Entropy={ent_sel}\n"
        f"- 계층 크기: " + ", ".join(f"계층 {r['계층']}: {r['N']}명({r['비율(%)']}%)".replace("계층 계층", "계층")
                                     for r in prop.to_dict("records")) + "\n"
        f"[계층별 프로필]\n{prof_txt}"
    )
    return {"tables": tables, "summary": summary}


# ── 10. 구조방정식모형 (SEM) ──────────────────────────────────

SEM_EXAMPLE = """# 측정모형 (=~ : 잠재변수 정의)
동기 =~ 문항1 + 문항2 + 문항3
성취 =~ 문항4 + 문항5 + 문항6
# 구조모형 (~ : 회귀 경로)
성취 ~ 동기"""


def run_sem(df, model_spec):
    import semopy
    if not model_spec.strip():
        raise ValueError("모델식을 입력해주세요.")

    # 모델식에 등장하는 관측변수 확인
    used = set()
    for line in model_spec.splitlines():
        line = line.split("#")[0]
        for sep in ["=~", "~~", "~"]:
            if sep in line:
                lhs, rhs = line.split(sep, 1)
                for tok in [lhs] + rhs.split("+"):
                    tok = tok.strip()
                    if tok and tok in df.columns:
                        used.add(tok)
                break
    if not used:
        raise ValueError("모델식의 변수명이 데이터 열 이름과 하나도 일치하지 않아요. 열 이름을 확인해주세요.")

    _numeric_check(df, sorted(used))
    sub, n = _clean(df, sorted(used))

    model = semopy.Model(model_spec)
    model.fit(sub)
    est = model.inspect(std_est=True)

    # 경로계수 표 정리
    est_disp = est.rename(columns={
        "lval": "종속/지표", "op": "관계", "rval": "설명/잠재",
        "Estimate": "비표준화", "Est. Std": "표준화(β)",
        "Std. Err": "SE", "z-value": "z", "p-value": "p",
    })
    for c in ["비표준화", "표준화(β)", "SE", "z", "p"]:
        if c in est_disp.columns:
            est_disp[c] = pd.to_numeric(est_disp[c], errors="coerce").round(3)

    fit = semopy.calc_stats(model).T.reset_index()
    fit.columns = ["지수", "값"]
    fit["값"] = pd.to_numeric(fit["값"], errors="coerce").round(3)
    fit_d = dict(zip(fit["지수"], fit["값"]))

    def fit_line(name, good, direction):
        v = fit_d.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        ok = (v >= good) if direction == ">=" else (v <= good)
        return f"{name}={v} ({'양호' if ok else '기준 미달'}, 기준 {direction} {good})"

    checks = [x for x in [
        fit_line("CFI", 0.90, ">="), fit_line("TLI", 0.90, ">="),
        fit_line("RMSEA", 0.08, "<="), fit_line("SRMR", 0.08, "<="),
    ] if x]

    # 사용자가 모델식에 직접 선언한 구조 경로(~)만 요약에 표시
    # (semopy inspect는 측정 부하량도 op '~'로 내놓으므로 스펙에서 선언된 쌍만 추림)
    declared = set()
    for line in model_spec.splitlines():
        line = line.split("#")[0]
        if "~" in line and "=~" not in line and "~~" not in line:
            lhs, rhs = line.split("~", 1)
            for tok in rhs.split("+"):
                declared.add((lhs.strip(), tok.strip()))
    paths = est[est["op"] == "~"]
    path_lines = []
    for _, r in paths.iterrows():
        if (str(r["lval"]), str(r["rval"])) not in declared:
            continue  # 측정 부하량 등은 '모수 추정치' 표에서 확인
        try:
            pv = float(r["p-value"])
            beta = float(r["Est. Std"])
            path_lines.append(
                f"- {r['rval']} → {r['lval']}: β={beta:.3f}, p {_p_str(pv)}"
                f" {'(유의)' if pv < 0.05 else '(비유의)'}"
            )
        except (ValueError, TypeError):
            path_lines.append(f"- {r['rval']} → {r['lval']}: 추정 실패")

    summary = (
        f"구조방정식모형(SEM) 분석 (N={n}, 관측변수 {len(used)}개)\n"
        f"[모델식]\n{model_spec.strip()}\n"
        f"[적합도] " + ("; ".join(checks) if checks else "적합도 지수 산출 불가") + "\n"
        f"[구조 경로]\n" + ("\n".join(path_lines) if path_lines else "- 구조 경로 없음(측정모형만 추정)")
    )
    return {"tables": {"모수 추정치": est_disp, "적합도 지수": fit}, "summary": summary}
