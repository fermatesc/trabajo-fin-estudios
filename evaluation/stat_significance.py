"""
Validación estadística de la significancia de estrategias de cartera.

Compara estrategias a partir de sus series de retornos diarios mediante:
  1. Deflated / Probabilistic Sharpe Ratio (Bailey & López de Prado, 2014),
     que corrige el sesgo de selección por haber probado múltiples
     configuraciones (multiple testing) y la no-normalidad de los retornos.
  2. Test de Jobson-Korkie con la corrección de Memmel (2003) para la
     diferencia entre dos ratios de Sharpe sobre series solapadas.
  3. Intervalos de confianza por bootstrap estacionario de bloques
     (Politis & Romano, 1994) para cualquier métrica.

Funciones puras (numpy / scipy). El cálculo del Sharpe es CONSISTENTE con
`evaluation/metrics.py::calculate_portfolio_metrics`:
    daily_rf  = rf_annual / periods_per_year
    excess    = returns - daily_rf
    sharpe    = mean(excess) * periods_per_year / (std(returns) * sqrt(periods_per_year))

Casos límite: series vacías o con volatilidad nula devuelven np.nan en lugar
de lanzar excepciones.

Referencias:
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".
  The Journal of Portfolio Management.
- Memmel, C. (2003). "Performance Hypothesis Testing with the Sharpe Ratio".
  Finance Letters.
- Politis, D. & Romano, J. (1994). "The Stationary Bootstrap". JASA.
"""
import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def annualized_sharpe(returns, rf_annual=0.045, periods_per_year=252):
    """
    Ratio de Sharpe anualizado, consistente con
    `evaluation/metrics.py::calculate_portfolio_metrics`.

    Exceso sobre la tasa libre de riesgo diaria (rf_annual / periods_per_year);
    Sharpe = media(exceso) * periods_per_year / (std(returns) * sqrt(periods_per_year)).
    Nótese que la volatilidad se calcula sobre los retornos brutos (no sobre el
    exceso), igual que en `metrics.py`. Se usa std poblacional (ddof=0) para
    reproducir exactamente `np.std` de ese módulo.

    Args:
        returns: array 1D de retornos diarios.
        rf_annual: tasa libre de riesgo anualizada (por defecto 0.045, FRED DTB3).
        periods_per_year: periodos por año (252 días de trading).

    Returns:
        float con el Sharpe anualizado, o np.nan si la serie es vacía o la
        volatilidad es nula.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size == 0:
        return np.nan

    daily_rf = rf_annual / periods_per_year
    excess = r - daily_rf

    vol = np.std(r)
    if not np.isfinite(vol) or vol == 0.0:
        return np.nan

    annualized_excess = np.mean(excess) * periods_per_year
    annualized_vol = vol * np.sqrt(periods_per_year)
    return float(annualized_excess / annualized_vol)


def _sharpe_per_period(returns):
    """
    Sharpe NO anualizado (por periodo), sobre exceso nulo (rf=0).

    Es la forma natural para los estadísticos de Bailey-LdP y Jobson-Korkie,
    que trabajan con momentos por periodo. Devuelve np.nan en casos límite.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2:
        return np.nan
    sd = np.std(r, ddof=1)
    if not np.isfinite(sd) or sd == 0.0:
        return np.nan
    return float(np.mean(r) / sd)


def deflated_sharpe_ratio(returns, n_trials, benchmark_sr=0.0,
                          periods_per_year=252, sr_trials_std=None):
    """
    Deflated / Probabilistic Sharpe Ratio (Bailey & López de Prado, 2014).

    Corrige el sesgo de selección que aparece al elegir la mejor de `n_trials`
    configuraciones probadas. Se estima el Sharpe esperado máximo bajo H0
    (estrategia sin habilidad) y se calcula la probabilidad de que el Sharpe
    observado lo supere, teniendo en cuenta skewness y curtosis de la serie.

    El Sharpe se mide aquí en forma POR PERIODO sobre exceso nulo (rf=0); lo que
    importa para el test es la *forma* de la distribución de retornos, no el
    nivel de la tasa libre de riesgo. El campo `sr_obs` del resultado se reporta
    anualizado (sr_periodo * sqrt(periods_per_year)) para que sea legible y
    comparable con `annualized_sharpe` cuando rf=0.

    Umbral esperado bajo H0 (máximo de N gaussianas i.i.d.):
        SR0 = sigma_SR * [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]
    donde γ es la constante de Euler-Mascheroni y sigma_SR la dispersión de los
    Sharpe (por periodo) entre las N configuraciones probadas.

    HEURÍSTICA: si no se conoce esa dispersión `sr_trials_std`, se usa
    sr_trials_std = |SR_obs (por periodo)| / 2. Es una aproximación pragmática
    (no parte del paper original): a falta de la varianza real entre estimadores,
    asumimos que la dispersión de los ensayos es del orden de la mitad del Sharpe
    observado. Para n_trials=1, SR0 colapsa a `benchmark_sr` (no hay corrección
    por selección) y el resultado equivale al PSR clásico.

    Args:
        returns: array 1D de retornos diarios.
        n_trials: número de configuraciones/estrategias probadas (N).
        benchmark_sr: Sharpe de referencia POR PERIODO bajo H0 (por defecto 0.0).
        periods_per_year: periodos por año, solo para anualizar `sr_obs`.
        sr_trials_std: dispersión (std) de los Sharpe por periodo de las N
            configuraciones probadas. Si es None, se aplica la heurística.

    Returns:
        dict con:
            sr_obs: Sharpe observado ANUALIZADO (rf=0).
            sr0: umbral esperado máximo bajo H0 (por periodo).
            dsr: probabilidad Φ(...) de que la habilidad sea real (DSR/PSR).
            T: número de observaciones.
        En casos límite, sr_obs/sr0/dsr pueden ser np.nan.
    """
    r = np.asarray(returns, dtype=float).ravel()
    T = int(r.size)
    nan_result = {"sr_obs": np.nan, "sr0": np.nan, "dsr": np.nan, "T": T}

    if T < 2:
        return nan_result

    sr_period = _sharpe_per_period(r)
    if not np.isfinite(sr_period):
        return nan_result

    n = max(int(n_trials), 1)

    if n <= 1:
        sr0 = float(benchmark_sr)
    else:
        if sr_trials_std is None:
            sigma_sr = abs(sr_period) / 2.0
        else:
            sigma_sr = float(sr_trials_std)

        gamma = EULER_MASCHERONI
        z1 = stats.norm.ppf(1.0 - 1.0 / n)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
        sr0 = float(benchmark_sr + sigma_sr * ((1.0 - gamma) * z1 + gamma * z2))

    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))

    denom_var = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * sr_period ** 2
    if not np.isfinite(denom_var) or denom_var <= 0.0 or T < 2:
        dsr = np.nan
    else:
        z = (sr_period - sr0) * np.sqrt(T - 1.0) / np.sqrt(denom_var)
        dsr = float(stats.norm.cdf(z))

    return {
        "sr_obs": float(sr_period * np.sqrt(periods_per_year)),
        "sr0": sr0,
        "dsr": dsr,
        "T": T,
    }


def sharpe_difference_test(returns_a, returns_b, periods_per_year=252):
    """
    Test de Jobson-Korkie (1981) con corrección de Memmel (2003) para la
    diferencia entre dos ratios de Sharpe sobre series SOLAPADAS (mismas fechas).

    Bajo H0: SR_a = SR_b. El estadístico z es asintóticamente N(0,1) y tiene en
    cuenta la correlación entre ambas series. Los Sharpe se calculan por periodo
    sobre exceso nulo (rf=0); como rf es común a ambas series, no afecta a la
    diferencia. Los campos `sharpe_a`/`sharpe_b` se reportan ANUALIZADOS para
    facilitar la lectura; `diff` es su diferencia anualizada.

    Estadístico (Memmel, 2003):
        var(SR_a - SR_b) = (1/T) · [ 2 - 2ρ
                                     + 0.5(SR_a² + SR_b² - 2·SR_a·SR_b·ρ²) ]
        z = (SR_a - SR_b) / sqrt(var)
    con SR_i por periodo y ρ la correlación de Pearson entre las series.

    Args:
        returns_a, returns_b: arrays 1D de retornos diarios, misma longitud.
        periods_per_year: periodos por año, solo para anualizar los Sharpe reportados.

    Returns:
        dict con sharpe_a, sharpe_b (anualizados), diff (anualizada), z_stat,
        p_value (bilateral). z_stat/p_value pueden ser np.nan en casos límite.
    """
    a = np.asarray(returns_a, dtype=float).ravel()
    b = np.asarray(returns_b, dtype=float).ravel()

    ann = np.sqrt(periods_per_year)
    sr_a_p = _sharpe_per_period(a)
    sr_b_p = _sharpe_per_period(b)

    result = {
        "sharpe_a": float(sr_a_p * ann) if np.isfinite(sr_a_p) else np.nan,
        "sharpe_b": float(sr_b_p * ann) if np.isfinite(sr_b_p) else np.nan,
        "diff": np.nan,
        "z_stat": np.nan,
        "p_value": np.nan,
    }

    if a.size != b.size or a.size < 3:
        return result
    if not (np.isfinite(sr_a_p) and np.isfinite(sr_b_p)):
        return result

    result["diff"] = float((sr_a_p - sr_b_p) * ann)

    T = a.size
    sd_a, sd_b = np.std(a, ddof=1), np.std(b, ddof=1)
    if sd_a == 0.0 or sd_b == 0.0:
        return result
    rho = float(np.corrcoef(a, b)[0, 1])

    var_diff = (1.0 / T) * (
        2.0 - 2.0 * rho
        + 0.5 * (sr_a_p ** 2 + sr_b_p ** 2 - 2.0 * sr_a_p * sr_b_p * rho ** 2)
    )
    if not np.isfinite(var_diff) or var_diff <= 0.0:
        return result

    z = (sr_a_p - sr_b_p) / np.sqrt(var_diff)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    result["z_stat"] = float(z)
    result["p_value"] = float(p)
    return result


def block_bootstrap_ci(returns, metric_fn, n_boot=2000, block_size=21,
                       alpha=0.05, seed=0):
    """
    Intervalo de confianza por bootstrap estacionario de bloques (Politis &
    Romano, 1994) para una métrica arbitraria `metric_fn(returns) -> float`.

    Reconstruye series remuestreadas concatenando bloques de longitud aleatoria
    (geométrica de media `block_size`), preservando así la autocorrelación de
    corto plazo de los retornos. El IC se obtiene por percentiles de la
    distribución bootstrap de la métrica.

    Args:
        returns: array 1D de retornos diarios.
        metric_fn: función que mapea un array de retornos a un escalar
            (p.ej. lambda r: annualized_sharpe(r)).
        n_boot: número de réplicas bootstrap.
        block_size: longitud media esperada de los bloques (en periodos).
        alpha: nivel de significancia (IC al 1 - alpha).
        seed: semilla para np.random.default_rng.

    Returns:
        dict con point (métrica observada), ci_low, ci_high, alpha.
        ci_low/ci_high pueden ser np.nan si no hay réplicas válidas.
    """
    r = np.asarray(returns, dtype=float).ravel()
    T = r.size

    point = metric_fn(r) if T > 0 else np.nan
    try:
        point = float(point)
    except (TypeError, ValueError):
        point = np.nan

    result = {"point": point, "ci_low": np.nan, "ci_high": np.nan,
              "alpha": float(alpha)}

    if T < 2:
        return result

    rng = np.random.default_rng(seed)
    p_geom = 1.0 / max(block_size, 1)

    stats_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = np.empty(T, dtype=float)
        filled = 0
        while filled < T:
            start = rng.integers(0, T)
            length = rng.geometric(p_geom)
            length = min(length, T - filled)
            idx = (start + np.arange(length)) % T
            sample[filled:filled + length] = r[idx]
            filled += length
        try:
            stats_boot[i] = float(metric_fn(sample))
        except (TypeError, ValueError):
            stats_boot[i] = np.nan

    valid = stats_boot[np.isfinite(stats_boot)]
    if valid.size == 0:
        return result

    lo = np.percentile(valid, 100.0 * (alpha / 2.0))
    hi = np.percentile(valid, 100.0 * (1.0 - alpha / 2.0))
    result["ci_low"] = float(lo)
    result["ci_high"] = float(hi)
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    T = 1000

    daily_vol = 0.01
    returns_a = rng.normal(loc=0.0008, scale=daily_vol, size=T)
    returns_b = rng.normal(loc=0.0001, scale=daily_vol, size=T)

    print("=" * 70)
    print("TEST (a): A domina a B  ->  diff > 0 y p_value bajo")
    print("=" * 70)
    diff_test = sharpe_difference_test(returns_a, returns_b)
    for k, v in diff_test.items():
        print(f"  {k:>10}: {v}")
    assert diff_test["diff"] > 0, "A debería tener mayor Sharpe que B"
    assert diff_test["p_value"] < 0.10, "diferencia debería ser significativa"
    print("  OK: diff > 0 y p_value < 0.10")

    print()
    print("=" * 70)
    print("TEST (b): DSR cae al aumentar n_trials (mismo ruido)")
    print("=" * 70)
    noisy = rng.normal(loc=0.0004, scale=daily_vol, size=T)
    dsr_1 = deflated_sharpe_ratio(noisy, n_trials=1)
    dsr_many = deflated_sharpe_ratio(noisy, n_trials=500)
    print(f"  n_trials=1   -> sr_obs={dsr_1['sr_obs']:.3f}  sr0={dsr_1['sr0']:.4f}  dsr={dsr_1['dsr']:.4f}")
    print(f"  n_trials=500 -> sr_obs={dsr_many['sr_obs']:.3f}  sr0={dsr_many['sr0']:.4f}  dsr={dsr_many['dsr']:.4f}")
    assert dsr_many["dsr"] < dsr_1["dsr"], "DSR debe bajar con más trials"
    print("  OK: dsr(n_trials=500) < dsr(n_trials=1)")

    print()
    print("=" * 70)
    print("TEST (c): el IC bootstrap contiene el Sharpe puntual")
    print("=" * 70)
    metric = lambda x: annualized_sharpe(x, rf_annual=0.045)
    ci = block_bootstrap_ci(returns_a, metric, n_boot=1000, block_size=21, seed=7)
    print(f"  point   = {ci['point']:.4f}")
    print(f"  CI 95%  = [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"], "el punto debe caer en el IC"
    print("  OK: ci_low <= point <= ci_high")

    print()
    print("=" * 70)
    print("CASOS LÍMITE (no deben romper; devuelven np.nan)")
    print("=" * 70)
    empty = np.array([])
    zero_vol = np.zeros(50)
    print(f"  annualized_sharpe(vacio)        = {annualized_sharpe(empty)}")
    print(f"  annualized_sharpe(vol=0)        = {annualized_sharpe(zero_vol)}")
    print(f"  deflated_sharpe_ratio(vacio)    = {deflated_sharpe_ratio(empty, n_trials=10)}")
    print(f"  sharpe_difference_test(vacios)  = {sharpe_difference_test(empty, empty)}")
    print(f"  block_bootstrap_ci(vacio)       = {block_bootstrap_ci(empty, metric)}")
    print()
    print("Todos los tests pasaron.")
