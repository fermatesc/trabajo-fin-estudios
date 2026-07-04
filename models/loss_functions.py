"""Funciones de pérdida diferenciables para entrenar carteras orientadas a
Sharpe, varianza mínima o CVaR, con penalizaciones opcionales de rotación y
anclaje a Equal-Weight."""

import torch
import torch.nn as nn
from scipy.stats import norm


class DifferentiableSharpeLoss(nn.Module):
    """Función de pérdida que calcula el Ratio de Sharpe negativo para que el
    optimizador (Adam) lo minimice.

    Notes:
        El cálculo del Sharpe sobre un solo día (en lugar de una
        secuencia/batch) es una limitación conocida del pipeline actual
        iterativo, pero resulta aceptable para los objetivos de este
        Trabajo de Fin de Máster.

    Args:
        risk_free_rate: Tasa libre de riesgo diaria (defecto 0.0).
        risk_mode: Modo de cálculo del riesgo. ``"std"`` usa la dispersión
            transversal de un día (variante auditoría); ``"cov"`` usa la
            volatilidad de cartera ex-ante ``sqrt(w'Sigma w)`` con shrinkage.
        shrinkage: Intensidad del shrinkage de Ledoit-Wolf cuando
            ``risk_mode="cov"`` (defecto 0.1).
    """

    def __init__(self, risk_free_rate=0.0, risk_mode="std", shrinkage=0.1):
        super(DifferentiableSharpeLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.risk_mode = risk_mode
        self.shrinkage = shrinkage

    def forward(self, weights, returns, cov_matrix=None):
        """Calcula el Sharpe negativo de la cartera.

        Args:
            weights: Tensor (N,) con los pesos de inversión predichos por la QGNN.
            returns: Tensor (N,) con los retornos reales futuros (y).
            cov_matrix: Tensor (N, N) con la matriz de covarianza para el riesgo.

        Returns:
            Tensor escalar con el Ratio de Sharpe negado.

        Notes:
            Si ``risk_mode="cov"``, la covarianza se estabiliza mediante
            shrinkage de Ledoit-Wolf hacia la identidad escalada, lo que
            estabiliza el gradiente sin descartar la información de
            covarianza.
        """
        port_return = torch.dot(weights, returns)

        if self.risk_mode == "cov" and cov_matrix is not None:
            n = cov_matrix.shape[0]
            diag_mean = torch.diagonal(cov_matrix).mean()
            eye = torch.eye(n, device=cov_matrix.device, dtype=cov_matrix.dtype)
            cov_s = (1.0 - self.shrinkage) * cov_matrix + self.shrinkage * diag_mean * eye
            port_variance = torch.matmul(weights, torch.matmul(cov_s, weights))
            port_volatility = torch.sqrt(port_variance + 1e-6)
        else:
            port_volatility = torch.std(weights * returns) * 100 + 1e-6

        sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility

        return -sharpe_ratio


class TurnoverAwareSharpeLoss(nn.Module):
    """Variante de DifferentiableSharpeLoss que añade una penalización L1 sobre la
    rotación (turnover) de la cartera entre dos pasos consecutivos.

    Notes:
        La penalización actúa SOLO sobre los pesos actuales (weights) para
        que el gradiente fluya hacia atrás a través de ellos. prev_weights
        debe llegar ya detached desde el bucle de entrenamiento (no se llama
        .detach() aquí).

    Args:
        risk_free_rate  : Tasa libre de riesgo diaria (defecto 0.0).
        turnover_lambda : Peso de la penalización L1 de rotación (defecto 0.0015).
    """

    def __init__(self, risk_free_rate=0.0, turnover_lambda=0.0015):
        super(TurnoverAwareSharpeLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.turnover_lambda = turnover_lambda

    def forward(self, weights, returns, cov_matrix=None, prev_weights=None):
        """Calcula el Sharpe negativo más la penalización de rotación.

        Args:
            weights: Tensor (N,) con los pesos predichos (requires_grad=True).
            returns: Tensor (N,) con los retornos reales futuros.
            cov_matrix: Tensor (N, N) covarianza (opcional).
            prev_weights: Tensor (N,) pesos del paso anterior, ya detached (opcional).

        Returns:
            Tensor escalar diferenciable respecto a weights.
        """
        port_return = torch.dot(weights, returns)

        if cov_matrix is not None:
            port_variance = torch.matmul(weights, torch.matmul(cov_matrix, weights))
            port_vol = torch.sqrt(port_variance + 1e-6)
        else:
            port_vol = torch.std(weights * returns) * 100 + 1e-6

        base = -(port_return - self.risk_free_rate) / port_vol

        if prev_weights is not None:
            turnover_penalty = self.turnover_lambda * torch.sum(torch.abs(weights - prev_weights))
        else:
            turnover_penalty = 0.0

        return base + turnover_penalty


class WindowedSharpeLoss(nn.Module):
    """Sharpe Loss calculado sobre una ventana de retornos de cartera para
    reducir el ruido del objetivo puntual de un solo día.

    Notes:
        El harness de evaluación mantiene un buffer de los últimos retornos
        de cartera (escalares ya detached) y lo pasa como return_buffer. El
        gradiente fluye ÚNICAMENTE por el retorno de hoy (port_ret),
        calculado con los pesos actuales de la red; el buffer son escalares
        históricos sin grafo.

    Args:
        risk_free_rate  : Tasa libre de riesgo diaria (defecto 0.0).
        turnover_lambda : Peso de la penalización L1 de rotación (defecto 0.0015).
        anchor_lambda   : Peso de la penalización de anclaje a Equal-Weight
                          (defecto 0.0, desactivado).
    """

    def __init__(self, risk_free_rate=0.0, turnover_lambda=0.0015, anchor_lambda=0.0):
        super(WindowedSharpeLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.turnover_lambda = turnover_lambda
        self.anchor_lambda = anchor_lambda

    def forward(self, weights, returns, cov_matrix=None, prev_weights=None, return_buffer=None):
        """Calcula el Sharpe negativo sobre la ventana de retornos.

        Args:
            weights: Tensor (N,) con los pesos predichos (requires_grad=True).
            returns: Tensor (N,) con los retornos reales futuros de HOY.
            cov_matrix: No se usa en esta loss; admitido por compatibilidad de firma.
            prev_weights: Tensor (N,) pesos del paso anterior, ya detached (opcional).
            return_buffer: Lista/tensor de floats detached con retornos históricos
                de la ventana (opcional).

        Returns:
            Tensor escalar diferenciable respecto a weights.
        """
        port_ret = torch.dot(weights, returns)

        if return_buffer is not None and len(return_buffer) > 0:
            history = torch.as_tensor(return_buffer, dtype=port_ret.dtype, device=port_ret.device)
            serie = torch.cat([history, port_ret.unsqueeze(0)])
        else:
            serie = port_ret.unsqueeze(0)

        sharpe = (serie.mean() - self.risk_free_rate) / (serie.std(unbiased=False) + 1e-6)
        base = -sharpe

        if prev_weights is not None:
            base = base + self.turnover_lambda * torch.sum(torch.abs(weights - prev_weights))

        if self.anchor_lambda > 0.0:
            N = weights.shape[0]
            ew = torch.full_like(weights, 1.0 / N)
            base = base + self.anchor_lambda * torch.sum(torch.abs(weights - ew))

        return base


class MinVarianceLoss(nn.Module):
    """Función de pérdida orientada al riesgo: minimiza la varianza de cartera
    w^T Σ w. Opcionalmente añade penalizaciones de rotación y anclaje
    a Equal-Weight.

    Args:
        turnover_lambda : Peso de la penalización L1 de rotación (defecto 0.0015).
        anchor_lambda   : Peso de la penalización de anclaje a Equal-Weight
                          (defecto 0.0, desactivado).
    """

    def __init__(self, turnover_lambda=0.0015, anchor_lambda=0.0):
        super(MinVarianceLoss, self).__init__()
        self.turnover_lambda = turnover_lambda
        self.anchor_lambda = anchor_lambda

    def forward(self, weights, returns=None, cov_matrix=None, prev_weights=None):
        """Calcula la varianza de cartera más las penalizaciones opcionales.

        Args:
            weights: Tensor (N,) con los pesos predichos (requires_grad=True).
            returns: Tensor (N,) con los retornos; solo se usa en el fallback
                cuando cov_matrix es None.
            cov_matrix: Tensor (N, N) matriz de covarianza (preferido).
            prev_weights: Tensor (N,) pesos del paso anterior, ya detached (opcional).

        Returns:
            Tensor escalar diferenciable respecto a weights.

        Raises:
            ValueError: Si no se proporciona ni cov_matrix ni returns.
        """
        if cov_matrix is not None:
            port_var = torch.matmul(weights, torch.matmul(cov_matrix, weights))
        else:
            if returns is None:
                raise ValueError("MinVarianceLoss necesita cov_matrix o returns para calcular la varianza.")
            weighted_rets = weights * returns
            port_var = torch.var(weighted_rets, unbiased=False) + 1e-6

        base = port_var

        if prev_weights is not None:
            base = base + self.turnover_lambda * torch.sum(torch.abs(weights - prev_weights))

        if self.anchor_lambda > 0.0:
            N = weights.shape[0]
            ew = torch.full_like(weights, 1.0 / N)
            base = base + self.anchor_lambda * torch.sum(torch.abs(weights - ew))

        return base


class CVaRLoss(nn.Module):
    """Función de pérdida orientada al riesgo de cola: minimiza el Expected
    Shortfall (CVaR) paramétrico gaussiano de la cartera.

        ES_alpha = -mu_p + c * sigma_p,    c = phi(Phi^{-1}(alpha)) / (1 - alpha)

    donde mu_p = w·returns (retorno esperado de hoy) y
    sigma_p = sqrt(w^T Σ w) (volatilidad de cartera). El factor c es la
    constante del Expected Shortfall gaussiano (densidad normal evaluada en
    el cuantil alpha, normalizada por la masa de cola 1-alpha) y se
    precalcula una sola vez en __init__ con scipy.stats.norm.

    Minimizar ES equivale a maximizar el retorno esperado penalizando
    fuertemente el riesgo de cola (a diferencia de MinVarianceLoss, que
    ignora el retorno).

    Args:
        alpha           : Nivel de confianza del CVaR (defecto 0.95).
        turnover_lambda : Peso de la penalización L1 de rotación (defecto 0.0015).
        anchor_lambda   : Peso de la penalización de anclaje a Equal-Weight
                          (defecto 0.0, desactivado).
    """

    def __init__(self, alpha=0.95, turnover_lambda=0.0015, anchor_lambda=0.0):
        super(CVaRLoss, self).__init__()
        self.alpha = alpha
        self.c = float(norm.pdf(norm.ppf(alpha)) / (1.0 - alpha))
        self.turnover_lambda = turnover_lambda
        self.anchor_lambda = anchor_lambda

    def forward(self, weights, returns, cov_matrix=None, prev_weights=None):
        """Calcula el Expected Shortfall gaussiano más las penalizaciones opcionales.

        Args:
            weights: Tensor (N,) con los pesos predichos (requires_grad=True).
            returns: Tensor (N,) con los retornos reales futuros de hoy.
            cov_matrix: Tensor (N, N) covarianza (preferido para sigma_p).
            prev_weights: Tensor (N,) pesos del paso anterior, ya detached (opcional).

        Returns:
            Tensor escalar diferenciable respecto a weights.
        """
        mu_p = torch.dot(weights, returns)

        if cov_matrix is not None:
            sigma_p = torch.sqrt(torch.matmul(weights, torch.matmul(cov_matrix, weights)) + 1e-8)
        else:
            sigma_p = torch.std(weights * returns) + 1e-8

        es = -mu_p + self.c * sigma_p
        base = es

        if prev_weights is not None:
            base = base + self.turnover_lambda * torch.sum(torch.abs(weights - prev_weights))

        if self.anchor_lambda > 0.0:
            N = weights.shape[0]
            ew = torch.full_like(weights, 1.0 / N)
            base = base + self.anchor_lambda * torch.sum(torch.abs(weights - ew))

        return base


if __name__ == "__main__":
    loss_fn = DifferentiableSharpeLoss()
    w = torch.tensor([0.6, 0.4])
    r = torch.tensor([0.02, -0.01])
    print(f"[DifferentiableSharpeLoss] Loss de prueba (Negative Sharpe): {loss_fn(w, r).item():.4f}")

    import torch

    N = 5
    torch.manual_seed(0)

    raw_w = torch.rand(N)
    weights = (raw_w / raw_w.sum()).detach().requires_grad_(True)

    returns_t = torch.randn(N) * 0.01
    cov = torch.eye(N) * 0.0001

    raw_pw = torch.rand(N)
    prev_weights = (raw_pw / raw_pw.sum()).detach()

    fn_normal = TurnoverAwareSharpeLoss(turnover_lambda=0.0015)
    loss_val = fn_normal(weights, returns_t, cov_matrix=cov, prev_weights=prev_weights)
    is_scalar = (loss_val.dim() == 0)
    print(f"\n[TurnoverAwareSharpeLoss] CHECK (a) — tensor escalar: {is_scalar}  | valor: {loss_val.item():.6f}")
    assert is_scalar, "FAIL: la loss no es escalar"

    loss_val.backward()
    grad_ok = (weights.grad is not None)
    print(f"[TurnoverAwareSharpeLoss] CHECK (b) — weights.grad not None: {grad_ok}  | grad norm: {weights.grad.norm().item():.6f}")
    assert grad_ok, "FAIL: weights.grad es None tras backward()"

    weights2 = (raw_w / raw_w.sum()).detach().requires_grad_(True)
    fn_zero   = TurnoverAwareSharpeLoss(turnover_lambda=0.0)
    fn_high   = TurnoverAwareSharpeLoss(turnover_lambda=1.0)

    loss_zero = fn_zero(weights2, returns_t, cov_matrix=cov, prev_weights=prev_weights)
    loss_high = fn_high(weights2, returns_t, cov_matrix=cov, prev_weights=prev_weights)

    penalty_acts = (loss_high.item() > loss_zero.item())
    print(f"[TurnoverAwareSharpeLoss] CHECK (c) — penalización actúa (lambda=1.0 > lambda=0.0): {penalty_acts}")
    print(f"   loss con lambda=0.0 : {loss_zero.item():.6f}")
    print(f"   loss con lambda=1.0 : {loss_high.item():.6f}")
    assert penalty_acts, "FAIL: la penalización de rotación no incrementa la loss"

    print("\n=== TODOS LOS CHECKS PASADOS ===")

    print("\n" + "="*60)
    print("Tests WindowedSharpeLoss")
    print("="*60)

    N = 5
    torch.manual_seed(42)

    raw_w3 = torch.rand(N)
    w3 = (raw_w3 / raw_w3.sum()).detach().requires_grad_(True)
    r3 = torch.randn(N) * 0.01

    buf_stable = [0.008, 0.009, 0.010, 0.008, 0.009, 0.010, 0.008, 0.009, 0.010, 0.009]

    fn_win = WindowedSharpeLoss(risk_free_rate=0.0, turnover_lambda=0.0)

    loss_win = fn_win(w3, r3, return_buffer=buf_stable)
    is_scalar_win = (loss_win.dim() == 0)
    print(f"\n[WindowedSharpeLoss] CHECK (a) — tensor escalar: {is_scalar_win}  | valor: {loss_win.item():.6f}")
    assert is_scalar_win, "FAIL: WindowedSharpeLoss no devuelve escalar"

    loss_win.backward()
    grad_ok_win = (w3.grad is not None)
    print(f"[WindowedSharpeLoss] CHECK (b) — weights.grad not None: {grad_ok_win}  | grad norm: {w3.grad.norm().item():.6f}")
    assert grad_ok_win, "FAIL: weights.grad es None tras backward() en WindowedSharpeLoss"

    buf_volatile = [0.05, -0.04, 0.06, -0.05, 0.07, -0.06, 0.05, -0.04, 0.06, -0.05]

    raw_w3b = torch.rand(N)
    w3_stable = (raw_w3b / raw_w3b.sum()).detach().requires_grad_(True)
    w3_volatile = (raw_w3b / raw_w3b.sum()).detach().requires_grad_(True)
    r3_neutral = torch.zeros(N)

    loss_stable   = fn_win(w3_stable,   r3_neutral, return_buffer=buf_stable)
    loss_volatile = fn_win(w3_volatile, r3_neutral, return_buffer=buf_volatile)

    coherent_win = (loss_stable.item() < loss_volatile.item())
    print(f"[WindowedSharpeLoss] CHECK (c) — buffer estable da MENOR loss que volátil: {coherent_win}")
    print(f"   loss con buffer estable  : {loss_stable.item():.6f}")
    print(f"   loss con buffer volátil  : {loss_volatile.item():.6f}")
    assert coherent_win, "FAIL: buffer estable debería dar menor loss (mejor Sharpe)"

    print("\n=== WindowedSharpeLoss — TODOS LOS CHECKS PASADOS ===")

    print("\n" + "="*60)
    print("Tests MinVarianceLoss")
    print("="*60)

    N = 4
    torch.manual_seed(7)

    variances = torch.tensor([0.10, 0.01, 0.01, 0.01])
    cov4 = torch.diag(variances)

    w_concentrated_raw = torch.tensor([0.97, 0.01, 0.01, 0.01])
    w_concentrated = (w_concentrated_raw / w_concentrated_raw.sum()).detach().requires_grad_(True)

    w_diversified_raw = torch.ones(N)
    w_diversified = (w_diversified_raw / w_diversified_raw.sum()).detach().requires_grad_(True)

    r4 = torch.randn(N) * 0.01
    fn_minvar = MinVarianceLoss(turnover_lambda=0.0)

    loss_conc = fn_minvar(w_concentrated, returns=r4, cov_matrix=cov4)
    is_scalar_mv = (loss_conc.dim() == 0)
    print(f"\n[MinVarianceLoss] CHECK (a) — tensor escalar: {is_scalar_mv}  | valor: {loss_conc.item():.6f}")
    assert is_scalar_mv, "FAIL: MinVarianceLoss no devuelve escalar"

    loss_conc.backward()
    grad_ok_mv = (w_concentrated.grad is not None)
    print(f"[MinVarianceLoss] CHECK (b) — weights.grad not None: {grad_ok_mv}  | grad norm: {w_concentrated.grad.norm().item():.6f}")
    assert grad_ok_mv, "FAIL: weights.grad es None tras backward() en MinVarianceLoss"

    loss_div = fn_minvar(w_diversified, returns=r4, cov_matrix=cov4)
    coherent_mv = (loss_conc.item() > loss_div.item())
    print(f"[MinVarianceLoss] CHECK (c) — concentrada da MAYOR loss que diversificada: {coherent_mv}")
    print(f"   loss cartera concentrada  (w~[1,0,0,0]): {loss_conc.item():.6f}")
    print(f"   loss cartera diversificada (EW):          {loss_div.item():.6f}")
    assert coherent_mv, "FAIL: cartera concentrada en activo de alta var debería dar mayor loss"

    print("\n=== MinVarianceLoss — TODOS LOS CHECKS PASADOS ===")

    print("\n" + "="*60)
    print("Tests CVaRLoss")
    print("="*60)

    fn_cvar_info = CVaRLoss(alpha=0.95)
    print(f"\n[CVaRLoss] Constante c precalculada para alpha=0.95: {fn_cvar_info.c:.6f}")

    N = 4
    torch.manual_seed(7)

    variances_c = torch.tensor([0.10, 0.01, 0.01, 0.01])
    cov_c = torch.diag(variances_c)

    w_conc_raw_c = torch.tensor([0.97, 0.01, 0.01, 0.01])
    w_concentrated_c = (w_conc_raw_c / w_conc_raw_c.sum()).detach().requires_grad_(True)

    r_c = torch.randn(N) * 0.01
    fn_cvar = CVaRLoss(alpha=0.95, turnover_lambda=0.0)

    loss_cvar_conc = fn_cvar(w_concentrated_c, r_c, cov_matrix=cov_c)
    is_scalar_cvar = (loss_cvar_conc.dim() == 0)
    print(f"\n[CVaRLoss] CHECK (a) — tensor escalar: {is_scalar_cvar}  | valor: {loss_cvar_conc.item():.6f}")
    assert is_scalar_cvar, "FAIL: CVaRLoss no devuelve escalar"

    loss_cvar_conc.backward()
    grad_ok_cvar = (w_concentrated_c.grad is not None)
    print(f"[CVaRLoss] CHECK (b) — weights.grad not None: {grad_ok_cvar}  | grad norm: {w_concentrated_c.grad.norm().item():.6f}")
    assert grad_ok_cvar, "FAIL: weights.grad es None tras backward() en CVaRLoss"

    r_equal = torch.tensor([0.01, 0.01])
    cov_diff_var = torch.diag(torch.tensor([0.20, 0.01]))

    w_high_var = torch.tensor([1.0, 0.0], requires_grad=True)
    w_low_var = torch.tensor([0.0, 1.0], requires_grad=True)

    loss_high_var = fn_cvar(w_high_var, r_equal, cov_matrix=cov_diff_var)
    loss_low_var = fn_cvar(w_low_var, r_equal, cov_matrix=cov_diff_var)

    coherent_cvar = (loss_high_var.item() > loss_low_var.item())
    print(f"[CVaRLoss] CHECK (c) — misma rentabilidad esperada, cartera en activo de MAYOR varianza da ES MAYOR: {coherent_cvar}")
    print(f"   ES (loss) concentrada en activo de ALTA varianza (var=0.20): {loss_high_var.item():.6f}")
    print(f"   ES (loss) concentrada en activo de BAJA varianza (var=0.01): {loss_low_var.item():.6f}")
    assert coherent_cvar, "FAIL: la cartera en el activo de mayor varianza debería dar mayor ES"

    print("\n=== CVaRLoss — TODOS LOS CHECKS PASADOS ===")
    print("\n" + "="*60)
    print("=== TODOS LOS TESTS PASADOS (5/5 clases) ===")
    print("="*60)
