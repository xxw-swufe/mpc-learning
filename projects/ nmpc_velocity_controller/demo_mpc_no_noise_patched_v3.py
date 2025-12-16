"""
demo_mpc_no_noise_patched.py

对原 demo_mpc_no_noise.py 的“更稳健可运行”修复版，主要改动：
1) 更稳健的导入：无论你从哪个工作目录运行，都能找到 GPT/ 里的求解器文件。
2) 更稳健的动力学调用：兼容 f_dyn(x,u) 或 f_dyn(x,u,p) 两种签名；兼容 numpy / CasADi DM 输入。
3) 参数获取更稳健：优先用公开方法，其次再退回私有 _get_default_parameters；都没有则 p=None。
4) 一些 shape/类型上的小修补，减少 CasADi/numpy 混用导致的报错。

运行方式（推荐在项目根目录）：
    python GPT/demo_mpc_no_noise_patched.py
或
    python demo_mpc_no_noise_patched.py   # 只要文件所在目录结构没变，也能跑
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# ----------------------------
# 1) 让导入对“运行目录”不敏感
# ----------------------------
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent  # 该文件所在目录
# 允许本文件放在 GPT/ 里或根目录里：都尝试把“根目录”和“GPT/”加入 sys.path
_CANDIDATES = [
    _ROOT,
    _ROOT / "GPT",
    _ROOT.parent,
    _ROOT.parent / "GPT",
]
for p in _CANDIDATES:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from nmpc_ipopt_centroidal_fixed import CentroidalNMPC_IPOPT_Fixed
except ImportError as e:
    raise ImportError(
        "无法导入 CentroidalNMPC_IPOPT_Fixed。请确认目录结构包含 GPT/nmpc_ipopt_centroidal_fixed.py，"
        "以及你运行脚本时没有把文件移动到别处。原始错误：\n" + str(e)
    )

# 可选：CasADi（如果你的求解器内部用的是 CasADi Function，一般都会装）
try:
    import casadi as ca  # type: ignore
except Exception:
    ca = None


def _call_dyn(f_dyn, x: np.ndarray, u: np.ndarray, p):
    """更稳健地调用动力学：支持 f(x,u) 或 f(x,u,p)；支持 CasADi / numpy。"""
    # CasADi Function 有 n_in()；普通 python callable 可能没有
    n_in = getattr(f_dyn, "n_in", None)
    if callable(n_in):
        nin = int(f_dyn.n_in())
    else:
        # 猜：优先带参数
        nin = 3 if p is not None else 2

    # CasADi DM 输入更稳（如果可用）
    if ca is not None:
        x_in = ca.DM(x)
        u_in = ca.DM(u)
        if nin >= 3:
            dx = f_dyn(x_in, u_in, p)
        else:
            dx = f_dyn(x_in, u_in)
        return np.array(dx).squeeze()

    # 纯 numpy/数值 callable
    if nin >= 3:
        return np.array(f_dyn(x, u, p)).squeeze()
    return np.array(f_dyn(x, u)).squeeze()


def rk4_step(f_dyn, x: np.ndarray, u: np.ndarray, p, dt: float):
    """对连续时间动力学 xdot=f(x,u,p) 做一步 RK4（仅当 f_dyn 返回 xdot 时才用）。"""
    x = np.asarray(x).reshape(-1)
    u = np.asarray(u).reshape(-1)

    k1 = _call_dyn(f_dyn, x, u, p)
    k2 = _call_dyn(f_dyn, x + 0.5 * dt * k1, u, p)
    k3 = _call_dyn(f_dyn, x + 0.5 * dt * k2, u, p)
    k4 = _call_dyn(f_dyn, x + dt * k3, u, p)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def plant_step(nmpc, x: np.ndarray, u: np.ndarray, p, contact_now: np.ndarray | None = None):
    """推进 plant 一步（关键：不要对离散的 f_dyn 再做 RK4！）

    你的 nmpc_ipopt_centroidal_fixed.py 明确把 nmpc.f_dyn 定义为：
        f_dyn(x,u,p) -> x_next
    且内部已经做过 RK4 离散化。

    因此这里正确的推进方式就是：
        x_next = f_dyn(x,u,p)

    为了后续迁移到“含步态/接触变化”的版本，这里支持把 contact_now 写入参数 p 的前 4 维：
        p = [contact(4), other_params(25)]
    """
    f_dyn = getattr(nmpc, "f_dyn", None)
    if f_dyn is None:
        raise AttributeError("nmpc 对象没有 f_dyn（动力学函数）。请检查你的 NMPC 实现。")

    if p is None:
        return _call_dyn(f_dyn, x, u, p)

    p_arr = np.asarray(p).reshape(-1).copy()
    if contact_now is not None and p_arr.size >= 4:
        p_arr[:4] = np.asarray(contact_now).reshape(4)

    # 关键：这里的 _call_dyn 返回的就是 x_next（因为 f_dyn 已是离散映射）
    x_next = _call_dyn(f_dyn, x, u, p_arr)
    return np.asarray(x_next).reshape(-1)


def _infer_mass_g(p, default_m=15.019, default_g=9.81):
    """尽量从参数里推断质量/重力（如果 p 是 dict 或含这些字段的结构）。"""
    m, g = default_m, default_g
    if isinstance(p, dict):
        for key in ["m", "mass", "robot_mass"]:
            if key in p:
                m = float(p[key])
                break
        for key in ["g", "gravity"]:
            if key in p:
                g = float(p[key])
                break
    return m, g


def make_reference_traj(x_now: np.ndarray, v_cmd: np.ndarray, z_cmd: float, nmpc, p):
    """构造渐进式 reference（避免阶跃响应导致的速度失控）：

    - 速度参考：从当前速度渐进变化到目标速度
    - COM 高度 = z_cmd
    - 使用时间常数避免过度补偿
    """
    N = int(nmpc.horizon)
    nx, nu = int(nmpc.nx), int(nmpc.nu)
    dt = float(nmpc.dt)

    x_ref = np.repeat(np.asarray(x_now).reshape(nx, 1), N + 1, axis=1)
    u_ref = np.zeros((nu, N))

    # 当前速度
    v_now = x_now[3:6]

    # 关键：使用渐进的参考速度（一阶响应）
    tau = 1.0  # 增加时间常数，进一步降低加速度
    for k in range(N + 1):
        alpha = 1.0 - np.exp(-k * dt / tau)  # 渐进因子 [0, 1)，更缓慢
        # 速度参考：从当前速度渐进到目标速度
        v_ref_k = v_now + (v_cmd - v_now) * alpha
        x_ref[3:6, k] = v_ref_k

        # 位置参考：基于参考速度积分
        if k == 0:
            x_ref[0:3, k] = x_now[0:3]
        else:
            # 使用梯形积分计算位置
            v_avg = (x_ref[3:6, k-1] + v_ref_k) / 2.0
            x_ref[0:3, k] = x_ref[0:3, k-1] + v_avg * dt

        x_ref[2, k] = z_cmd  # 固定高度
        # 期望姿态/角速度为 0
        x_ref[6:12, k] = 0.0

    # 控制参考：
    # 常见：u[0:12]=4足端速度, u[12:24]=4足端力（GRF）
    # 这里给一个"站立力分配"参考：每只脚 fz=mg/4
    m, g = _infer_mass_g(p)
    fz = m * g / 4.0
    if nu >= 24:
        for k in range(N):
            for leg in range(4):
                u_ref[12 + 3 * leg + 2, k] = fz

    return x_ref, u_ref


def shift_warmstart(w_opt: np.ndarray | None, nmpc):
    """把上一次解 shift 一格当作下一次初值（MPC warm-start）。"""
    if w_opt is None:
        return None

    w_opt = np.asarray(w_opt).reshape(-1)
    nx, nu, N = int(nmpc.nx), int(nmpc.nu), int(nmpc.horizon)
    nx_block = nx * (N + 1)
    if w_opt.size < nx_block:
        # 解向量结构不匹配，直接不 warm-start
        return None

    X = w_opt[:nx_block].reshape((nx, N + 1), order="F")
    U = w_opt[nx_block:].reshape((nu, N), order="F") if w_opt.size == nx_block + nu * N else None

    X0 = np.hstack([X[:, 1:], X[:, -1:]])
    if U is None:
        return X0.reshape(-1, order="F")
    U0 = np.hstack([U[:, 1:], U[:, -1:]])
    return np.concatenate([X0.reshape(-1, order="F"), U0.reshape(-1, order="F")])


def _get_parameters(nmpc):
    """尽量拿到求解器/模型默认参数。"""
    for name in ["get_default_parameters", "get_parameters", "_get_default_parameters"]:
        fn = getattr(nmpc, name, None)
        if callable(fn):
            return fn()
    return None


def main():
    # =============================
    # 1) 初始化 NMPC
    # =============================
    nmpc = CentroidalNMPC_IPOPT_Fixed(horizon=10, dt=0.02)
    p = _get_parameters(nmpc)

    # =============================
    # PATCH A: 按 nx=30 的 state 顺序初始化 + 打印权重 + 放大 Δu 正则
    # state 顺序（你这套模型里常见是）：
    # 0:3   com_pos
    # 3:6   com_vel
    # 6:9   rpy
    # 9:12  omega
    # 12:15 foot_FL
    # 15:18 foot_FR
    # 18:21 foot_RL
    # 21:24 foot_RR
    # 24:30 integrals(6)
    # =============================

    # 1) 初始化状态：COM 高度 + 足端位置（很重要：不要全 0）
    x = np.zeros(int(nmpc.nx))
    x[2] = 0.30  # COM z

    # 一个"合理站立"的足端初值（机体坐标系/世界坐标系按你模型定义；至少别是全 0）
    # 这组数在你 nmpc_ipopt_centroidal_fixed.py 文件自测里也用过
    foot_FL = np.array([ 0.20,  0.15, -0.30])
    foot_FR = np.array([ 0.20, -0.15, -0.30])
    foot_RL = np.array([-0.20,  0.15, -0.30])
    foot_RR = np.array([-0.20, -0.15, -0.30])

    x[12:15] = foot_FL
    x[15:18] = foot_FR
    x[18:21] = foot_RL
    x[21:24] = foot_RR

    # 积分态清零（避免风up从一开始就影响）
    x[24:30] = 0.0

    print("\n=== State Init Check ===")
    print("com_z =", x[2])
    print("foot_FL =", x[12:15], "foot_FR =", x[15:18])
    print("foot_RL =", x[18:21], "foot_RR =", x[21:24])
    print("integrals(6) =", x[24:30])

    # 2) 打印权重矩阵的关键段
    print("\n=== Weight Matrix Diagnostics ===")
    Q_diag = np.diag(nmpc.Q)
    R_diag = np.diag(nmpc.R)
    print("Q diag last 6 (integrals):", Q_diag[-6:])
    print("Q diag [0:12] (pos/vel/rpy/omega):", Q_diag[:12])
    print("R diag [0:12] (foot vel):", R_diag[:12])
    print("R diag [12:24] (GRF):", R_diag[12:24])

    # 3) 放大 Δu 正则，并"重建 solver"（否则不生效）
    # 你当前 nmpc_ipopt_centroidal_fixed.py 里 Rdu = 0.1 * diag(R)
    # 这里用 factor = 10 相当于变成 1.0*diag(R) （先保守点）
    try:
        import casadi as cs
        factor = 500.0
        nmpc.Rdu = cs.diag((0.1 * factor) * np.diag(nmpc.R))
        nmpc.solver = nmpc._build_solver()
        print("\n=== Δu Regularization Updated ===")
        print(f"Rdu scaled: old=0.1*R, new={(0.1*factor):.3f}*R (and solver rebuilt)")
    except Exception as e:
        print("\n[WARN] Failed to rebuild solver after Rdu change:", e)

    # =============================
    # 2) 其他初始化
    # =============================
    # 命令：原地站立
    v_cmd = np.array([0.2, 0.0, 0.0], dtype=float)
    v_cmd_target = v_cmd.copy()  # 目标速度
    z_cmd = 0.30

    # 接触序列：最小 demo 先全支撑（4xN 全 1）
    contact_seq = np.ones((4, int(nmpc.horizon)), dtype=int)

    # 记录
    T = 3.0
    steps = int(T / float(nmpc.dt))

    # 平滑加速时间：1秒内从0加速到目标速度
    T_RAMP = 1.0

    w0 = None
    vel_err_hist = []
    u_prev = None  # 用于计算控制变化

    print(f"nx={nmpc.nx}, nu={nmpc.nu}, horizon={nmpc.horizon}, dt={nmpc.dt}")

    # 检查权重设置
    print("\n=== Cost Function Weights ===")
    print("Qzz =", nmpc.Q[2,2], "Qvz =", nmpc.Q[5,5])
    print("R_fz (FL) =", nmpc.R[14,14], "(should be u[12+2] = force FL_z)")
    print("R_force_diagonal:", np.diag(nmpc.R)[12:24])  # 所有力元素的权重
    print("\nExpected: Qzz and Qvz should be > 0 to track height")
    print("Expected: R_fz should be reasonable (not too large)\n")

    # =============================
    # 3) MPC 循环：predict → optimize → control → plant step
    # =============================
    try:
        for t in range(steps):
            # 平滑速度参考：从 0 线性爬升到 v_cmd_target
            alpha = min(1.0, (t * float(nmpc.dt)) / T_RAMP)
            v_cmd_now = alpha * v_cmd_target
            x_ref, u_ref = make_reference_traj(x, v_cmd_now, z_cmd, nmpc, p)

            # Reference sanity check
            if t % 10 == 0:
                print("\n=== REF sanity ===")
                print("x_ref com_vel(k=0):", x_ref[3:6,0], "  x_ref com_vel(k=1):", x_ref[3:6,1])
                print("x_ref com_z range:", float(x_ref[2,:].min()), float(x_ref[2,:].max()))

            # solve（注意：你的 solve() 形参名可能不同；这里按原 demo 的调用）
            res = nmpc.solve(x, x_ref, u_ref, contact_seq, w0=w0)
            if res.get("status", "") != "success":
                print(f"\n=== MPC Solve Failed at step {t} ===")
                print(f"Solver status: {res.get('status', 'unknown')}")
                print(f"Solver message: {res.get('return_status', 'no message')}")
                if 'error' in res:
                    print(f"Error: {res['error']}")
                # 检查是否有其他统计信息
                if 'solver_stats' in res:
                    stats = res['solver_stats']
                    print(f"Iterations: {stats.get('iter_count', 'N/A')}")
                    print(f"Exit status: {stats.get('exit_status', 'N/A')}")
                raise RuntimeError(f"MPC solve failed at step {t}: {res.get('error', res)}")

            u0 = np.asarray(res["u0"]).reshape(-1)

            # === 打印足端地面反力（GRF） ===
            F = u0[12:24].reshape(4, 3)
            print("GRF per leg [fx, fy, fz] (N):")
            print(np.round(F, 2))
            print("Sum Fz =", np.round(F[:, 2].sum(), 2), f"(expected ~{15.019 * 9.81:.1f} N)")

            # Friction cone slack check
            mu = float(np.asarray(p).reshape(-1)[4]) if p is not None else 0.5
            fx, fy, fz = F[:,0], F[:,1], F[:,2]
            slack_fx = mu*fz - np.abs(fx)
            slack_fy = mu*fz - np.abs(fy)
            print("mu =", mu)
            print("min slack_fx, slack_fy =", float(slack_fx.min()), float(slack_fy.min()))
            print("fz min/max =", float(fz.min()), float(fz.max()))

            # Control change magnitude check
            if t > 0 and u_prev is not None:
                du = u0 - u_prev
                print("||du|| =", float(np.linalg.norm(du)), "  ||du_force|| =", float(np.linalg.norm(du[12:24])))

            # plant step：用同一个动力学滚动
            f_dyn = getattr(nmpc, "f_dyn", None)
            if f_dyn is None:
                raise AttributeError("nmpc 对象没有 f_dyn（动力学函数）。请检查你的 NMPC 实现。")
            x = plant_step(nmpc, x, u0, p, contact_now=contact_seq[:, 0])

            # Save previous control for du calculation
            u_prev = u0.copy()

            # warm-start
            w0 = shift_warmstart(res.get("w_opt", None), nmpc)

            # Dynamics consistency check
            if t % 10 == 0:
                print("state com_vel =", x[3:6], " com_z =", x[2])

            # 简单监控：COM 速度误差（默认认为 x[3:6] 是线速度）
            if x.size >= 6:
                v_now = x[3:6]
                err = v_now - (v_cmd_now if 'v_cmd_now' in locals() else v_cmd)
                vel_err_hist.append(float(np.linalg.norm(err)))
            else:
                vel_err_hist.append(float("nan"))

            if t % 10 == 0 and x.size >= 6:
                print(
                    f"t={t*float(nmpc.dt):5.2f}s | v_now={np.round(v_now, 3)} | v_ref={np.round(v_cmd, 3)} | |e|={vel_err_hist[-1]:.4f}"
                )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，已停止仿真（正常退出）。")
        return

    vel_err_hist = np.asarray(vel_err_hist, dtype=float)
    print("\nDone.")
    if np.all(np.isfinite(vel_err_hist)):
        print(f"Mean |v_err|: {vel_err_hist.mean():.4f}  Max |v_err|: {vel_err_hist.max():.4f}")
    else:
        print("速度误差里包含 NaN（可能是状态维度/排列与你的模型不一致）。")


if __name__ == "__main__":
    main()
