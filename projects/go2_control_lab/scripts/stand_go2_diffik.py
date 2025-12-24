#!/usr/bin/env python3
import argparse
import torch
from isaaclab.app import AppLauncher

# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description="Go2 Stand + Diff-IK (stable version: q_cmd anchored to q_nom)")
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--num-envs", type=int, default=1)

# footprint target (only used for slow shaping, optional)
parser.add_argument("--x-front", type=float, default=0.25)
parser.add_argument("--x-rear", type=float, default=-0.25)
parser.add_argument("--y", type=float, default=0.14)

# IK params (defaults already usable; CLI is for快速调参，不传就用默认)
parser.add_argument("--alpha", type=float, default=0.03, help="IK integration step (smaller = more stable)")
parser.add_argument("--dls-lambda", type=float, default=0.10, help="DLS damping")
parser.add_argument("--dq-max", type=float, default=0.03, help="per-step clamp on dq (rad)")

# posture anchoring (very important!)
parser.add_argument("--post-kp", type=float, default=0.01, help="pull q_cmd back to q_nom each step")

# timing / safety
parser.add_argument("--settle-sec", type=float, default=1.0, help="hold q_nom before enabling IK")
parser.add_argument("--min-base-z", type=float, default=0.20, help="reset if base lower than this")
parser.add_argument("--print-every", type=int, default=500)

# optional: slowly reshape to a rectangle after settle
parser.add_argument("--enable-shape", action="store_true")
parser.add_argument("--shape-rate", type=float, default=0.001, help="lerp rate towards target rectangle (very small)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# IMPORTANT: import after Isaac app starts
import sys, os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from envs.stand_env import StandEnv
from isaaclab.utils.math import quat_apply

def main():
    env = StandEnv.make(num_envs=args.num_envs, dt=args.dt, device=args.device)
    robot = env.robot.robot  # Articulation

    body_names = robot.data.body_names
    bn2id = {n: i for i, n in enumerate(body_names)}
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_ids = [bn2id[n] for n in foot_names]

    # your confirmed joint order
    leg_joint_ids = {
        "FL": [0, 4, 8],
        "FR": [1, 5, 9],
        "RL": [2, 6, 10],
        "RR": [3, 7, 11],
    }
    legs = ["FL", "FR", "RL", "RR"]

    # Jacobian layout
    J_all = robot.root_physx_view.get_jacobians()
    nb_jac = J_all.shape[1]
    nb_body = len(body_names)
    if nb_jac == nb_body:
        body_jac_index = lambda bid: bid
    elif nb_jac == nb_body - 1:
        body_jac_index = lambda bid: bid - 1
    else:
        raise RuntimeError(f"Unexpected jacobian body dim: {nb_jac}, body_names: {nb_body}")

    nj = robot.data.joint_pos.shape[1]
    last_dim = J_all.shape[-1]
    if last_dim == nj:
        col_offset = 0
    elif last_dim == nj + 6:
        col_offset = 6
    else:
        col_offset = 6 if last_dim > nj else 0

    device = robot.data.joint_pos.device
    I3 = torch.eye(3, device=device).unsqueeze(0)

    # ------------------ Nominal standing pose (use your stable one!) ------------------
    q_nom = torch.tensor(
        [[
            0.10, -0.10,  0.10, -0.10,   # hips
            0.65,  0.65,  0.90,  0.90,   # thighs
           -1.25, -1.25, -1.25, -1.25,   # calves
        ]],
        device=device, dtype=torch.float32
    )

    # command that we will keep & integrate (IMPORTANT: do NOT chase q_now)
    q_cmd = q_nom.clone()

    settle_steps = int(args.settle_sec / args.dt)
    print(f"=== Debug Info ===")
    print(f"dt: {args.dt} device: {args.device}")
    print(f"joint_names: {robot.data.joint_names}")
    print(f"body_names: {body_names}")
    print(f"J_all.shape: {tuple(J_all.shape)} col_offset: {col_offset} nj: {nj}")
    print(f"settle_steps: {settle_steps} min_base_z: {args.min_base_z}")
    print("========================================\n")

    # ---- settle: hold q_nom ----
    for _ in range(settle_steps):
        env.set_actions(q_nom)
        env.step()

    # lock initial feet targets in world (after settle)
    feet_des_w = robot.data.body_pos_w[:, foot_ids, :].clone()
    ground_z = feet_des_w[:, :, 2].min().item()
    feet_des_w[:, :, 2] = ground_z
    print(f"[LOCK] ground_z = {ground_z:.3f}")

    # optional shaping target rectangle in world
    if args.enable_shape:
        feet_des_b = torch.tensor(
            [
                [args.x_front,  args.y,  0.0],
                [args.x_front, -args.y,  0.0],
                [args.x_rear,   args.y,  0.0],
                [args.x_rear,  -args.y,  0.0],
            ],
            device=device, dtype=torch.float32
        ).unsqueeze(0)  # (1,4,3)

        root_pos = robot.data.root_pos_w.clone()    # (1,3)
        root_quat = robot.data.root_quat_w.clone()  # (1,4)
        rq = root_quat.repeat_interleave(4, dim=0)              # (4,4)
        vw = quat_apply(rq, feet_des_b.reshape(-1, 3)).reshape(1,4,3)
        target_feet_w = root_pos.unsqueeze(1) + vw
        target_feet_w[:, :, 2] = ground_z
    else:
        target_feet_w = None

    step = 0
    while simulation_app.is_running():
        # safety reset
        base_z = robot.data.root_pos_w[0, 2].item()
        if base_z < args.min_base_z:
            print(f"[RESET] base_z too low: {base_z:.3f} < {args.min_base_z:.3f}")
            env.reset()
            # re-settle & re-lock
            q_cmd = q_nom.clone()
            for _ in range(settle_steps):
                env.set_actions(q_nom)
                env.step()
            feet_des_w = robot.data.body_pos_w[:, foot_ids, :].clone()
            ground_z = feet_des_w[:, :, 2].min().item()
            feet_des_w[:, :, 2] = ground_z
            print(f"[LOCK] ground_z = {ground_z:.3f} (after reset)")
            step = 0
            continue

        # gradual shaping of locked feet targets (optional)
        if args.enable_shape and target_feet_w is not None:
            feet_des_w = (1.0 - args.shape_rate) * feet_des_w + args.shape_rate * target_feet_w
            feet_des_w[:, :, 2] = ground_z  # keep z projected

        # current feet
        feet_pos_w = robot.data.body_pos_w[:, foot_ids, :]  # (1,4,3)
        err_w = feet_des_w - feet_pos_w                     # (1,4,3)

        # IMPORTANT: in stance, usually only correct x/y; do NOT fight z with this minimal IK
        err_w[:, :, 2] = 0.0

        # jacobians
        J_all = robot.root_physx_view.get_jacobians()

        # integrate IK onto q_cmd (NOT q_now)
        for i, leg in enumerate(legs):
            bid = foot_ids[i]
            bj = body_jac_index(bid)
            jids = leg_joint_ids[leg]
            cols = [col_offset + j for j in jids]

            Jv = J_all[:, bj, 0:3, cols]              # (1,3,3) world translational jacobian
            e = err_w[:, i, :].unsqueeze(-1)          # (1,3,1)

            A = Jv @ Jv.transpose(-1, -2) + (args.dls_lambda ** 2) * I3
            dq = Jv.transpose(-1, -2) @ torch.linalg.solve(A, e)  # (1,3,1)
            dq = dq.squeeze(-1)                                    # (1,3)
            dq = torch.clamp(dq, -args.dq_max, args.dq_max)

            q_cmd[:, jids] = q_cmd[:, jids] + args.alpha * dq

        # posture anchoring: keep stiffness / prevent slow collapse
        q_cmd = q_cmd + args.post_kp * (q_nom - q_cmd)

        # apply
        env.set_actions(q_cmd)
        env.step()

        if step % args.print_every == 0:
            q_now = robot.data.joint_pos[0].detach()
            dq_norm = torch.norm((q_cmd[0] - q_now)).item()
            rl_err = err_w[0, 2].detach().cpu().numpy()
            fpw = robot.data.body_pos_w[0, foot_ids, :].detach().cpu().numpy()
            print(f"Step {step:5d} | base_z={base_z:.3f} | |q_cmd-q|={dq_norm:.4f}")
            print(f"  RL_err(x,y,z): {rl_err[0]:+.4f}, {rl_err[1]:+.4f}, {rl_err[2]:+.4f}")
            print("  feet_w:",
                  f"FL({fpw[0,0]:+.3f},{fpw[0,1]:+.3f},{fpw[0,2]:+.3f})",
                  f"FR({fpw[1,0]:+.3f},{fpw[1,1]:+.3f},{fpw[1,2]:+.3f})",
                  f"RL({fpw[2,0]:+.3f},{fpw[2,1]:+.3f},{fpw[2,2]:+.3f})",
                  f"RR({fpw[3,0]:+.3f},{fpw[3,1]:+.3f},{fpw[3,2]:+.3f})")
        step += 1

    simulation_app.close()

if __name__ == "__main__":
    main()
