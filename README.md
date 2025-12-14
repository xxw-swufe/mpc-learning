# Real-Time MPC Walkthrough

This repository documents my step-by-step learning and implementation of
Model Predictive Control (MPC) for legged robots, from basic state-space models
to real-time NMPC integrated with low-level control.

The goal of this project is not only to make MPC work, but to make it
**engineerable, real-time capable, and reproducible**.

---

## Project Roadmap

### Stage 1: MPC Foundations
- [ ] Continuous and discrete state-space modeling
- [ ] Linear MPC and LQR baseline
- [ ] Nonlinear dynamics modeling (centroidal / simplified legged model)
- [ ] System discretization and numerical stability
- [ ] CasADi symbolic modeling and automatic differentiation

### Stage 2: MPC Controller Construction
- [ ] Tracking cost and smoothness cost design
- [ ] Input and state constraints
- [ ] NMPC formulation using CasADi + IPOPT
- [ ] Predict–Optimize–Control loop
- [ ] Noise-free closed-loop simulation demo

### Stage 3: Real-Time MPC and Robot Interface
- [ ] 50–200 Hz MPC control loop
- [ ] State estimation interface (IMU, kinematics, contact)
- [ ] Warm start and solver initialization
- [ ] Latency compensation
- [ ] Output reference commands (velocity, footholds, contact schedule)

### Stage 4: Low-Level Control Integration
- [ ] Mapping MPC outputs to joint-level commands
- [ ] Interface with 1 kHz PD / FOC controllers
- [ ] Fail-safe design and fallback strategies
- [ ] Simulation validation (IsaacGym / IsaacLab)
- [ ] High-frequency deployable MPC module

---

## Repository Structure (Planned)

.
├─ docs/ # Blog-style documentation and derivations
├─ src/ # MPC, dynamics, estimation, interfaces
├─ configs/ # Controller and experiment configurations
├─ scripts/ # Demo and visualization scripts
├─ experiments/ # Logs and results
└─ tests/ # Unit tests


---

## Status

🚧 This repository is under active development.  
Each completed module will be accompanied by:
- runnable code
- quantitative evaluation
- visual results (plots / videos)

---

## License
MIT License

