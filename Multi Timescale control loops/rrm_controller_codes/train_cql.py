import json
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

# --- IMPORT YOUR CONTROLLER CORE ---
# Adjust the module name if your file is called differently
from rrm_controller_pretrain import (
    GNNQNetwork,
    NUM_ACTIONS,
    build_networkx_interference_graph,
    build_pyg_graph,
    NO_OP,
)

# Limit how many times we complain about node count mismatches
NODE_MISMATCH_WARNINGS = 0
NODE_MISMATCH_WARNINGS_MAX = 10


# -------------------------
# Hyperparams
# -------------------------
GAMMA = 0.98          # slightly less “long-term”
CQL_ALPHA = 0.3       # was 1.0, make it less conservative
LR = 5e-4             # was 1e-3
NUM_EPOCHS = 100   
PRINT_EVERY = 1   # print every epoch


# We’ll treat each slow-loop step as one “graph sample”
# and each AP node inside as part of the batch.


class GraphTransition:
    """
    One offline transition:

      state_graph:  Data for prev_snapshot
      next_graph:   Data for curr_snapshot
      actions:      torch.LongTensor [num_nodes] (per-AP action index)
      reward:       float (global)
    """

    def __init__(self, state_graph, next_graph, actions, reward: float):
        self.state_graph = state_graph
        self.next_graph = next_graph
        self.actions = actions
        self.reward = float(reward)


def load_dataset_from_jsonl(path: str,
                            skip_guardrail_violations: bool = False) -> List[GraphTransition]:
    """
    Reads rrm_experience_log.jsonl and converts to graph transitions.

    Each log entry:
      {
        "step": ...,
        "prev_snapshot": {...},
        "curr_snapshot": {...},
        "actions": {"ap1": 1, "ap2": 0, ...},
        "reward": float,
        "guardrail_violated": bool,
        ...
      }
    """
    transitions: List[GraphTransition] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            if skip_guardrail_violations and rec.get("guardrail_violated", False):
                continue

            prev_snapshot = rec.get("prev_snapshot")
            curr_snapshot = rec.get("curr_snapshot")
            actions_dict = rec.get("actions") or {}
            reward = float(rec.get("reward", 0.0))

            if not prev_snapshot or not curr_snapshot:
                continue

            # Build graphs
            G_prev = build_networkx_interference_graph(prev_snapshot)
            data_prev, ap_index_prev = build_pyg_graph(prev_snapshot, G_prev)

            G_curr = build_networkx_interference_graph(curr_snapshot)
            data_curr, ap_index_curr = build_pyg_graph(curr_snapshot, G_curr)

            num_nodes = data_prev.num_nodes
            if num_nodes == 0:
                continue

            # Build per-node action vector (default NO_OP)
            actions_vec = torch.full((num_nodes,), NO_OP, dtype=torch.long)

            # For each AP with a logged action, map into index
            for ap_id, act in actions_dict.items():
                if ap_id in ap_index_prev:
                    idx = ap_index_prev[ap_id]
                    try:
                        actions_vec[idx] = int(act)
                    except Exception:
                        pass

            transitions.append(GraphTransition(
                state_graph=data_prev,
                next_graph=data_curr,
                actions=actions_vec,
                reward=reward,
            ))

    print(f"[DATA] Loaded {len(transitions)} transitions from {path}")
    return transitions


def cql_loss(q_values: torch.Tensor,
             actions: torch.Tensor,
             alpha: float) -> torch.Tensor:
    """
    Conservative Q-Learning regularizer for discrete actions.

    q_values: [N, A]
    actions:  [N] (behavior actions)
    alpha:    scalar weight

    loss = alpha * ( E[logsumexp(Q(s,:))] - E[Q(s,a_beta)] )
    """
    # log-sum-exp over actions
    logsumexp_all = torch.logsumexp(q_values, dim=1)  # [N]
    # Q(s,a_beta)
    q_behavior = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [N]

    cql_term = (logsumexp_all - q_behavior).mean()
    return alpha * cql_term

def split_train_val(transitions, val_frac: float = 0.2, seed: int = 42):
    """
    Randomly split transitions into train/val sets.
    """
    n = len(transitions)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(n * (1.0 - val_frac))
    train_idx = idx[:split]
    val_idx = idx[split:]
    train_tr = [transitions[i] for i in train_idx]
    val_tr   = [transitions[i] for i in val_idx]
    return train_tr, val_tr


def evaluate_model_on_val(
    q_net: nn.Module,
    val_transitions,
    gamma: float,
    cql_alpha: float,
    device: str = "cpu",
) -> float:
    """
    Compute average (Bellman + CQL) loss on the validation transitions.
    This is the objective we minimize in grid search.
    """
    q_net.eval()
    losses = []

    with torch.no_grad():
        for tr in val_transitions:
            data_s  = tr.state_graph.to(device)
            data_sp = tr.next_graph.to(device)
            actions = tr.actions.to(device)
            reward  = tr.reward

            # Q(s, ·)
            q_values, _ = q_net(data_s)        # [N_prev, A]
            # Q(s', ·) using SAME net for evaluation
            q_next, _   = q_net(data_sp)       # [N_next, A]
            max_next, _ = q_next.max(dim=1)    # [N_next]

            N = q_values.size(0)
            if max_next.size(0) >= N:
                max_next_aligned = max_next[:N]
            else:
                pad = torch.zeros(N, device=max_next.device, dtype=max_next.dtype)
                pad[:max_next.size(0)] = max_next
                max_next_aligned = pad

            target = float(reward) + gamma * max_next_aligned   # [N]
            q_sa   = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [N]

            bellman_loss = F.mse_loss(q_sa, target)
            cql_reg      = cql_loss(q_values, actions, cql_alpha)
            total_loss   = bellman_loss + cql_reg

            losses.append(total_loss.item())

    return float(np.mean(losses)) if losses else 0.0

def train_one_run(
    transitions_train,
    transitions_val,
    hidden_dim: int,
    gamma: float,
    alpha: float,
    lr: float,
    device: str = "cpu",
    max_epochs: int = 50,
    print_every: int = 10,
    run_label: str = "",
):
    """
    Train CQL once on a train subset, validate on val subset.
    Returns (best_train_loss, final_val_loss).
    """

    global NODE_MISMATCH_WARNINGS, NODE_MISMATCH_WARNINGS_MAX

    # ----- build nets -----
    in_dim = transitions_train[0].state_graph.x.size(1)
    q_net = GNNQNetwork(in_dim=in_dim, hidden_dim=hidden_dim, num_actions=NUM_ACTIONS).to(device)
    target_net = GNNQNetwork(in_dim=in_dim, hidden_dim=hidden_dim, num_actions=NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = Adam(q_net.parameters(), lr=lr)
    TARGET_UPDATE_INTERVAL = 10

    best_train_loss = float("inf")
    patience = 20
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        random.shuffle(transitions_train)
        epoch_losses = []

        for tr in transitions_train:
            data_s = tr.state_graph.to(device)
            data_sp = tr.next_graph.to(device)
            actions = tr.actions.to(device)
            reward = tr.reward

            # ----- forward current state -----
            q_values, _ = q_net(data_s)           # [N_prev, A]

            # ----- target for next state -----
            with torch.no_grad():
                q_next, _ = target_net(data_sp)   # [N_next, A]
                max_next, _ = q_next.max(dim=1)   # [N_next]

                N_prev = q_values.size(0)
                N_next = max_next.size(0)

                if N_prev != N_next and NODE_MISMATCH_WARNINGS < NODE_MISMATCH_WARNINGS_MAX:
                    print(
                        f"[WARN] train_one_run: node mismatch prev={N_prev}, next={N_next} "
                        f"(cropping/padding next-state values)"
                    )
                    NODE_MISMATCH_WARNINGS += 1

                if N_next >= N_prev:
                    max_next_aligned = max_next[:N_prev]
                else:
                    pad = torch.zeros(N_prev, device=max_next.device, dtype=max_next.dtype)
                    pad[:N_next] = max_next
                    max_next_aligned = pad

                target = float(reward) + gamma * max_next_aligned  # [N_prev]

            # ----- Bellman + CQL -----
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [N_prev]

            bellman_loss = F.mse_loss(q_sa, target)
            cql_reg = cql_loss(q_values, actions, alpha)
            loss = bellman_loss + cql_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # Target network update
        if epoch % TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())

        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # ----- epoch printout for this config -----
        if epoch % print_every == 0 or epoch == max_epochs:
            print(
                f"[GRID][{run_label}] Epoch {epoch}/{max_epochs}, "
                f"mean train loss={mean_train_loss:.4f}"
            )

        # track best train loss for this config
        if mean_train_loss < best_train_loss - 1e-3:
            best_train_loss = mean_train_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                # early stop
                print(
                    f"[GRID][{run_label}] Early stopping at epoch {epoch}, "
                    f"best_train_loss={best_train_loss:.4f}"
                )
                break

    # ----- validation loss with final weights -----
    q_net.eval()
    val_losses = []
    with torch.no_grad():
        for tr in transitions_val:
            data_s = tr.state_graph.to(device)
            data_sp = tr.next_graph.to(device)
            actions = tr.actions.to(device)
            reward = tr.reward

            q_values, _ = q_net(data_s)
            q_next, _ = target_net(data_sp)
            max_next, _ = q_next.max(dim=1)

            N_prev = q_values.size(0)
            N_next = max_next.size(0)

            if N_next >= N_prev:
                max_next_aligned = max_next[:N_prev]
            else:
                pad = torch.zeros(N_prev, device=max_next.device, dtype=max_next.dtype)
                pad[:N_next] = max_next
                max_next_aligned = pad

            target = float(reward) + gamma * max_next_aligned
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            bellman_loss = F.mse_loss(q_sa, target)
            cql_reg = cql_loss(q_values, actions, alpha)
            val_losses.append((bellman_loss + cql_reg).item())

    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    return best_train_loss, val_loss



def train_cql(
    log_path: str,
    out_path: str,
    hidden_dim: int = 32,
    device: str = "cpu",
    skip_guardrail_violations: bool = False,
):
    # -------------------------
    # Load dataset
    # -------------------------
    transitions = load_dataset_from_jsonl(
        log_path,
        skip_guardrail_violations=skip_guardrail_violations,
    )
    if not transitions:
        print("[TRAIN] No transitions found, aborting.")
        return

    # Infer input dim from first sample
    in_dim = transitions[0].state_graph.x.size(1)

    # -------------------------
    # Build networks
    # -------------------------
    q_net = GNNQNetwork(in_dim=in_dim, hidden_dim=hidden_dim, num_actions=NUM_ACTIONS).to(device)
    target_net = GNNQNetwork(in_dim=in_dim, hidden_dim=hidden_dim, num_actions=NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = Adam(q_net.parameters(), lr=LR)

    # Simple target update schedule
    TARGET_UPDATE_INTERVAL = 10  # epochs

    # -------------------------
    # Training loop
    # -------------------------
    best_loss = float("inf")
    patience = 10      # stop if no improvement for 10 epochs
    bad_epochs = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        random.shuffle(transitions)
        epoch_losses = []

        for tr in transitions:
            # Move graphs to device
            data_s = tr.state_graph.to(device)
            data_sp = tr.next_graph.to(device)
            actions = tr.actions.to(device)
            reward = tr.reward

            # q(s,·)
            q_values, _ = q_net(data_s)           # [N_prev, A]
            # q(s',·) for target
            with torch.no_grad():
                q_next, _ = target_net(data_sp)   # [N_next, A]
                max_next, _ = q_next.max(dim=1)   # [N_next]

                # --- Align next-state values to the state graph size ---
                N = q_values.size(0)              # N_prev
                
                if max_next.size(0) != N:
                    print(f"[WARN] node mismatch in transition: prev={N}, next={max_next.size(0)}")

                if max_next.size(0) >= N:
                    max_next_aligned = max_next[:N]
                else:
                    # pad with zeros if next-state has fewer nodes
                    pad = torch.zeros(N, device=max_next.device, dtype=max_next.dtype)
                    pad[:max_next.size(0)] = max_next
                    max_next_aligned = pad

                # global reward + discounted next-value, same length as q_sa
                target = float(reward) + GAMMA * max_next_aligned   # [N]

            # Q(s,a_beta)
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [N]

            bellman_loss = F.mse_loss(q_sa, target)


            # CQL conservative loss
            cql_reg = cql_loss(q_values, actions, CQL_ALPHA)

            loss = bellman_loss + cql_reg

            optimizer.zero_grad()
            loss.backward()
            
            # --- gradient clipping (PREVENTS HUGE GRADIENTS) ---
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # Periodic target network update
        if epoch % TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())

        if epoch % PRINT_EVERY == 0:
                mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                print(f"[TRAIN] Epoch {epoch}/{NUM_EPOCHS}, mean loss={mean_loss:.4f}")

                # --- EARLY STOPPING + BEST CHECKPOINT SAVE ---
                if mean_loss < best_loss - 1e-3:
                    best_loss = mean_loss
                    bad_epochs = 0
                    # save best weights each time we improve
                    torch.save(q_net.state_dict(), out_path)
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"[TRAIN] Early stopping at epoch {epoch}, best_loss={best_loss:.4f}")
                        break


    # -------------------------
    # Save final policy
    # -------------------------
    torch.save(q_net.state_dict(), out_path)
    print(f"[TRAIN] Saved trained CQL model to {out_path}")

def grid_search_cql(
    log_path: str,
    out_path: str,
    device: str = "cpu",
    skip_guardrail_violations: bool = False,
):
    transitions = load_dataset_from_jsonl(
        log_path,
        skip_guardrail_violations=skip_guardrail_violations,
    )
    if not transitions:
        print("[GRID] No transitions found, aborting.")
        return

    # train/val split
    random.shuffle(transitions)
    split = int(0.8 * len(transitions))
    train_set = transitions[:split]
    val_set = transitions[split:]
    print(f"[GRID] Train size={len(train_set)}, Val size={len(val_set)}")

    hidden_dims = [16, 32]
    gammas = [0.95, 0.98]
    alphas = [0.1, 0.3, 0.5]
    lrs = [1e-3, 5e-4]

    best_val_loss = float("inf")
    best_cfg = None

    for h in hidden_dims:
        for g in gammas:
            for a in alphas:
                for lr in lrs:
                    label = f"h{h}_g{g}_a{a}_lr{lr}"
                    print(f"[GRID] Trying hidden={h}, gamma={g}, alpha={a}, lr={lr}")

                    train_best, val_loss = train_one_run(
                        train_set,
                        val_set,
                        hidden_dim=h,
                        gamma=g,
                        alpha=a,
                        lr=lr,
                        device=device,
                        max_epochs=50,      # reduce to 30–40 if it's too slow
                        print_every=10,
                        run_label=label,
                    )

                    print(
                        f"[GRID] Result {label} -> train_best={train_best:.4f}, "
                        f"val={val_loss:.4f}"
                    )

                    if val_loss < best_val_loss - 1e-3:
                        best_val_loss = val_loss
                        best_cfg = (h, g, a, lr)
                        print(
                            f"[GRID] New best config {best_cfg} with val={best_val_loss:.4f}"
                        )
                        # save model weights for this best run
                        # NOTE: if you want the actual weights, you can either
                        #   - re-train once with best config (using train_cql),
                        #   - or modify train_one_run to return the model.
                        # Here we just mark the config; final training will use train_cql.
    
    print("[GRID] =========================")
    if best_cfg is None:
        print("[GRID] No valid config found")
        return

    h, g, a, lr = best_cfg
    print(
        f"[GRID] Best config: hidden={h}, gamma={g}, alpha={a}, lr={lr}, "
        f"best_val_loss={best_val_loss:.4f}"
    )

    # Re-train once on the full dataset with best hyperparams and save
    print("[GRID] Re-training on all data with best config and saving model...")
    train_cql(
        log_path=log_path,
        out_path=out_path,
        hidden_dim=h,
        device=device,
        skip_guardrail_violations=skip_guardrail_violations,
        # if you want, you can pass g, a, lr via globals or arguments
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="rrm_experience_log.jsonl",
                        help="Path to JSONL experience log from controller")
    parser.add_argument("--out_path", type=str, default="cql_rrm_gnn_current.pth",
                        help="Where to save the trained model")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_guardrail_violations", action="store_true",
                        help="If set, skip transitions where guardrails were violated")
    parser.add_argument("--mode", choices=["train", "grid"], default="train",
                        help="'train' = single run (old behaviour), 'grid' = grid search CV")
    args = parser.parse_args()

    if args.mode == "train":
        # old behaviour: single run using global GAMMA, CQL_ALPHA, LR, NUM_EPOCHS
        train_cql(
            log_path=args.log_path,
            out_path=args.out_path,
            hidden_dim=args.hidden_dim,
            device=args.device,
            skip_guardrail_violations=args.skip_guardrail_violations,
        )
    else:
        # new behaviour: grid search + CV
        grid_search_cql(
            log_path=args.log_path,
            out_path=args.out_path,
            device=args.device,
            skip_guardrail_violations=args.skip_guardrail_violations,
        )
if __name__=="__main__":
    main()