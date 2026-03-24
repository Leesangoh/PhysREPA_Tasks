"""Simple wrapper to load and run RL-Games LSTM policy from checkpoint."""

import torch
import torch.nn as nn


class RlGamesPolicy:
    """Load rl_games A2C+LSTM checkpoint and run inference.

    Architecture (from Factory): obs(19) → LSTM(2-layer, h=1024) → MLP(512→128→64) → mu(6)
    """

    def __init__(self, checkpoint_path: str, device: str | torch.device = "cuda:0"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = ckpt["model"]

        # Extract running mean/std for observation normalization
        self.obs_mean = model["running_mean_std.running_mean"].float().to(self.device)
        self.obs_var = model["running_mean_std.running_var"].float().to(self.device)
        self.obs_count = model["running_mean_std.count"].to(self.device)

        # Build LSTM
        self.rnn = nn.LSTM(input_size=19, hidden_size=1024, num_layers=2, batch_first=True).to(self.device)
        # Load LSTM weights
        rnn_state = {}
        for k, v in model.items():
            if "rnn.rnn." in k:
                rnn_key = k.replace("a2c_network.rnn.rnn.", "")
                rnn_state[rnn_key] = v
        self.rnn.load_state_dict(rnn_state)

        # Build layer norm
        self.layer_norm = nn.LayerNorm(1024).to(self.device)
        self.layer_norm.weight.data = model["a2c_network.layer_norm.weight"]
        self.layer_norm.bias.data = model["a2c_network.layer_norm.bias"]

        # Build actor MLP
        self.actor_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        ).to(self.device)
        self.actor_mlp[0].weight.data = model["a2c_network.actor_mlp.0.weight"]
        self.actor_mlp[0].bias.data = model["a2c_network.actor_mlp.0.bias"]
        self.actor_mlp[2].weight.data = model["a2c_network.actor_mlp.2.weight"]
        self.actor_mlp[2].bias.data = model["a2c_network.actor_mlp.2.bias"]
        self.actor_mlp[4].weight.data = model["a2c_network.actor_mlp.4.weight"]
        self.actor_mlp[4].bias.data = model["a2c_network.actor_mlp.4.bias"]

        # Mu head
        self.mu = nn.Linear(64, 6).to(self.device)
        self.mu.weight.data = model["a2c_network.mu.weight"]
        self.mu.bias.data = model["a2c_network.mu.bias"]

        # LSTM hidden state
        self.hidden = None
        self.eval()

    def eval(self):
        self.rnn.eval()
        self.actor_mlp.eval()
        self.mu.eval()
        self.layer_norm.eval()

    def reset(self, env_ids=None):
        """Reset LSTM hidden state for specific envs, or all if env_ids=None."""
        if env_ids is None or self.hidden is None:
            self.hidden = None
        else:
            # Zero out hidden state for specific envs
            for idx in env_ids:
                self.hidden[0][:, idx, :] = 0.0
                self.hidden[1][:, idx, :] = 0.0

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Run inference. obs: (1, obs_dim) → action: (1, 6)"""
        with torch.no_grad():
            obs = obs.float()  # ensure float32
            # Normalize obs
            obs_norm = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)

            # LSTM expects (batch, seq_len, input_size)
            if obs_norm.dim() == 2:
                obs_norm = obs_norm.unsqueeze(1)  # (1, 1, 19)

            if self.hidden is None:
                h0 = torch.zeros(2, obs_norm.shape[0], 1024, device=self.device)
                c0 = torch.zeros(2, obs_norm.shape[0], 1024, device=self.device)
                self.hidden = (h0, c0)

            rnn_out, self.hidden = self.rnn(obs_norm, self.hidden)
            rnn_out = rnn_out[:, -1, :]  # (1, 1024)

            # Layer norm
            rnn_out = self.layer_norm(rnn_out)

            # Actor MLP
            x = self.actor_mlp(rnn_out)
            action = self.mu(x)  # (1, 6)
            return action.clamp(-1.0, 1.0)
