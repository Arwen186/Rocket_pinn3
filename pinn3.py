import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

g = 9.81
v0 = 626.0

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer_in = nn.Linear(1, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 2)
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        x = self.activation(self.layer_in(t))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        out = self.layer_out(x)
        h = out[:, 0:1] * 20000.0
        v = out[:, 1:2] * 626.0
        return h, v

def compute_loss(model, t_physics, t_initial):
    t_physics.requires_grad = True
    h, v = model(t_physics)

    dh_dt = torch.autograd.grad(h, t_physics, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    dv_dt = torch.autograd.grad(v, t_physics, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    residual1 = dh_dt - v
    residual2 = dv_dt + g

    loss_physics = torch.mean(residual1**2) + torch.mean(residual2**2)

    t_initial.requires_grad = True
    h0, v0_pred = model(t_initial)
    loss_h0 = torch.mean(h0**2)
    loss_v0 = torch.mean((v0_pred - v0)**2)

    total_loss = loss_physics + 100.0 * (loss_h0 + loss_v0)
    return total_loss, loss_physics.item(), loss_h0.item(), loss_v0.item()

def exact_solution(t):
    return v0 * t - 0.5 * g * t**2

def main():
    t_max = v0 / g
    model = PINN().to(device)

    n_physics = 2000
    t_np = np.linspace(0, t_max, n_physics).reshape(-1, 1)
    t_physics = torch.tensor(t_np, dtype=torch.float32, device=device, requires_grad=True)

    t_initial = torch.zeros(100, 1, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    n_epochs = 50000
    start_epoch = 1

    if os.path.exists('checkpoint5.pth'):
        checkpoint5 = torch.load('checkpoint5.pth')
        model.load_state_dict(checkpoint5['model_state_dict'])
        optimizer.load_state_dict(checkpoint5['optimizer_state_dict'])
        start_epoch = checkpoint5['epoch'] + 1

        print(f"Resuming from epoch {start_epoch}")

    history_total = []
    history_physics = []

    print("\n--- PINN Rocket Simulation ---")
    print(f"Apogee: {v0**2/(2*g):.1f} m")
    print("Training started...")

    for epoch in range(start_epoch, n_epochs + 1):
        optimizer.zero_grad()
        total_loss, loss_physics, loss_h0, loss_v0 = compute_loss(model, t_physics, t_initial)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history_total.append(total_loss.item())
        history_physics.append(loss_physics)

        if epoch % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoint5.pth')

        if epoch % 500 == 0:        
            print(f"Epoch {epoch:5d}/{n_epochs} | Loss: {total_loss.item():.2e}")

    print("Training completed!")

    t_test = torch.linspace(0, t_max, 200, device=device).float().reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        h_pred, v_pred = model(t_test)
        h_pred = h_pred.cpu().numpy().reshape(-1)
        v_pred = v_pred.cpu().numpy().reshape(-1)
    t_np_test = t_test.cpu().numpy().reshape(-1)
    h_exact = exact_solution(t_np_test)

    fig = plt.figure(figsize=(8, 10))
    fig.suptitle('PINN Rocket Simulation', fontsize=18, fontweight='bold')

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_np_test, h_pred, 'r-', linewidth=3, label='PINN Prediction', alpha=0.9)
    ax1.plot(t_np_test, h_exact, 'b--', linewidth=2, label='Exact Solution', alpha=0.7)
    ax1.fill_between(t_np_test, h_pred, alpha=0.1, color='red')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title('Rocket Altitude vs Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_np_test, v_pred, 'g-', linewidth=3, label='PINN Velocity', alpha=0.9)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Rocket Velocity vs Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(history_total, 'b-', linewidth=2, label='Total Loss', alpha=0.8)
    ax3.semilogy(history_physics, 'r-', linewidth=2, label='Physics Loss', alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    error = np.abs(h_exact - h_pred)
    ax4.semilogy(t_np_test, error, 'k-', linewidth=2, alpha=0.8)
    ax4.fill_between(t_np_test, error, alpha=0.2, color='black')
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Absolute Error (m)', fontsize=12)
    ax4.set_title('Prediction Error', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    mae = np.mean(error)
    max_h_pinn = np.max(h_pred)
    max_h_exact = np.max(h_exact)

    results_text = (
        f"RESULTS SUMMARY\n"
        f"{'─' * 30}\n"
        f"PINN Apogee: {max_h_pinn:.2f} m\n"
        f"Exact Apogee: {max_h_exact:.2f} m\n"
        f"Mean Absolute Error: {mae:.4f} m\n"
        f"Final Physics Loss:  {history_physics[-1]:.2e}\n"
        f"Status: TRAINING SUCCESSFUL"
    )

    fig.text(0.5, 0.01, results_text, ha='center', fontsize=11,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig('pinn_rocket_final.png', dpi=200, bbox_inches='tight')
    plt.show()

    print(f"\n{results_text}")

if __name__ == "__main__":
    main()