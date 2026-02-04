import torch


def set_bg_flow(bgFlow, speed):
    def get_flow(X):
        N = X.shape[0] // 2  # Assuming the input X is split into two halves
        if bgFlow == "relax":
            return torch.zeros_like(X)  # Relaxation
        elif bgFlow == "shear":
            return speed * torch.vstack((X[N:], torch.zeros_like(X[:N])))  # Shear
        elif bgFlow == "tayGreen":
            return speed * torch.vstack(
                (
                    torch.sin(X[:N]) * torch.cos(X[N:]),
                    -torch.cos(X[:N]) * torch.sin(X[N:]),
                )
            )  # Taylor-Green
        elif bgFlow == "parabolic":
            return torch.vstack(
                (speed * (1 - (X[N:] / 0.375) ** 2), torch.zeros_like(X[:N]))
            )  # Parabolic
        elif bgFlow == "rotation":
            r = torch.sqrt(X[:N] ** 2 + X[N:] ** 2)
            theta = torch.atan2(X[N:], X[:N])
            return speed * torch.vstack(
                (-torch.sin(theta) / r, torch.cos(theta) / r)
            )  # Rotation
        elif bgFlow == "vortex":
            chanWidth = 2.5 * 2
            return speed * torch.cat(
                [
                    torch.sin(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.cos(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                    -torch.cos(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.sin(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                ],
                dim=0,
            )

        else:
            return torch.zeros_like(X)

    return get_flow
