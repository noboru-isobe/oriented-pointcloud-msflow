"""Linear algebra utilities."""

import logging
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TruncatedSVD:
    """Pre-computed truncated SVD for efficient pseudo-inverse application.

    Stores U, S_inv, Vh from SVD of A, with small singular values truncated.
    Provides apply() method to compute A^† @ b efficiently.

    Usage:
        svd = TruncatedSVD.from_matrix(A, rcond=1e-6)
        x = svd.apply(b)  # x = A^† @ b
    """

    U: torch.Tensor       # (M, M)
    S_inv: torch.Tensor   # (min(M,N),)
    Vh: torch.Tensor      # (N, N)
    n_truncated: int      # number of truncated singular values

    @classmethod
    def from_matrix(
        cls,
        A: torch.Tensor,
        rcond: float | None = None,
        log_level: int = logging.DEBUG,
    ) -> "TruncatedSVD":
        """Compute truncated SVD of matrix A.

        Args:
            A: (M, N) matrix
            rcond: Relative cutoff for small singular values.
                   Singular values <= rcond * S.max() are treated as zero.
                   If None, uses eps * max(M, N) (same as torch.linalg.lstsq).
            log_level: Logging level for SVD details (DEBUG, INFO, etc.)

        Returns:
            TruncatedSVD instance with pre-computed decomposition.
        """
        try:
            U, S, Vh = torch.linalg.svd(A.detach())
        except torch._C._LinAlgError as e:
            if A.device.type == "cuda":
                logger.warning(
                    f"CUDA SVD failed, falling back to CPU: {e}"
                )
                A_cpu = A.detach().cpu()
                U, S, Vh = torch.linalg.svd(A_cpu)
                U = U.to(A.device)
                S = S.to(A.device)
                Vh = Vh.to(A.device)
            else:
                raise

        # Default rcond: torch.linalg.lstsq style
        if rcond is None:
            eps = torch.finfo(A.dtype).eps
            rcond = eps * max(A.shape)

        tau = rcond * S.max()
        S_inv = torch.where(S > tau, 1.0 / S, torch.zeros_like(S))
        n_truncated = (S <= tau).sum().item()

        # Log SVD details
        logger.log(
            log_level,
            f"TruncatedSVD: shape={tuple(A.shape)}, S_min={S.min():.2e}, "
            f"S_max={S.max():.2e}, tau={tau:.2e}, truncated={n_truncated}/{len(S)}"
        )

        return cls(U=U, S_inv=S_inv, Vh=Vh, n_truncated=n_truncated)

    def apply(self, b: torch.Tensor) -> torch.Tensor:
        """Apply pseudo-inverse: x = A^† @ b.

        Args:
            b: (M,) or (M, K) right-hand side vector/matrix

        Returns:
            x: (N,) or (N, K) minimum-norm solution
        """
        # A^† = Vh.T @ diag(S_inv) @ U.T
        return self.Vh.T @ (self.S_inv.unsqueeze(-1) * (self.U.T @ b.unsqueeze(-1))).squeeze(-1)
