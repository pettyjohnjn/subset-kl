# tests/test_subset_kl.py
"""Tests for subset-kl package."""

import pytest
import torch
import torch.nn.functional as F

from subset_kl import (
    # Core functional
    select_topk_indices,
    select_indices_with_sampling,
    select_indices_with_importance_sampling,
    subset_kl_from_gathered,
    compute_subset_kl,
    full_kl,
    # Classes
    SubsetKLLoss,
    KLDivergenceLoss,
    HajekKLLoss,
    ImportanceKLLoss,
    # Sampling
    pps_sample_indices_batched,
    hajek_kl_estimate,
    # Base
    apply_reduction,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def logits():
    """Standard test logits."""
    torch.manual_seed(42)
    B, T, V = 2, 8, 1000
    teacher = torch.randn(B, T, V) * 2  # Peaked distribution
    student = torch.randn(B, T, V, requires_grad=True)
    return teacher, student


@pytest.fixture
def small_logits():
    """Small logits for quick tests."""
    torch.manual_seed(42)
    B, T, V = 1, 4, 100
    teacher = torch.randn(B, T, V)
    student = torch.randn(B, T, V, requires_grad=True)
    return teacher, student


@pytest.fixture
def attention_mask():
    """Standard attention mask."""
    B, T = 2, 8
    mask = torch.ones(B, T)
    mask[0, -2:] = 0
    return mask


# =============================================================================
# Core Functional Tests
# =============================================================================

class TestSelectTopkIndices:
    """Tests for select_topk_indices."""
    
    def test_basic_shapes(self, small_logits):
        """Test output shapes."""
        teacher, _ = small_logits
        B, T, V = teacher.shape
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        
        assert indices.shape == (B, T, k)
        assert teacher_k.shape == (B, T, k)
    
    def test_indices_are_valid(self, small_logits):
        """Test indices are in valid range."""
        teacher, _ = small_logits
        V = teacher.shape[-1]
        
        indices, _ = select_topk_indices(teacher, k=32)
        
        assert (indices >= 0).all()
        assert (indices < V).all()
    
    def test_indices_are_top_k(self, small_logits):
        """Test that indices correspond to highest logits."""
        teacher, _ = small_logits
        k = 10
        
        indices, teacher_k = select_topk_indices(teacher, k)
        
        # Gather should give same values
        gathered = torch.gather(teacher, -1, indices)
        assert torch.allclose(gathered, teacher_k)


class TestSelectIndicesWithImportanceSampling:
    """Tests for select_indices_with_importance_sampling."""
    
    def test_basic_shapes(self, small_logits):
        """Test output shapes."""
        teacher, _ = small_logits
        B, T, V = teacher.shape
        k = 16
        
        indices, teacher_k, inc_probs, mask = select_indices_with_importance_sampling(
            teacher, k=k, oversample=10
        )
        S = indices.shape[-1]
        
        assert S <= k
        assert indices.shape == (B, T, S)
        assert teacher_k.shape == (B, T, S)
        assert inc_probs.shape == (B, T, S)
        assert mask.shape == (B, T, S)
        assert mask.any()


class TestSubsetKLFromGathered:
    """Tests for subset_kl_from_gathered."""
    
    def test_basic(self, small_logits):
        """Test basic computation."""
        teacher, student = small_logits
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss = subset_kl_from_gathered(student_k, teacher_k)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # KL is non-negative
        assert torch.isfinite(loss)
    
    def test_identical_distributions(self):
        """Test KL=0 for identical distributions."""
        B, T, k = 1, 4, 32
        logits_k = torch.randn(B, T, k)
        
        loss = subset_kl_from_gathered(logits_k, logits_k)
        
        assert loss.item() < 1e-5
    
    def test_gradient_flow(self, small_logits):
        """Test gradients flow correctly."""
        teacher, student = small_logits
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss = subset_kl_from_gathered(student_k, teacher_k)
        loss.backward()
        
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()
    
    def test_reduction_none(self, small_logits):
        """Test reduction='none' returns per-token."""
        teacher, student = small_logits
        B, T, V = teacher.shape
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss = subset_kl_from_gathered(student_k, teacher_k, reduction="none")
        
        assert loss.shape == (B, T)
    
    def test_with_attention_mask(self, small_logits):
        """Test attention mask is applied."""
        teacher, student = small_logits
        B, T, V = teacher.shape
        k = 32
        
        mask = torch.ones(B, T)
        mask[0, -1] = 0
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss_masked = subset_kl_from_gathered(student_k, teacher_k, mask)
        loss_unmasked = subset_kl_from_gathered(student_k, teacher_k)
        
        # Should be different (masked excludes some positions)
        assert not torch.allclose(loss_masked, loss_unmasked)


class TestComputeSubsetKL:
    """Tests for compute_subset_kl convenience function."""
    
    def test_matches_manual(self, small_logits):
        """Test it matches manual computation."""
        teacher, student = small_logits
        k = 32
        
        # Manual
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student.detach(), -1, indices)
        expected = subset_kl_from_gathered(student_k, teacher_k)
        
        # Convenience function
        actual = compute_subset_kl(student.detach(), teacher, k=k)
        
        assert torch.allclose(expected, actual)


class TestFullKL:
    """Tests for full_kl baseline."""
    
    def test_basic(self, small_logits):
        """Test basic computation."""
        teacher, student = small_logits
        
        loss = full_kl(student, teacher)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)
    
    def test_subset_approaches_full(self, logits):
        """Test subset KL approaches full as k increases."""
        teacher, student = logits
        V = teacher.shape[-1]
        
        full_loss = full_kl(student.detach(), teacher).item()
        
        # Higher k should be closer to full
        loss_k64 = compute_subset_kl(student.detach(), teacher, k=64).item()
        loss_k256 = compute_subset_kl(student.detach(), teacher, k=256).item()
        loss_k512 = compute_subset_kl(student.detach(), teacher, k=512).item()
        
        # k=512 should be closer to full than k=64
        assert abs(loss_k512 - full_loss) <= abs(loss_k64 - full_loss)


# =============================================================================
# Class Interface Tests
# =============================================================================

class TestSubsetKLLoss:
    """Tests for SubsetKLLoss class."""
    
    def test_select_indices(self, small_logits):
        """Test select_indices method."""
        teacher, _ = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        indices, teacher_k = loss_fn.select_indices(teacher)
        
        assert indices.shape[-1] == 32
        assert loss_fn.last_indices is not None
    
    def test_forward_gathered(self, small_logits):
        """Test forward_gathered method."""
        teacher, student = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        indices, teacher_k = loss_fn.select_indices(teacher)
        student_k = torch.gather(student, -1, indices)
        
        loss = loss_fn.forward_gathered(student_k, teacher_k)
        
        assert torch.isfinite(loss)
    
    def test_forward_convenience(self, small_logits):
        """Test forward() convenience method."""
        teacher, student = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        loss = loss_fn(student, teacher)
        
        assert torch.isfinite(loss)
    
    def test_two_paths_match(self, small_logits):
        """Test that gathered path matches convenience path."""
        teacher, student = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        
        # Convenience path
        loss1 = loss_fn(student.detach(), teacher)
        
        # Gathered path
        indices, teacher_k = loss_fn.select_indices(teacher)
        student_k = torch.gather(student.detach(), -1, indices)
        loss2 = loss_fn.forward_gathered(student_k, teacher_k)
        
        assert torch.allclose(loss1, loss2)


class TestKLDivergenceLoss:
    """Tests for KLDivergenceLoss class."""
    
    def test_basic(self, small_logits):
        """Test basic usage."""
        teacher, student = small_logits
        
        loss_fn = KLDivergenceLoss()
        loss = loss_fn(student, teacher)
        
        assert torch.isfinite(loss)
    
    def test_chunked_matches_full(self, small_logits):
        """Test chunked computation matches full."""
        teacher, student = small_logits
        
        loss_full = KLDivergenceLoss()(student.detach(), teacher)
        loss_chunked = KLDivergenceLoss(chunk_size=2)(student.detach(), teacher)
        
        assert torch.allclose(loss_full, loss_chunked, rtol=1e-4)


class TestHajekKLLoss:
    """Tests for HajekKLLoss class."""
    
    def test_basic(self, small_logits):
        """Test basic usage."""
        teacher, student = small_logits
        
        loss_fn = HajekKLLoss(k_head=16, k_tail=8)
        loss = loss_fn(student, teacher)
        
        assert torch.isfinite(loss)
        assert loss_fn.variance_proxy is not None
        assert loss_fn.diagnostics is not None


class TestImportanceKLLoss:
    """Tests for ImportanceKLLoss class."""
    
    def test_basic(self, small_logits):
        """Test basic usage."""
        teacher, student = small_logits
        
        loss_fn = ImportanceKLLoss(k=16)
        loss = loss_fn(student, teacher)
        
        assert torch.isfinite(loss)


# =============================================================================
# Sampling Tests
# =============================================================================

class TestPPSSampling:
    """Tests for PPS sampling."""
    
    def test_basic(self):
        """Test basic sampling."""
        torch.manual_seed(42)
        N, V = 10, 100
        log_probs = F.log_softmax(torch.randn(N, V), dim=-1)
        
        indices, inc_probs, mask, diag = pps_sample_indices_batched(
            log_probs, k_head=16, k_tail=8
        )
        
        assert indices.shape[0] == N
        assert (indices >= 0).all()
        assert (indices < V).all()
        assert (inc_probs > 0).all()
        assert (inc_probs <= 1).all()
    
    def test_no_tail(self):
        """Test with k_tail=0."""
        N, V = 5, 50
        log_probs = F.log_softmax(torch.randn(N, V), dim=-1)
        
        indices, inc_probs, mask, _ = pps_sample_indices_batched(
            log_probs, k_head=16, k_tail=0
        )
        
        assert indices.shape == (N, 16)
        assert (inc_probs == 1.0).all()  # All deterministic

    def test_pure_importance_sampling(self):
        """Test k_head=0 (pure importance sampling)."""
        N, V = 6, 80
        log_probs = F.log_softmax(torch.randn(N, V), dim=-1)
        
        indices, inc_probs, mask, _ = pps_sample_indices_batched(
            log_probs, k_head=0, k_tail=16, oversample=10
        )
        S = indices.shape[-1]
        
        assert S <= 16
        assert indices.shape == (N, S)
        assert mask.shape == (N, S)
        assert (indices >= 0).all()
        assert (indices < V).all()
        assert (inc_probs > 0).all()
        assert (inc_probs <= 1).all()


class TestHajekEstimate:
    """Tests for Hajek estimator."""
    
    def test_basic(self):
        """Test basic estimation."""
        torch.manual_seed(42)
        N, V = 10, 100
        
        teacher = torch.randn(N, V)
        student = torch.randn(N, V)
        log_probs = F.log_softmax(teacher, dim=-1)
        
        indices, inc_probs, mask, _ = pps_sample_indices_batched(
            log_probs, k_head=16, k_tail=8
        )
        
        teacher_sel = torch.gather(log_probs, -1, indices)
        student_sel = torch.gather(student, -1, indices)
        
        kl, var = hajek_kl_estimate(
            teacher_sel, student_sel, indices, inc_probs, mask
        )
        
        assert kl.shape == (N,)
        assert torch.isfinite(kl).all()
        assert var >= 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_efficient_workflow(self, logits):
        """Test the recommended efficient workflow."""
        teacher, student = logits
        k = 128
        
        # Step 1: Select indices
        indices, teacher_k = select_topk_indices(teacher, k)
        
        # Step 2: "Compute" student logits for indices
        # (In real code, this would be lens.forward(hidden, vocab_indices=indices))
        student_k = torch.gather(student, -1, indices)
        
        # Step 3: Compute KL
        loss = subset_kl_from_gathered(student_k, teacher_k)
        
        # Should work and be finite
        assert torch.isfinite(loss)
        
        # Should be able to backprop
        loss.backward()
        assert student.grad is not None
    
    def test_training_loop(self, logits):
        """Simulate a training loop."""
        teacher, student = logits
        
        optimizer = torch.optim.Adam([student], lr=0.01)
        
        initial_loss = compute_subset_kl(student, teacher, k=128).item()
        
        for _ in range(10):
            optimizer.zero_grad()
            loss = compute_subset_kl(student, teacher, k=128)
            loss.backward()
            optimizer.step()
        
        final_loss = compute_subset_kl(student, teacher, k=128).item()
        
        # Student should learn to match teacher
        assert final_loss < initial_loss
    
    def test_memory_scaling_theoretical(self):
        """Verify theoretical memory savings."""
        B, T, V, k = 2, 256, 128000, 256
        
        full_elements = B * T * V
        subset_elements = B * T * k
        
        reduction = full_elements / subset_elements
        assert reduction == 500  # 128000 / 256 = 500x
