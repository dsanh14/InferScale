"""Tests for speculative decoding verification logic."""

import pytest
import torch

from app.speculative_service.verifier import verify_draft_tokens, VerificationResult


class TestVerifyDraftTokens:
    def _make_dummy_model(self, vocab_size: int = 100, seq_agree_up_to: int = 3):
        """Build a mock 'model' that returns fixed logits."""

        class MockModel(torch.nn.Module):
            def __init__(self, agree_up_to: int):
                super().__init__()
                self.agree_up_to = agree_up_to
                self.vocab_size = vocab_size

            def forward(self, input_ids):
                batch, seq_len = input_ids.shape
                logits = torch.zeros(batch, seq_len, self.vocab_size)
                for i in range(seq_len):
                    logits[0, i, 42] = 10.0  # default prediction is token 42
                return type("Out", (), {"logits": logits})()

        return MockModel(seq_agree_up_to)

    def test_all_accepted(self):
        model = self._make_dummy_model()
        input_ids = torch.tensor([[1, 2, 3]])
        draft_tokens = [42, 42, 42]
        result = verify_draft_tokens(model, input_ids, draft_tokens)
        assert result.accepted_count == 3
        assert result.proposed_count == 3
        assert result.bonus_token == 42

    def test_partial_acceptance(self):
        model = self._make_dummy_model()
        input_ids = torch.tensor([[1, 2, 3]])
        draft_tokens = [42, 42, 99]  # third token diverges
        result = verify_draft_tokens(model, input_ids, draft_tokens)
        assert result.accepted_count == 2
        assert result.proposed_count == 3
        assert result.bonus_token == 42

    def test_no_acceptance(self):
        model = self._make_dummy_model()
        input_ids = torch.tensor([[1, 2, 3]])
        draft_tokens = [99]  # first token diverges
        result = verify_draft_tokens(model, input_ids, draft_tokens)
        assert result.accepted_count == 0
        assert result.proposed_count == 1
        assert result.bonus_token == 42

    def test_empty_draft(self):
        model = self._make_dummy_model()
        input_ids = torch.tensor([[1, 2, 3]])
        result = verify_draft_tokens(model, input_ids, [])
        assert result.accepted_count == 0
        assert result.proposed_count == 0
