"""Unit tests for mathematical core"""

import pytest
import numpy as np
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.core.access import AccessPropagation


class TestKnowledgeEvolution:
    """Test knowledge model"""
    
    def test_initialization(self):
        model = KnowledgeEvolution(alpha=0.1, beta=0.01)
        assert model.alpha == 0.1
        assert model.beta == 0.01
    
    def test_knowledge_growth(self):
        model = KnowledgeEvolution(noise_scale=0.0)
        model.reset()
        initial = model.knowledge
        model.step(suspicion=0.1, learning_effort=1.0)
        assert model.knowledge > initial


class TestSuspicionField:
    """Test suspicion model"""
    
    def test_diffusion(self):
        field = SuspicionField(width=5, height=5, noise_scale=0.0)
        field.reset()
        field._field[2, 2] = 1.0
        field.step(attack_positions=[], knowledge=0, access=0)
        assert field._field[2, 1] > 0  # Neighbor heated up


class TestAccessPropagation:
    """Test access model"""
    
    def test_add_host(self):
        access = AccessPropagation()
        access.add_host("test", initial_access=0.3)
        assert "test" in access.hosts
        assert access.hosts["test"].access == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
