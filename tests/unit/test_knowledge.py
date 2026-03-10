"""Unit tests for Knowledge Evolution model"""

import pytest
import numpy as np
from blackpanther.core.knowledge import KnowledgeEvolution


class TestKnowledgeEvolution:
    """Test suite for KnowledgeEvolution class"""
    
    def test_initialization(self):
        """Test proper parameter initialization"""
        model = KnowledgeEvolution(alpha=0.2, beta=0.02, gamma=0.1, k_max=50.0)
        
        assert model.alpha == 0.2
        assert model.beta == 0.02
        assert model.gamma == 0.1
        assert model.k_max == 50.0
    
    def test_invalid_parameters(self):
        """Test validation of invalid parameters"""
        with pytest.raises(ValueError):
            KnowledgeEvolution(alpha=3.0)  # Too high
        
        with pytest.raises(ValueError):
            KnowledgeEvolution(beta=-0.1)  # Negative
        
        with pytest.raises(ValueError):
            KnowledgeEvolution(k_max=0)  # Zero
    
    def test_reset(self):
        """Test reset functionality"""
        model = KnowledgeEvolution()
        state = model.reset(initial_knowledge=25.0)
        
        assert state.knowledge == 25.0
        assert len(model.history) == 1
    
    def test_knowledge_growth(self):
        """Test that knowledge increases with positive learning"""
        model = KnowledgeEvolution(noise_scale=0.0)  # Deterministic
        model.reset()
        
        initial = model.knowledge
        for _ in range(50):
            model.step(suspicion=0.2, learning_action=1.0)
        
        assert model.knowledge > initial
    
    def test_knowledge_decay(self):
        """Test that knowledge decreases without learning"""
        model = KnowledgeEvolution(beta=0.1, noise_scale=0.0)
        model.reset(initial_knowledge=50.0)
        
        initial = model.knowledge
        for _ in range(50):
            model.step(suspicion=0.0, learning_action=0.0)
        
        assert model.knowledge < initial
    
    def test_saturation(self):
        """Test that knowledge never exceeds K_max"""
        model = KnowledgeEvolution(k_max=100.0, noise_scale=0.0)
        model.reset()
        
        for _ in range(1000):
            state = model.step(suspicion=0.5, learning_action=2.0)
            assert state.knowledge <= model.k_max
    
    def test_suspicion_effect(self):
        """Test that higher suspicion increases learning"""
        model = KnowledgeEvolution(gamma=0.1, noise_scale=0.0)
        model.reset()
        
        # Low suspicion
        low_results = []
        for _ in range(50):
            state = model.step(suspicion=0.1, learning_action=0.5)
            low_results.append(state.knowledge)
        
        model.reset()
        
        # High suspicion
        high_results = []
        for _ in range(50):
            state = model.step(suspicion=0.9, learning_action=0.5)
            high_results.append(state.knowledge)
        
        assert high_results[-1] > low_results[-1]