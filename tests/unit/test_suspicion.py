"""Unit tests for Suspicion Field model"""

import pytest
import numpy as np
from blackpanther.core.suspicion import SuspicionField


class TestSuspicionField:
    """Test suite for SuspicionField class"""
    
    def test_initialization(self):
        """Test proper initialization"""
        field = SuspicionField(width=10, height=10, D=0.2, r=0.1)
        
        assert field.width == 10
        assert field.height == 10
        assert field.D == 0.2
        assert field.r == 0.1
    
    def test_reset(self):
        """Test reset to zero field"""
        field = SuspicionField(width=5, height=5)
        field.reset()
        
        assert np.all(field.field == 0)
        assert len(field.history) == 1
    
    def test_laplacian(self):
        """Test Laplacian calculation"""
        field = SuspicionField(width=3, height=3)
        
        # Create a peak in the center
        test_field = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        laplacian = field._laplacian_2d(test_field)
        
        # Center should have negative Laplacian (hotter than neighbors)
        assert laplacian[1, 1] < 0
        
        # Neighbors should have positive Laplacian
        assert laplacian[1, 0] > 0
        assert laplacian[1, 2] > 0
        assert laplacian[0, 1] > 0
        assert laplacian[2, 1] > 0
    
    def test_diffusion(self):
        """Test that diffusion spreads suspicion"""
        field = SuspicionField(width=5, height=5, D=0.5, noise_scale=0.0)
        field.reset()
        
        # Set a single hotspot
        field._field[2, 2] = 1.0
        
        # Step with no attacks
        field.step(attack_positions=[], knowledge=0.0, access=0.0)
        
        # Neighbors should have increased
        assert field._field[2, 1] > 0
        assert field._field[2, 3] > 0
        assert field._field[1, 2] > 0
        assert field._field[3, 2] > 0
    
    def test_attack_suppression(self):
        """Test that attacks reduce suspicion at location"""
        field = SuspicionField(width=10, height=10, delta=0.1, noise_scale=0.0)
        field.reset()
        
        # Uniform field
        field._field.fill(0.5)
        
        # Attack at (0.5, 0.5)
        attacks = [(0.5, 0.5, 1.0)]
        field.step(attacks, knowledge=0.8, access=0.7)
        
        # Get attack location
        ix, iy = 5, 5
        
        # Suspicion at attack point should be lower than before
        assert field._field[iy, ix] < 0.5
    
    def test_hotspot_detection(self):
        """Test detection of hotspots (S > 0.7)"""
        field = SuspicionField(width=5, height=5)
        field.reset()
        
        # Create hotspots
        field._field[1, 1] = 0.8
        field._field[3, 3] = 0.9
        field._field[2, 2] = 0.5  # Not a hotspot
        
        state = field._create_state(timestamp=1.0, episode=1)
        
        # Should detect 2 hotspots
        assert len(state.hotspots) == 2
        
        hotspot_coords = [(x, y) for (x, y, _) in state.hotspots]
        assert (1, 1) in hotspot_coords
        assert (3, 3) in hotspot_coords
        assert (2, 2) not in hotspot_coords