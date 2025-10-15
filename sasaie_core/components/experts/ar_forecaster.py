# Part of the new RegimeVAE architecture as of 2025-10-13

import numpy as np
from collections import deque
from typing import Dict, Any, Optional

from .base_expert import BaseExpert

class ARForecaster:
    """
    Autoregressive forecaster using Ordinary Least Squares.
    
    Simple, interpretable, and computationally efficient.
    Good baseline expert for immediate deployment.
    """
    
    def __init__(self, 
                 regime_id: int,
                 order: int = 10,
                 min_observations: int = 20,
                 buffer_size: int = 1000):
        """
        Initialize AR forecaster.
        
        Args:
            regime_id: Regime this expert specializes in
            order: AR model order (number of lags)
            min_observations: Minimum data before fitting
            buffer_size: Maximum history to retain
        """
        self._regime_id = regime_id
        self.order = order
        self.min_observations = min_observations
        
        # Data buffer (circular for efficiency)
        self.history = deque(maxlen=buffer_size)
        
        # Model parameters
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        
        # Uncertainty estimation
        self.residual_variance: float = 1.0
        
        # Metadata
        self._n_observations = 0
        self._needs_refit = True
    
    @property
    def regime_id(self) -> int:
        return self._regime_id
    
    @property
    def n_observations(self) -> int:
        return self._n_observations
    
    def update(self, observation: float, context: Optional[Dict[str, Any]] = None) -> None:
        """Add observation and mark for refitting"""
        self.history.append(observation)
        self._n_observations += 1
        self._needs_refit = True
        
        # Refit periodically (every 10 observations)
        if self._n_observations % 10 == 0:
            self.fit()
    
    def fit(self) -> None:
        """
        Fit AR model using OLS.
        
        Solves: y_t = β₀ + β₁*y_{t-1} + ... + βₚ*y_{t-p} + ε_t
        """
        if len(self.history) < self.min_observations:
            return
        
        # Prepare design matrix
        data = np.array(list(self.history))
        X, y = self._create_lagged_features(data)
        
        if len(X) < self.order:
            return
        
        # Ordinary Least Squares
        # β = (X'X)^(-1) X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            
            # Add regularization for numerical stability
            ridge_penalty = 1e-6 * np.eye(X.shape[1])
            self.coefficients = np.linalg.solve(XtX + ridge_penalty, Xty)
            
            # Estimate intercept
            predictions = X @ self.coefficients
            self.intercept = np.mean(y - predictions)
            
            # Estimate residual variance (for uncertainty)
            residuals = y - (predictions + self.intercept)
            self.residual_variance = np.var(residuals)
            
            self._needs_refit = False
            
        except np.linalg.LinAlgError:
            # Singular matrix - keep previous parameters
            pass
    
    def _create_lagged_features(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create lagged feature matrix for AR model.
        
        Returns:
            X: [n_samples, order] lagged features
            y: [n_samples] target values
        """
        n = len(data)
        X = np.zeros((n - self.order, self.order))
        y = np.zeros(n - self.order)
        
        for i in range(self.order, n):
            X[i - self.order] = data[i-self.order:i][::-1]  # Most recent first
            y[i - self.order] = data[i]
        
        return X, y
    
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate multi-step ahead forecast.
        
        Uses iterative prediction: ŷ_{t+h} depends on ŷ_{t+h-1}
        """
        if self.coefficients is None or self._needs_refit:
            self.fit()
        
        if self.coefficients is None:
            # Fallback: return last observation
            if len(self.history) > 0:
                return np.full(horizon, self.history[-1])
            return np.zeros(horizon)
        
        # Initialize with recent history
        forecast = []
        window = list(self.history)[-self.order:]
        
        for _ in range(horizon):
            # Predict next value
            if len(window) >= self.order:
                x = np.array(window[-self.order:])[::-1]
                pred = self.intercept + np.dot(self.coefficients, x)
            else:
                # Not enough history - use mean
                pred = np.mean(window) if window else 0.0
            
            forecast.append(pred)
            window.append(pred)
        
        return np.array(forecast)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Serialize model state"""
        return {
            'regime_id': self._regime_id,
            'order': self.order,
            'coefficients': self.coefficients.tolist() if self.coefficients is not None else None,
            'intercept': float(self.intercept),
            'residual_variance': float(self.residual_variance),
            'history': list(self.history),
            'n_observations': self._n_observations
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Deserialize model state"""
        self._regime_id = params['regime_id']
        self.order = params['order']
        
        if params['coefficients'] is not None:
            self.coefficients = np.array(params['coefficients'])
        
        self.intercept = params['intercept']
        self.residual_variance = params['residual_variance']
        self.history = deque(params['history'], maxlen=self.history.maxlen)
        self._n_observations = params['n_observations']
        self._needs_refit = False
    
    def clone(self) -> 'ARForecaster':
        """Create independent copy"""
        new_expert = ARForecaster(
            regime_id=self._regime_id,
            order=self.order,
            min_observations=self.min_observations,
            buffer_size=self.history.maxlen
        )
        new_expert.set_parameters(self.get_parameters())
        return new_expert
    
    def get_uncertainty(self) -> float:
        """Return prediction standard deviation"""
        return np.sqrt(self.residual_variance)
