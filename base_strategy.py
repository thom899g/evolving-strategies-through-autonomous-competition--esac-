"""
Base Strategy Abstract Class
Defines the interface and common functionality for all trading strategies
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

@dataclass
class Trade:
    """Data class representing a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    signal_type: SignalType = SignalType.NEUTRAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyMetadata:
    """Metadata for strategy tracking"""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_score: float = 0.0
    backtest_count: int = 0
    live_performance: Dict[str, float] = field(default_factory=dict)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    All strategies must implement the generate_signals method.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 parameters: Dict[str, Any],
                 metadata: Optional[StrategyMetadata] = None):
        """
        Initialize strategy with unique ID and parameters
        
        Args:
            strategy_id: Unique identifier for the strategy
            parameters: Dictionary of strategy-specific parameters
            metadata: Optional strategy metadata
        """
        self.strategy_id = strategy_id
        self.parameters = self._validate_parameters(parameters)
        self.metadata = metadata or StrategyMetadata()
        self.metadata.last_updated = datetime.utcnow()
        
        # Initialize state variables
        self.current_position = 0.0
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve =