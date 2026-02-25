# Evolving Strategies through Autonomous Competition (ESAC)

## Objective
A framework where AI generates multiple trading strategies which compete in simulated environments. The most successful strategies evolve and improve over time through a process of natural selection.

## Strategy
Develop AI algorithms to generate diverse strategies, create competitive environments for testing, implement feedback loops based on strategy performance, and ensure scalability across various markets.

## Execution Output
SUMMARY: I've architected the ESAC (Evolving Strategies through Autonomous Competition) framework with production-grade Python modules featuring type hints, comprehensive error handling, logging, and Firebase integration for state management. The system includes core components for strategy generation, simulation environments, evolutionary algorithms, and performance tracking with strict separation of concerns.

OUTPUT:

### FILE: config.py
```python
"""
ESAC Configuration and Constants
Centralized configuration management with environment variables and type-safe defaults
"""
import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

class EvolutionMethod(Enum):
    GENETIC = "genetic"
    GRADIENT = "gradient"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class ESACConfig:
    """Main configuration dataclass with type hints and validation"""
    
    # Evolution Parameters
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elitism_count: int = 5
    
    # Simulation Parameters
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1%
    max_position_size: float = 0.2  # 20% of capital
    risk_free_rate: float = 0.02  # 2% annual
    
    # Performance Metrics
    metrics: tuple = (
        'sharpe_ratio',
        'max_drawdown',
        'profit_factor',
        'win_rate',
        'calmar_ratio',
        'sortino_ratio'
    )
    
    # Firebase Configuration
    firebase_project_id: Optional[str] = None
    firestore_collection: str = "esac_strategies"
    realtime_db_url: Optional[str] = None
    
    # Logging Configuration
    log_level: int = logging.INFO
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_rates()
        self._validate_sizes()
        self._setup_environment()
    
    def _validate_rates(self) -> None:
        """Validate probability rates are within bounds"""
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError(f"Mutation rate {self.mutation_rate} must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError(f"Crossover rate {self.crossover_rate} must be between 0 and 1")
        if not 0 <= self.transaction_cost <= 0.1:
            raise ValueError(f"Transaction cost {self.transaction_cost} must be between 0 and 0.1")
    
    def _validate_sizes(self) -> None:
        """Validate size parameters"""
        if self.population_size < 10:
            raise ValueError(f"Population size {self.population_size} must be at least 10")
        if self.initial_capital <= 0:
            raise ValueError(f"Initial capital {self.initial_capital} must be positive")
    
    def _setup_environment(self) -> None:
        """Load environment variables with fallbacks"""
        self.firebase_project_id = os.getenv('FIREBASE_PROJECT_ID', self.firebase_project_id)
        self.realtime_db_url = os.getenv('FIREBASE_DATABASE_URL', self.realtime_db_url)
        
        # Set log level from environment
        log_level_str = os.getenv('ESAC_LOG_LEVEL', '').upper()
        if log_level_str in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            self.log_level = getattr(logging, log_level_str)

# Global configuration instance
config = ESACConfig()
```

### FILE: base_strategy.py
```python
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