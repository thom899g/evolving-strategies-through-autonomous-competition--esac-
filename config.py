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