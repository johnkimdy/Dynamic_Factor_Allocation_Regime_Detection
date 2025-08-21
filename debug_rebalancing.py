#!/usr/bin/env python3.11
"""Debug rebalancing behavior"""

from helix_factor_strategy import HelixFactorStrategy
import logging

# Set up minimal logging
logging.basicConfig(level=logging.WARNING)

def debug_rebalancing():
    strategy = HelixFactorStrategy(rebalance_threshold=0.02)  # 2% threshold
    
    print("=== DEBUGGING REBALANCING BEHAVIOR ===\n")
    
    # Test with a shorter period to see what's happening
    print("Running short backtest (2024-01-01 to 2024-03-31)...")
    
    try:
        results = strategy.backtest('2024-01-01', '2024-03-31')
        
        print("Results:")
        print("- Total Return: {:.2%}".format(results['total_return']))
        print("- Rebalances: {}".format(results['n_rebalances']))
        print("- Rebalance dates: {}".format(len(results['rebalance_dates'])))
        
        print("\nWeight history details:")
        for i, (date, weights) in enumerate(results['weights_history']):
            print("Rebalance {}: {} - Weights: {}".format(
                i+1, date.strftime('%Y-%m-%d'), 
                {k: "{:.3f}".format(v) for k, v in weights.items()}
            ))
        
        # Check if weights are changing at all
        if len(results['weights_history']) > 0:
            first_weights = results['weights_history'][0][1]
            print(f"\nFirst allocation: {first_weights}")
            
            # Check if it's just equal weighting
            n_assets = len(first_weights)
            equal_weight = 1.0 / n_assets
            is_equal_weight = all(abs(w - equal_weight) < 0.01 for w in first_weights.values())
            print(f"Is equal weighted? {is_equal_weight} (1/{n_assets} = {equal_weight:.3f})")
        
        # Let's also check what regimes were detected
        print(f"\nRegime models fitted: {len(strategy.regime_models)}")
        for etf, model in strategy.regime_models.items():
            if hasattr(model, 'regimes_') and model.regimes_ is not None:
                unique_regimes = model.regimes_.nunique()
                regime_changes = (model.regimes_.diff() != 0).sum()
                print(f"{etf}: {unique_regimes} unique regimes, {regime_changes} regime changes")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rebalancing()