"""
Complete Retail Price Optimization System
A comprehensive ML-powered pricing solution with interactive web interface
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import base64
import io
import json
import warnings
warnings.filterwarnings('ignore')

class RetailPriceOptimizer:
    """
    Advanced Retail Price Optimization System using Multiple ML Models
    """
    
    def __init__(self):
        """Initialize the price optimizer with ML models"""
        self.models = {}
        self.model_performance = {}
        self.data = None
        self.optimal_price_cache = {}
        print("Initializing Retail Price Optimization System...")
        
    def generate_realistic_dataset(self, n_samples=2000):
        """
        Generate realistic retail dataset with complex relationships
        """
        np.random.seed(42)
        
        # Generate product categories
        categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
        category = np.random.choice(categories, n_samples)
        
        # Base price varies by category
        category_multiplier = {
            'Electronics': 1.5, 'Clothing': 1.0, 'Home': 1.2, 
            'Sports': 1.1, 'Books': 0.6
        }
        
        base_prices = []
        for cat in category:
            base_prices.append(np.random.uniform(500, 8000) * category_multiplier[cat])
        
        price = np.array(base_prices)
        
        # Cost structure (60-80% of price with some variation)
        cost_ratio = np.random.uniform(0.6, 0.8, n_samples)
        cost = price * cost_ratio
        
        # Realistic demand model with proper price elasticity
        seasonal_factor = 1 + 0.3 * np.sin(np.random.uniform(0, 2*np.pi, n_samples))
        
        # Price elasticity varies by category (more realistic)
        elasticity_by_category = {
            'Electronics': -0.8,   # Less elastic (luxury/necessity)
            'Clothing': -1.2,      # Moderately elastic
            'Home': -0.9,          # Less elastic
            'Sports': -1.5,        # More elastic (discretionary)
            'Books': -1.8          # Highly elastic (many substitutes)
        }
        
        # Calculate base demand (higher for lower-priced categories)
        base_demand_by_category = {
            'Electronics': 800,
            'Clothing': 1200,
            'Home': 900,
            'Sports': 1000,
            'Books': 1500
        }
        
        demand = []
        for i, cat in enumerate(category):
            # Get category-specific parameters
            elasticity = elasticity_by_category[cat]
            base_demand = base_demand_by_category[cat] * seasonal_factor[i]
            
            # Realistic demand curve: D = base_demand * (price/reference_price)^elasticity
            reference_price = 2000  # Reference price in rupees
            price_ratio = price[i] / reference_price
            
            # Apply price elasticity formula
            demand_value = base_demand * (price_ratio ** elasticity)
            
            # Add some randomness but keep it realistic
            demand_value += np.random.normal(0, demand_value * 0.1)  # 10% noise
            
            # Ensure minimum demand (even luxury items have some demand)
            demand_value = max(demand_value, base_demand * 0.05)
            
            demand.append(demand_value)
        
        demand = np.array(demand)
        
        # Sales conversion rate (what % of demand converts to actual sales)
        # Higher prices typically have lower conversion rates
        conversion_rates = []
        for i, cat in enumerate(category):
            # Base conversion rate varies by category
            base_conversion = {
                'Electronics': 0.15,  # Lower conversion (high consideration)
                'Clothing': 0.25,     # Medium conversion
                'Home': 0.18,         # Lower conversion (high consideration)
                'Sports': 0.22,       # Medium conversion
                'Books': 0.35         # Higher conversion (impulse buy)
            }[cat]
            
            # Price affects conversion rate (higher prices = lower conversion)
            price_effect = max(0.1, 1 - (price[i] - 1000) / 10000)  # Decreases as price increases
            
            # Marketing and competition effects
            marketing_effect = np.random.uniform(0.8, 1.3)
            competition_effect = np.random.uniform(0.7, 1.1)
            
            final_conversion = base_conversion * price_effect * marketing_effect * competition_effect
            final_conversion = max(0.05, min(0.5, final_conversion))  # Keep realistic bounds
            
            conversion_rates.append(final_conversion)
        
        conversion_rates = np.array(conversion_rates)
        
        # Calculate actual sales
        sales = demand * conversion_rates
        
        # Add some market variability
        market_variability = np.random.uniform(0.8, 1.2, n_samples)
        sales = sales * market_variability
        
        sales = np.maximum(sales, 1)  # Minimum 1 unit sold
        
        # Calculate profit and margin
        profit = (price - cost) * sales
        margin = ((price - cost) / price) * 100
        
        # Store elasticity data for each row
        price_elasticity_values = [elasticity_by_category[cat] for cat in category]
        
        # Create comprehensive dataset
        self.data = pd.DataFrame({
            'category': category,
            'price': price,
            'cost': cost,
            'demand': demand,
            'sales': sales,
            'profit': profit,
            'margin': margin,
            'seasonal_factor': seasonal_factor,
            'marketing_effect': np.random.uniform(0.8, 1.3, n_samples),
            'competition_effect': np.random.uniform(0.7, 1.1, n_samples),
            'conversion_rate': conversion_rates,
            'price_elasticity': price_elasticity_values
        })
        
        print(f"Generated {n_samples} realistic retail data points")
        print(f"Price range: Rs.{self.data['price'].min():.2f} - Rs.{self.data['price'].max():.2f}")
        print(f"Average profit: Rs.{self.data['profit'].mean():.2f}")
        
        return self.data
    
    def train_ml_models(self):
        """
        Train multiple ML models for sales prediction
        """
        if self.data is None:
            self.generate_realistic_dataset()
        
        # Prepare features for ML models
        feature_columns = ['price', 'cost', 'demand', 'seasonal_factor', 'marketing_effect']
        X = self.data[feature_columns].values
        y = self.data['sales'].values
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize ML models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                random_state=42
            )
        }
        
        print("\nTraining ML Models...")
        print("-" * 60)
        
        # Train and evaluate each model
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate performance metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Store model and performance
            self.models[name] = model
            self.model_performance[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
            
            print(f"{name:20} | R² Score: {test_r2:.4f} | RMSE: {test_rmse:.2f}")
        
        print("-" * 60)
        print("All ML models trained successfully!")
        
        # Find best performing model
        best_model = max(self.model_performance.items(), 
                        key=lambda x: x[1]['test_r2'])
        print(f"Best Model: {best_model[0]} (R-squared = {best_model[1]['test_r2']:.4f})")
        
        return self.model_performance
    
    def predict_sales(self, price, cost, demand=None, model_name='Gradient Boosting'):
        """
        Predict sales for given price and cost
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        # Use average values if not provided
        if demand is None:
            demand = self.data['demand'].mean()
        
        seasonal_factor = self.data['seasonal_factor'].mean()
        marketing_effect = self.data['marketing_effect'].mean()
        
        # Prepare features
        features = np.array([[price, cost, demand, seasonal_factor, marketing_effect]])
        
        # Make prediction
        prediction = self.models[model_name].predict(features)[0]
        
        return max(prediction, 0)  # Ensure non-negative sales
    
    def calculate_profit_metrics(self, price, cost, predicted_sales):
        """
        Calculate comprehensive profit metrics
        """
        profit = (price - cost) * predicted_sales
        margin = ((price - cost) / price) * 100 if price > 0 else 0
        roi = (profit / (cost * predicted_sales)) * 100 if cost > 0 and predicted_sales > 0 else 0
        
        return {
            'profit': profit,
            'margin': margin,
            'roi': roi,
            'revenue': price * predicted_sales
        }
    
    def find_optimal_price(self, cost, demand=None, price_range=(500, 15000), 
                          model_name='Gradient Boosting', resolution=200):
        """
        Find optimal price that maximizes profit using advanced optimization
        """
        # Create cache key
        cache_key = f"{cost}_{demand}_{price_range}_{model_name}"
        if cache_key in self.optimal_price_cache:
            return self.optimal_price_cache[cache_key]
        
        if demand is None:
            demand = self.data['demand'].mean()
        
        # Generate price range for optimization with more focus on realistic range
        # Use both linear and logarithmic spacing for better coverage
        linear_prices = np.linspace(price_range[0], price_range[1], resolution//2)
        log_prices = np.logspace(np.log10(price_range[0]), np.log10(price_range[1]), resolution//2)
        prices = np.unique(np.concatenate([linear_prices, log_prices]))
        prices = np.sort(prices)
        
        profits = []
        sales_predictions = []
        margins = []
        
        # Test each price point
        for price in prices:
            predicted_sales = self.predict_sales(price, cost, demand, model_name)
            metrics = self.calculate_profit_metrics(price, cost, predicted_sales)
            
            profits.append(metrics['profit'])
            sales_predictions.append(predicted_sales)
            margins.append(metrics['margin'])
        
        # Find optimal price
        optimal_idx = np.argmax(profits)
        optimal_price = prices[optimal_idx]
        max_profit = profits[optimal_idx]
        optimal_sales = sales_predictions[optimal_idx]
        optimal_margin = margins[optimal_idx]
        
        # Calculate additional metrics
        optimal_metrics = self.calculate_profit_metrics(optimal_price, cost, optimal_sales)
        
        result = {
            'optimal_price': optimal_price,
            'max_profit': max_profit,
            'predicted_sales': optimal_sales,
            'margin': optimal_margin,
            'revenue': optimal_metrics['revenue'],
            'roi': optimal_metrics['roi'],
            'price_range': prices.tolist(),
            'profit_range': profits,
            'sales_range': sales_predictions,
            'margin_range': margins
        }
        
        # Cache result
        self.optimal_price_cache[cache_key] = result
        
        return result
    
    def analyze_price_sensitivity(self, base_price, cost, demand=None):
        """
        Analyze how profit changes with price variations
        """
        if demand is None:
            demand = self.data['demand'].mean()
        
        price_variations = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])
        results = []
        
        for variation in price_variations:
            test_price = base_price * (1 + variation/100)
            predicted_sales = self.predict_sales(test_price, cost, demand)
            metrics = self.calculate_profit_metrics(test_price, cost, predicted_sales)
            
            results.append({
                'price_change': variation,
                'new_price': test_price,
                'predicted_sales': predicted_sales,
                'profit': metrics['profit'],
                'margin': metrics['margin']
            })
        
        return results

# Initialize Flask app
app = Flask(__name__)

# Create global optimizer instance
optimizer = RetailPriceOptimizer()
optimizer.generate_realistic_dataset(2000)
optimizer.train_ml_models()

@app.route('/')
def home():
    """Home page with dashboard overview"""
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    """Price prediction interface"""
    return render_template('predict.html')



@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for sales prediction"""
    try:
        data = request.json
        price = float(data['price'])
        cost = float(data['cost'])
        demand = float(data.get('demand', optimizer.data['demand'].mean()))
        model = data.get('model', 'Gradient Boosting')
        
        # Make prediction
        predicted_sales = optimizer.predict_sales(price, cost, demand, model)
        metrics = optimizer.calculate_profit_metrics(price, cost, predicted_sales)
        
        # Find optimal price for comparison
        optimal_result = optimizer.find_optimal_price(cost, demand)
        
        # Determine price recommendation
        price_diff = price - optimal_result['optimal_price']
        if abs(price_diff) <= 2:
            recommendation = "optimal"
            message = "Your price is very close to optimal!"
        elif price_diff > 0:
            recommendation = "too_high"
            message = f"Consider reducing price by ₹{abs(price_diff):.2f}"
        else:
            recommendation = "too_low"
            message = f"Consider increasing price by ₹{abs(price_diff):.2f}"
        
        return jsonify({
            'success': True,
            'predicted_sales': round(predicted_sales, 2),
            'profit': round(metrics['profit'], 2),
            'margin': round(metrics['margin'], 2),
            'revenue': round(metrics['revenue'], 2),
            'roi': round(metrics['roi'], 2),
            'optimal_price': round(optimal_result['optimal_price'], 2),
            'max_profit': round(optimal_result['max_profit'], 2),
            'recommendation': recommendation,
            'message': message,
            'price_difference': round(price_diff, 2)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """API endpoint for price optimization"""
    try:
        data = request.json
        cost = float(data['cost'])
        demand = float(data.get('demand', optimizer.data['demand'].mean()))
        model = data.get('model', 'Gradient Boosting')
        min_price = float(data.get('min_price', 500))
        max_price = float(data.get('max_price', 15000))
        
        # Find optimal price
        result = optimizer.find_optimal_price(
            cost, demand, (min_price, max_price), model
        )
        
        # Generate visualization
        plot_url = generate_optimization_plots(result)
        
        return jsonify({
            'success': True,
            'optimal_price': round(result['optimal_price'], 2),
            'max_profit': round(result['max_profit'], 2),
            'predicted_sales': round(result['predicted_sales'], 2),
            'margin': round(result['margin'], 2),
            'revenue': round(result['revenue'], 2),
            'roi': round(result['roi'], 2),
            'plot_url': plot_url
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sensitivity', methods=['POST'])
def api_sensitivity():
    """API endpoint for price sensitivity analysis"""
    try:
        data = request.json
        base_price = float(data['price'])
        cost = float(data['cost'])
        demand = float(data.get('demand', optimizer.data['demand'].mean()))
        
        # Analyze price sensitivity
        sensitivity_data = optimizer.analyze_price_sensitivity(base_price, cost, demand)
        
        return jsonify({
            'success': True,
            'sensitivity_data': sensitivity_data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API endpoint for dataset statistics"""
    try:
        stats = {
            'total_products': len(optimizer.data),
            'avg_price': round(optimizer.data['price'].mean(), 2),
            'avg_profit': round(optimizer.data['profit'].mean(), 2),
            'avg_margin': round(optimizer.data['margin'].mean(), 2),
            'price_range': {
                'min': round(optimizer.data['price'].min(), 2),
                'max': round(optimizer.data['price'].max(), 2)
            },
            'categories': optimizer.data['category'].value_counts().to_dict(),
            'model_performance': optimizer.model_performance
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_optimization_plots(result):
    """Generate beautiful optimization visualization"""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Price Optimization Analysis', fontsize=16, fontweight='bold')
    
    prices = result['price_range']
    profits = result['profit_range']
    sales = result['sales_range']
    margins = result['margin_range']
    optimal_price = result['optimal_price']
    
    # Plot 1: Price vs Profit
    ax1.plot(prices, profits, 'b-', linewidth=3, alpha=0.7, label='Profit Curve')
    ax1.scatter(optimal_price, result['max_profit'], 
               color='red', s=150, zorder=5, label='Optimal Point')
    ax1.set_xlabel('Price (₹)', fontsize=12)
    ax1.set_ylabel('Profit (₹)', fontsize=12)
    ax1.set_title('Price vs Profit', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Price vs Sales
    ax2.plot(prices, sales, 'g-', linewidth=3, alpha=0.7, label='Sales Curve')
    ax2.scatter(optimal_price, result['predicted_sales'], 
               color='red', s=150, zorder=5, label='Optimal Point')
    ax2.set_xlabel('Price (₹)', fontsize=12)
    ax2.set_ylabel('Predicted Sales', fontsize=12)
    ax2.set_title('Price vs Predicted Sales', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Price vs Margin
    ax3.plot(prices, margins, 'purple', linewidth=3, alpha=0.7, label='Margin %')
    ax3.scatter(optimal_price, result['margin'], 
               color='red', s=150, zorder=5, label='Optimal Point')
    ax3.set_xlabel('Price (₹)', fontsize=12)
    ax3.set_ylabel('Profit Margin (%)', fontsize=12)
    ax3.set_title('Price vs Profit Margin', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Summary metrics
    ax4.axis('off')
    summary_text = f"""
    OPTIMIZATION RESULTS
    
    Optimal Price: ₹{optimal_price:.2f}
    Maximum Profit: ₹{result['max_profit']:.2f}
    Predicted Sales: {result['predicted_sales']:.0f}
    Profit Margin: {result['margin']:.1f}%
    Revenue: ₹{result['revenue']:.2f}
    ROI: {result['roi']:.1f}%
    """
    ax4.text(0.1, 0.9, summary_text, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    # Convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

if __name__ == '__main__':
    print("\n" + "="*70)
    print("RETAIL PRICE OPTIMIZATION SYSTEM")
    print("="*70)
    print("Starting web server...")
    print("Home: http://localhost:5000")
    print("Predictions: http://localhost:5000/predict")
    print("Press Ctrl+C to stop")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
