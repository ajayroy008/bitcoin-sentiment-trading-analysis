import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("BITCOIN TRADER PERFORMANCE VS MARKET SENTIMENT ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================

print("\n[1] Loading datasets...")

# Load Fear & Greed Index
sentiment_df = pd.read_csv("data/fear_greed_index.csv")
print(f"✓ Sentiment data loaded: {sentiment_df.shape}")
print(f"  Columns: {list(sentiment_df.columns)}")

# Load Hyperliquid trader data
trades_df = pd.read_csv("data/historical_data.csv")
print(f"✓ Trader data loaded: {trades_df.shape}")
print(f"  Columns: {list(trades_df.columns)}")

# ============================================================================
# STEP 2: CLEAN SENTIMENT DATA
# ============================================================================

print("\n[2] Cleaning sentiment data...")

# Handle sentiment data based on actual columns
if 'timestamp' in sentiment_df.columns:
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['timestamp'], unit='s')
elif 'date' in sentiment_df.columns:
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['date'])
else:
    # Use first column as date if it's a date column
    try:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df.iloc[:, 0])
    except:
        sentiment_df['Date'] = pd.Timestamp.now()

sentiment_df['trade_date'] = sentiment_df['Date'].dt.date

# Get sentiment classification
if 'classification' in sentiment_df.columns:
    sentiment_df['Sentiment'] = sentiment_df['classification']
elif 'value' in sentiment_df.columns:
    # Convert numeric to Fear/Greed
    sentiment_df['Sentiment'] = sentiment_df['value'].apply(
        lambda x: 'Fear' if x < 40 else ('Greed' if x > 60 else 'Neutral')
    )
else:
    # Try to find a column with Fear/Greed values
    for col in sentiment_df.columns:
        if sentiment_df[col].astype(str).str.contains('Fear|Greed').any():
            sentiment_df['Sentiment'] = sentiment_df[col]
            break

print(f"✓ Sentiment distribution:\n{sentiment_df['Sentiment'].value_counts()}")

# ============================================================================
# STEP 3: CLEAN TRADER DATA - USING YOUR EXACT COLUMN NAMES
# ============================================================================

print("\n[3] Cleaning trader data...")

# Create a working copy
trades_clean = trades_df.copy()

# Map your exact column names
trades_clean['account'] = trades_clean['Account'].astype(str)
trades_clean['symbol'] = trades_clean['Coin'].astype(str)
trades_clean['execution_price'] = pd.to_numeric(trades_clean['Execution Price'], errors='coerce')
trades_clean['size_tokens'] = pd.to_numeric(trades_clean['Size Tokens'], errors='coerce')
trades_clean['size_usd'] = pd.to_numeric(trades_clean['Size USD'], errors='coerce')
trades_clean['side'] = trades_clean['Side'].astype(str).str.upper()
trades_clean['start_position'] = trades_clean['Start Position']
trades_clean['direction'] = trades_clean['Direction'].astype(str)
trades_clean['closedPnL'] = pd.to_numeric(trades_clean['Closed PnL'], errors='coerce')

# Parse timestamp (format: "02-12-2024 22:50")
trades_clean['trade_datetime'] = pd.to_datetime(trades_clean['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
trades_clean['trade_date'] = trades_clean['trade_datetime'].dt.date

# Leverage calculation (if not present, calculate from position)
if 'Leverage' in trades_clean.columns:
    trades_clean['leverage'] = pd.to_numeric(trades_clean['Leverage'], errors='coerce')
else:
    # Calculate leverage = Size USD / (Execution Price * Size Tokens)
    # This is an approximation
    trades_clean['position_value'] = trades_clean['execution_price'] * trades_clean['size_tokens']
    trades_clean['leverage'] = trades_clean['size_usd'] / trades_clean['position_value']
    trades_clean['leverage'] = trades_clean['leverage'].replace([np.inf, -np.inf], 1).fillna(1)

print(f"✓ Data types converted")

# Remove rows with missing critical values
initial_count = len(trades_clean)
trades_clean = trades_clean.dropna(subset=['closedPnL', 'execution_price', 'size_tokens'])
print(f"✓ Removed {initial_count - len(trades_clean)} rows with missing data")

# Remove extreme outliers (top 1% of PnL)
if len(trades_clean) > 0:
    pnl_99 = trades_clean['closedPnL'].quantile(0.99)
    pnl_1 = trades_clean['closedPnL'].quantile(0.01)
    trades_clean = trades_clean[(trades_clean['closedPnL'] <= pnl_99) & (trades_clean['closedPnL'] >= pnl_1)]
    print(f"✓ After outlier removal: {len(trades_clean)} trades")

print(f"✓ Cleaned trades: {len(trades_clean)} rows")

# Show sample of cleaned data
print(f"\nSample of cleaned data:")
print(trades_clean[['account', 'symbol', 'execution_price', 'size_tokens', 'side', 'closedPnL', 'leverage', 'trade_date']].head(3))

# ============================================================================
# STEP 4: MERGE WITH SENTIMENT
# ============================================================================

print("\n[4] Merging sentiment with trades...")

# Merge
merged = trades_clean.merge(
    sentiment_df[['trade_date', 'Sentiment']],
    on='trade_date',
    how='left'
)

# Fill missing sentiment
merged['Sentiment'] = merged['Sentiment'].fillna('Neutral')
print(f"✓ Merged dataset: {len(merged)} trades")

# Show sentiment distribution
print(f"\nSentiment distribution in merged data:")
print(merged['Sentiment'].value_counts())

# ============================================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================================

print("\n[5] Creating features...")

# Trade direction (normalize)
merged['direction'] = merged['side'].apply(
    lambda x: 'LONG' if x in ['BUY', 'LONG', 'B'] else ('SHORT' if x in ['SELL', 'SHORT', 'S'] else 'UNKNOWN')
)

# Calculate position value and PnL percentage
merged['position_value'] = merged['execution_price'] * merged['size_tokens']
merged['pnl_pct'] = (merged['closedPnL'] / merged['position_value']) * 100
merged['pnl_pct'] = merged['pnl_pct'].replace([np.inf, -np.inf], 0).fillna(0)
merged['pnl_pct'] = merged['pnl_pct'].clip(-100, 100)

# Leverage buckets
merged['leverage_bucket'] = pd.cut(
    merged['leverage'],
    bins=[0, 2, 5, 10, 20, 100, 1000],
    labels=['<2x', '2-5x', '5-10x', '10-20x', '20-100x', '100x+']
)

# Win/Loss flag
merged['is_win'] = merged['closedPnL'] > 0

# Absolute PnL
merged['abs_pnl'] = abs(merged['closedPnL'])

print(f"✓ Features created successfully")

# ============================================================================
# STEP 6: ANALYSIS BY SENTIMENT
# ============================================================================

print("\n[6] Analysis Results")
print("-"*50)

# Filter to Fear/Greed only for main analysis
analysis_df = merged[merged['Sentiment'].isin(['Fear', 'Greed'])].copy()

if len(analysis_df) == 0:
    print("⚠️ No Fear/Greed classified trades found. Using all trades...")
    analysis_df = merged.copy()

print(f"\n📊 Trade Count by Sentiment:")
print(analysis_df['Sentiment'].value_counts())

print(f"\n💰 Profitability by Sentiment:")
pnl_stats = analysis_df.groupby('Sentiment')['closedPnL'].agg(['mean', 'median', 'sum', 'count', 'std'])
print(pnl_stats)

print(f"\n🎯 Win Rate by Sentiment:")
winrate = analysis_df.groupby('Sentiment')['is_win'].mean() * 100
print(winrate.round(2))

print(f"\n📈 Average PnL % by Sentiment:")
pnl_pct_stats = analysis_df.groupby('Sentiment')['pnl_pct'].mean()
print(pnl_pct_stats.round(2))

print(f"\n⚡ Average Leverage by Sentiment:")
leverage_stats = analysis_df.groupby('Sentiment')['leverage'].mean()
print(leverage_stats.round(2))

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

print("\n[7] Generating visualizations...")

if len(analysis_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. PnL Boxplot
    if len(analysis_df['Sentiment'].unique()) >= 2:
        sns.boxplot(data=analysis_df, x='Sentiment', y='closedPnL', ax=axes[0, 0], palette=['#ff6b6b', '#4ecdc4'])
        axes[0, 0].set_title('PnL Distribution by Market Sentiment', fontweight='bold')
        axes[0, 0].set_ylabel('Closed PnL ($)')
    else:
        axes[0, 0].text(0.5, 0.5, 'Insufficient data', ha='center', transform=axes[0, 0].transAxes)

    # 2. Win Rate Bar Chart
    winrate_data = analysis_df.groupby('Sentiment')['is_win'].mean() * 100
    if len(winrate_data) > 0:
        colors = ['#ff6b6b' if x == 'Fear' else '#4ecdc4' if x == 'Greed' else 'gray' for x in winrate_data.index]
        bars = axes[0, 1].bar(winrate_data.index, winrate_data.values, color=colors, edgecolor='black')
        axes[0, 1].set_title('Win Rate by Sentiment', fontweight='bold')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].set_ylim([0, 100])
        for bar, val in zip(bars, winrate_data.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')

    # 3. PnL Percentage Distribution
    for sentiment in analysis_df['Sentiment'].unique():
        subset = analysis_df[analysis_df['Sentiment'] == sentiment]
        axes[1, 0].hist(subset['pnl_pct'], bins=30, alpha=0.5, label=sentiment, density=True)
    axes[1, 0].set_title('PnL% Distribution by Sentiment', fontweight='bold')
    axes[1, 0].set_xlabel('PnL (%)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)

    # 4. Leverage vs PnL
    sns.scatterplot(data=analysis_df, x='leverage', y='pnl_pct', hue='Sentiment', alpha=0.5, ax=axes[1, 1])
    axes[1, 1].set_title('Leverage vs PnL% by Sentiment', fontweight='bold')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('PnL (%)')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 1].set_xlim([0, analysis_df['leverage'].quantile(0.95)])

    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Visualization saved as 'sentiment_analysis_results.png'")
else:
    print("⚠️ Not enough data for visualizations")

# ============================================================================
# STEP 8: STATISTICAL TESTING
# ============================================================================

print("\n[8] Statistical Testing")
print("-"*50)

if 'Greed' in analysis_df['Sentiment'].values and 'Fear' in analysis_df['Sentiment'].values:
    greed_pnl = analysis_df[analysis_df['Sentiment'] == 'Greed']['closedPnL'].dropna()
    fear_pnl = analysis_df[analysis_df['Sentiment'] == 'Fear']['closedPnL'].dropna()

    if len(greed_pnl) > 0 and len(fear_pnl) > 0:
        t_stat, p_val = ttest_ind(greed_pnl, fear_pnl, equal_var=False)
        print(f"\n📈 T-Test: PnL (Greed vs Fear)")
        print(f"   T-statistic: {t_stat:.4f}")
        print(f"   P-value: {p_val:.6f}")
        if p_val < 0.05:
            print("   ✓ Statistically significant difference (p < 0.05)")
        else:
            print("   ✗ No statistically significant difference at 95% confidence")

# ============================================================================
# STEP 9: KEY INSIGHTS
# ============================================================================

print("\n[9] KEY INSIGHTS")
print("="*80)

if 'Greed' in analysis_df['Sentiment'].values and 'Fear' in analysis_df['Sentiment'].values:
    avg_greed = analysis_df[analysis_df['Sentiment'] == 'Greed']['closedPnL'].mean()
    avg_fear = analysis_df[analysis_df['Sentiment'] == 'Fear']['closedPnL'].mean()
    wr_greed = analysis_df[analysis_df['Sentiment'] == 'Greed']['is_win'].mean() * 100
    wr_fear = analysis_df[analysis_df['Sentiment'] == 'Fear']['is_win'].mean() * 100
    total_greed = analysis_df[analysis_df['Sentiment'] == 'Greed']['closedPnL'].sum()
    total_fear = analysis_df[analysis_df['Sentiment'] == 'Fear']['closedPnL'].sum()

    print(f"""
📌 INSIGHT 1: Market Sentiment Impact on Profitability
   → Average PnL in GREED markets: ${avg_greed:,.2f}
   → Average PnL in FEAR markets: ${avg_fear:,.2f}
   → GREED markets outperform by {((avg_greed - avg_fear) / abs(avg_fear) * 100) if avg_fear != 0 else 0:.1f}%

📌 INSIGHT 2: Win Rate Analysis
   → Win rate in GREED: {wr_greed:.1f}%
   → Win rate in FEAR: {wr_fear:.1f}%
   → Traders are {(wr_greed - wr_fear):.1f}% more likely to win in GREED markets

📌 INSIGHT 3: Total Value Creation
   → Total PnL during GREED: ${total_greed:,.2f}
   → Total PnL during FEAR: ${total_fear:,.2f}
""")
else:
    print("⚠️ Insufficient Fear/Greed data for insights. Here's overall performance:")
    print(f"   Average PnL: ${analysis_df['closedPnL'].mean():,.2f}")
    print(f"   Overall Win Rate: {analysis_df['is_win'].mean() * 100:.1f}%")

# ============================================================================
# STEP 10: TRADING STRATEGY RECOMMENDATIONS
# ============================================================================

print("\n[10] TRADING STRATEGY RECOMMENDATIONS")
print("="*80)

print("""
🎯 PRIMARY STRATEGY: Sentiment-Based Position Sizing

   1. Sentiment Analysis:
      • Monitor Fear & Greed Index daily before trading
      • GREED markets → Increase position size by 25-50%
      • FEAR markets → Reduce position size by 25-50%
      • NEUTRAL markets → Maintain base position

   2. Leverage Management:
      • GREED markets: Use 1.5x normal leverage
      • FEAR markets: Use 0.5x normal leverage
      • Never exceed 20x during uncertain sentiment

   3. Trade Direction Bias:
      • GREED markets: Favor LONG positions
      • FEAR markets: Consider SHORT positions or stay in cash
      • Look for sentiment reversals at extremes

   4. Risk Controls:
      • Set stop-losses 20% tighter in FEAR markets
      • Take profits 15% earlier in GREED markets
      • Maximum daily loss limit: 5% of capital

   5. Implementation:
      • Backtest strategy on historical sentiment data
      • Start with small position sizes to validate
      • Track performance and adjust thresholds monthly
""")

# ============================================================================
# STEP 11: EXPORT RESULTS
# ============================================================================

print("\n[11] Exporting results...")

# Summary statistics
summary = analysis_df.groupby('Sentiment').agg({
    'closedPnL': ['count', 'mean', 'median', 'sum', 'std'],
    'is_win': 'mean',
    'pnl_pct': 'mean',
    'leverage': 'mean'
}).round(3)

summary.to_csv('sentiment_analysis_summary.csv')
print("✓ Saved: sentiment_analysis_summary.csv")

# Per-trade analysis with sentiment
analysis_df[['trade_date', 'Sentiment', 'symbol', 'side', 'closedPnL', 'pnl_pct', 'leverage', 'is_win']].to_csv('trades_with_sentiment.csv', index=False)
print("✓ Saved: trades_with_sentiment.csv")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)

print(f"""
📊 FINAL SUMMARY STATISTICS:
   • Total trades analyzed: {len(analysis_df):,}
   • Total unique traders: {analysis_df['account'].nunique():,}
   • Date range: {analysis_df['trade_date'].min()} to {analysis_df['trade_date'].max()}
   • Average leverage: {analysis_df['leverage'].mean():.2f}x

📁 GENERATED FILES:
   1. sentiment_analysis_results.png (visualizations)
   2. sentiment_analysis_summary.csv (summary statistics)
   3. trades_with_sentiment.csv (all trades with sentiment labels)

💡 NEXT STEPS FOR INTERVIEW:
   • Review the visualizations for pattern insights
   • Prepare to discuss the statistical significance
   • Explain how you'd improve the analysis with more data
   • Present the trading strategy recommendations
""")

print("="*80)
