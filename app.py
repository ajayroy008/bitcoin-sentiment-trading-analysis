from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import os
import json

app = Flask(__name__)

# File paths
SENTIMENT_CSV = "output/sentiment analysis summary.csv"
TRADES_CSV = "output/trades_with_sentiment.csv"
PNG_PATH = "output/sentiment analysis_results.png"


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/summary')
def get_summary():
    """API endpoint for sentiment summary statistics"""
    try:
        df = pd.read_csv(SENTIMENT_CSV)
        # Convert to dictionary for JSON response
        summary = df.to_dict(orient='records')
        return jsonify({
            'success': True,
            'data': summary,
            'columns': df.columns.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/trades')
def get_trades():
    """API endpoint for trade data with sentiment"""
    try:
        df = pd.read_csv(TRADES_CSV)
        # Limit to first 1000 rows for performance
        if len(df) > 1000:
            df = df.head(1000)
        trades = df.to_dict(orient='records')
        return jsonify({
            'success': True,
            'data': trades,
            'total': len(df)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stats')
def get_stats():
    """API endpoint for calculated statistics"""
    try:
        trades_df = pd.read_csv(TRADES_CSV)

        stats = {
            'total_trades': len(trades_df),
            'avg_pnl': trades_df['closedPnL'].mean(),
            'total_pnl': trades_df['closedPnL'].sum(),
            'win_rate': (trades_df['is_win'].mean() * 100),
            'avg_leverage': trades_df['leverage'].mean(),
            'greed_count': len(trades_df[trades_df['Sentiment'] == 'Greed']),
            'fear_count': len(trades_df[trades_df['Sentiment'] == 'Fear']),
            'greed_win_rate': (trades_df[trades_df['Sentiment'] == 'Greed']['is_win'].mean() * 100),
            'fear_win_rate': (trades_df[trades_df['Sentiment'] == 'Fear']['is_win'].mean() * 100),
            'greed_avg_pnl': trades_df[trades_df['Sentiment'] == 'Greed']['closedPnL'].mean(),
            'fear_avg_pnl': trades_df[trades_df['Sentiment'] == 'Fear']['closedPnL'].mean()
        }

        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chart-data')
def get_chart_data():
    """API endpoint for chart visualizations"""
    try:
        trades_df = pd.read_csv(TRADES_CSV)

        # PnL by sentiment
        pnl_by_sentiment = trades_df.groupby('Sentiment')['closedPnL'].agg(['mean', 'median', 'sum']).to_dict()

        # Win rate by sentiment
        winrate_by_sentiment = trades_df.groupby('Sentiment')['is_win'].mean().to_dict()

        # Trade count by sentiment
        count_by_sentiment = trades_df['Sentiment'].value_counts().to_dict()

        return jsonify({
            'success': True,
            'pnl_by_sentiment': pnl_by_sentiment,
            'winrate_by_sentiment': winrate_by_sentiment,
            'count_by_sentiment': count_by_sentiment
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/image')
def get_image():
    """Return the visualization image"""
    try:
        return send_file(PNG_PATH, mimetype='image/png')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)