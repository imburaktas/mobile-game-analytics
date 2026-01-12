"""
Mobile Game Analytics - Synthetic Data Generator
================================================
Generates realistic mobile game data including:
- Player demographics and device info
- Session and engagement data
- Retention metrics (Day 1, 7, 30)
- In-app purchase behavior
- A/B test groups
- Cohort information

Author: Burak (Portfolio Project)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_player_data(n_players=50000):
    """Generate base player data"""
    
    # Installation dates over 90 days
    base_date = datetime(2024, 10, 1)
    install_dates = [base_date + timedelta(days=random.randint(0, 90)) for _ in range(n_players)]
    
    # Countries with realistic distribution (mobile gaming markets)
    countries = ['USA', 'UK', 'Germany', 'Turkey', 'Brazil', 'Japan', 'South Korea', 
                 'India', 'Indonesia', 'Russia', 'France', 'Canada', 'Australia', 'Mexico']
    country_weights = [0.20, 0.08, 0.07, 0.06, 0.10, 0.08, 0.05, 
                       0.12, 0.06, 0.04, 0.04, 0.03, 0.03, 0.04]
    
    # Platforms
    platforms = np.random.choice(['iOS', 'Android'], n_players, p=[0.35, 0.65])
    
    # Age groups (typical mobile gamers)
    age_groups = ['13-17', '18-24', '25-34', '35-44', '45-54', '55+']
    age_weights = [0.10, 0.25, 0.30, 0.20, 0.10, 0.05]
    
    # A/B Test groups
    ab_groups = np.random.choice(['control', 'variant_A', 'variant_B'], n_players, p=[0.34, 0.33, 0.33])
    
    # Acquisition channels
    channels = ['Organic', 'Facebook Ads', 'Google Ads', 'TikTok', 'Influencer', 
                'Cross-Promo', 'Apple Search Ads', 'Unity Ads']
    channel_weights = [0.25, 0.20, 0.18, 0.12, 0.08, 0.07, 0.05, 0.05]
    
    players = pd.DataFrame({
        'user_id': [f'user_{i:06d}' for i in range(n_players)],
        'install_date': install_dates,
        'country': np.random.choice(countries, n_players, p=country_weights),
        'platform': platforms,
        'age_group': np.random.choice(age_groups, n_players, p=age_weights),
        'ab_group': ab_groups,
        'acquisition_channel': np.random.choice(channels, n_players, p=channel_weights),
        'device_model': None,  # Will be filled based on platform
    })
    
    # Device models based on platform
    ios_devices = ['iPhone 15 Pro', 'iPhone 15', 'iPhone 14', 'iPhone 13', 'iPhone 12', 'iPhone SE', 'iPad Pro', 'iPad Air']
    android_devices = ['Samsung Galaxy S24', 'Samsung Galaxy A54', 'Xiaomi 14', 'Google Pixel 8', 
                       'OnePlus 12', 'Oppo Find X7', 'Redmi Note 13', 'Samsung Galaxy Tab S9']
    
    players.loc[players['platform'] == 'iOS', 'device_model'] = np.random.choice(
        ios_devices, (players['platform'] == 'iOS').sum())
    players.loc[players['platform'] == 'Android', 'device_model'] = np.random.choice(
        android_devices, (players['platform'] == 'Android').sum())
    
    return players


def generate_engagement_data(players):
    """Generate engagement and session data"""
    
    n_players = len(players)
    
    # Base engagement varies by acquisition channel and A/B group
    channel_engagement = {
        'Organic': 1.2,
        'Facebook Ads': 0.9,
        'Google Ads': 0.95,
        'TikTok': 0.85,
        'Influencer': 1.1,
        'Cross-Promo': 1.15,
        'Apple Search Ads': 1.0,
        'Unity Ads': 0.8
    }
    
    ab_engagement = {
        'control': 1.0,
        'variant_A': 1.15,  # Better onboarding
        'variant_B': 0.95   # Worse onboarding
    }
    
    engagement_multiplier = players['acquisition_channel'].map(channel_engagement) * \
                           players['ab_group'].map(ab_engagement)
    
    # Total sessions in first 30 days (log-normal distribution)
    base_sessions = np.random.lognormal(mean=2.5, sigma=1.2, size=n_players)
    players['total_sessions_30d'] = np.clip(base_sessions * engagement_multiplier, 1, 200).astype(int)
    
    # Average session duration (minutes)
    base_duration = np.random.gamma(shape=3, scale=4, size=n_players)
    players['avg_session_duration_mins'] = np.clip(base_duration * engagement_multiplier, 1, 120).round(1)
    
    # Total playtime in first 30 days (hours)
    players['total_playtime_hours_30d'] = (players['total_sessions_30d'] * 
                                           players['avg_session_duration_mins'] / 60).round(2)
    
    # Levels completed (progression)
    base_levels = np.random.exponential(scale=15, size=n_players)
    players['levels_completed'] = np.clip(base_levels * engagement_multiplier, 0, 500).astype(int)
    
    # Last level reached (gate at level 30 and 60)
    players['max_level_reached'] = players['levels_completed'] + np.random.randint(0, 5, n_players)
    
    # Games played (rounds)
    players['games_played'] = (players['levels_completed'] * np.random.uniform(1.5, 3, n_players)).astype(int)
    
    # Tutorial completion
    tutorial_prob = 0.7 + 0.2 * (players['ab_group'] == 'variant_A').astype(float)
    players['tutorial_completed'] = np.random.binomial(1, tutorial_prob)
    
    # Social features
    players['friends_added'] = np.random.poisson(lam=2, size=n_players) * players['tutorial_completed']
    players['gifts_sent'] = np.random.poisson(lam=3, size=n_players) * (players['friends_added'] > 0).astype(int)
    players['gifts_received'] = np.random.poisson(lam=2.5, size=n_players) * (players['friends_added'] > 0).astype(int)
    
    return players


def generate_retention_data(players):
    """Generate retention metrics"""
    
    n_players = len(players)
    
    # Base retention rates (industry benchmarks for casual mobile games)
    # Day 1: ~40%, Day 7: ~15%, Day 30: ~5%
    
    # Factors affecting retention
    engagement_score = (players['total_sessions_30d'] / players['total_sessions_30d'].max() + 
                       players['tutorial_completed']) / 2
    
    ab_retention = {
        'control': 1.0,
        'variant_A': 1.12,  # 12% better retention
        'variant_B': 0.92   # 8% worse retention
    }
    
    channel_retention = {
        'Organic': 1.15,
        'Facebook Ads': 0.9,
        'Google Ads': 0.95,
        'TikTok': 0.85,
        'Influencer': 1.1,
        'Cross-Promo': 1.2,
        'Apple Search Ads': 1.05,
        'Unity Ads': 0.75
    }
    
    retention_mult = players['ab_group'].map(ab_retention) * \
                    players['acquisition_channel'].map(channel_retention)
    
    # Day 1 retention
    d1_base = 0.40
    d1_prob = np.clip(d1_base * retention_mult * (0.8 + 0.4 * engagement_score), 0, 0.8)
    players['retention_day1'] = np.random.binomial(1, d1_prob)
    
    # Day 7 retention (conditional on Day 1)
    d7_base = 0.35  # Of those who returned Day 1
    d7_prob = np.clip(d7_base * retention_mult * (0.7 + 0.6 * engagement_score), 0, 0.7)
    players['retention_day7'] = players['retention_day1'] * np.random.binomial(1, d7_prob, n_players)
    
    # Day 30 retention (conditional on Day 7)
    d30_base = 0.30  # Of those who returned Day 7
    d30_prob = np.clip(d30_base * retention_mult * (0.6 + 0.8 * engagement_score), 0, 0.6)
    players['retention_day30'] = players['retention_day7'] * np.random.binomial(1, d30_prob, n_players)
    
    # Days since last active
    max_inactive = 90
    players['days_since_last_active'] = np.where(
        players['retention_day30'] == 1,
        np.random.randint(0, 7, n_players),
        np.where(
            players['retention_day7'] == 1,
            np.random.randint(7, 30, n_players),
            np.where(
                players['retention_day1'] == 1,
                np.random.randint(1, 90, n_players),
                np.random.randint(0, 2, n_players)
            )
        )
    )
    
    # Churn status
    players['is_churned'] = (players['days_since_last_active'] > 14).astype(int)
    
    return players


def generate_monetization_data(players):
    """Generate in-app purchase and monetization data"""
    
    n_players = len(players)
    
    # Payer conversion rate (~2-5% for F2P games)
    engagement_factor = players['total_sessions_30d'] / players['total_sessions_30d'].max()
    retention_factor = (players['retention_day1'] + players['retention_day7'] + players['retention_day30']) / 3
    
    # Probability of being a payer
    payer_prob = np.clip(0.02 + 0.08 * engagement_factor + 0.05 * retention_factor, 0, 0.25)
    players['is_payer'] = np.random.binomial(1, payer_prob)
    
    # Number of purchases (for payers)
    players['num_purchases'] = np.where(
        players['is_payer'] == 1,
        np.random.geometric(p=0.3, size=n_players),
        0
    )
    
    # Total revenue (USD)
    # Price points: 0.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99
    price_points = [0.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99]
    price_weights = [0.30, 0.25, 0.20, 0.15, 0.06, 0.03, 0.01]
    
    def calculate_revenue(num_purchases):
        if num_purchases == 0:
            return 0
        purchases = np.random.choice(price_points, num_purchases, p=price_weights)
        return purchases.sum()
    
    players['total_revenue_usd'] = players['num_purchases'].apply(calculate_revenue)
    
    # Average revenue per purchase
    players['avg_purchase_value'] = np.where(
        players['num_purchases'] > 0,
        players['total_revenue_usd'] / players['num_purchases'],
        0
    ).round(2)
    
    # First purchase day (days after install)
    players['first_purchase_day'] = np.where(
        players['is_payer'] == 1,
        np.random.exponential(scale=5, size=n_players).astype(int),
        -1
    )
    
    # IAP categories
    iap_categories = ['Currency Pack', 'Starter Bundle', 'Battle Pass', 'Cosmetics', 
                      'Power-ups', 'Energy Refill', 'Special Offer']
    players['first_iap_category'] = np.where(
        players['is_payer'] == 1,
        np.random.choice(iap_categories, n_players),
        'None'
    )
    
    # Ad revenue (estimated from ad views)
    # Non-payers watch more ads
    ad_views = np.where(
        players['is_payer'] == 0,
        players['total_sessions_30d'] * np.random.uniform(3, 8, n_players),
        players['total_sessions_30d'] * np.random.uniform(0.5, 2, n_players)
    ).astype(int)
    players['ad_views_30d'] = ad_views
    
    # Ad revenue (estimated $0.01-0.05 per view)
    players['ad_revenue_usd'] = (ad_views * np.random.uniform(0.01, 0.05, n_players)).round(2)
    
    # Total LTV (Lifetime Value estimate)
    players['estimated_ltv_usd'] = (players['total_revenue_usd'] + players['ad_revenue_usd']).round(2)
    
    return players


def generate_event_data(players, n_events_per_player=20):
    """Generate detailed event logs for a sample of players"""
    
    # Sample 10% of players for detailed events
    sample_players = players.sample(frac=0.1, random_state=42)
    
    event_types = [
        'level_start', 'level_complete', 'level_fail', 'ad_watched', 
        'iap_view', 'iap_purchase', 'tutorial_step', 'social_share',
        'friend_invite', 'daily_reward_claimed', 'achievement_unlocked',
        'settings_changed', 'notification_clicked', 'session_start', 'session_end'
    ]
    
    events = []
    
    for _, player in sample_players.iterrows():
        install_date = player['install_date']
        n_events = np.random.randint(10, n_events_per_player * 3)
        
        for _ in range(n_events):
            event_time = install_date + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            events.append({
                'user_id': player['user_id'],
                'event_type': np.random.choice(event_types),
                'event_timestamp': event_time,
                'event_value': np.random.randint(1, 100)
            })
    
    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values(['user_id', 'event_timestamp'])
    
    return events_df


def main():
    """Generate all datasets"""
    
    print("🎮 Mobile Game Analytics - Data Generation")
    print("=" * 50)
    
    # Generate player data
    print("\n📊 Generating player data...")
    players = generate_player_data(n_players=50000)
    print(f"   Created {len(players)} players")
    
    # Add engagement data
    print("📈 Adding engagement metrics...")
    players = generate_engagement_data(players)
    
    # Add retention data
    print("🔄 Calculating retention metrics...")
    players = generate_retention_data(players)
    
    # Add monetization data
    print("💰 Generating monetization data...")
    players = generate_monetization_data(players)
    
    # Generate events
    print("📝 Creating event logs (10% sample)...")
    events = generate_event_data(players)
    print(f"   Created {len(events)} events")
    
    # Save datasets
    print("\n💾 Saving datasets...")
    players.to_csv('/home/claude/mobile-game-analytics/data/raw/players.csv', index=False)
    events.to_csv('/home/claude/mobile-game-analytics/data/raw/events.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 DATASET SUMMARY")
    print("=" * 50)
    print(f"\n👥 Total Players: {len(players):,}")
    print(f"📅 Date Range: {players['install_date'].min().date()} to {players['install_date'].max().date()}")
    print(f"\n📈 Retention Rates:")
    print(f"   Day 1:  {players['retention_day1'].mean()*100:.1f}%")
    print(f"   Day 7:  {players['retention_day7'].mean()*100:.1f}%")
    print(f"   Day 30: {players['retention_day30'].mean()*100:.1f}%")
    print(f"\n💰 Monetization:")
    print(f"   Payer Conversion: {players['is_payer'].mean()*100:.2f}%")
    print(f"   ARPPU: ${players[players['is_payer']==1]['total_revenue_usd'].mean():.2f}")
    print(f"   ARPU:  ${players['total_revenue_usd'].mean():.2f}")
    print(f"\n🧪 A/B Test Groups:")
    for group in ['control', 'variant_A', 'variant_B']:
        group_data = players[players['ab_group'] == group]
        print(f"   {group}: {len(group_data):,} players, D1 Retention: {group_data['retention_day1'].mean()*100:.1f}%")
    
    print("\n✅ Data generation complete!")
    print(f"   Files saved to: /home/claude/mobile-game-analytics/data/raw/")
    
    return players, events


if __name__ == "__main__":
    players, events = main()
