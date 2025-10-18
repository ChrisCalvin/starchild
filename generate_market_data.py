import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 500 entries
num_entries = 500
start_time = datetime(2025, 8, 15, 10, 0, 0)

data = []
current_price = 30000.0
current_volume = 10.0

for i in range(num_entries):
    timestamp = start_time + timedelta(minutes=i)
    
    # Simulate price movement
    open_price = current_price
    high_price = current_price + np.random.uniform(0, 5)
    low_price = current_price - np.random.uniform(0, 5)
    close_price = current_price + np.random.uniform(-3, 3)
    
    # Ensure prices are somewhat realistic
    if close_price < low_price:
        low_price = close_price
    if close_price > high_price:
        high_price = close_price
        
    # Simulate volume
    volume = current_volume + np.random.uniform(-2, 2)
    if volume < 1.0:
        volume = 1.0
        
    data.append({
        'timestamp': timestamp.isoformat() + 'Z',
        'open': round(open_price, 1),
        'high': round(high_price, 1),
        'low': round(low_price, 1),
        'close': round(close_price, 1),
        'volume': round(volume, 1)
    })
    
    current_price = close_price
    current_volume = volume

df = pd.DataFrame(data)
csv_content = df.to_csv(index=False)
print(csv_content)