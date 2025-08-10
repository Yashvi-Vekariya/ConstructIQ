#!/usr/bin/env python3
"""
Smart Home Planner - Data Collection Script
Downloads construction cost data from multiple sources
"""

import requests
import pandas as pd
import os
from pathlib import Path
import kaggle
from urllib.parse import urlparse

class DataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_government_data(self):
        """Download Indian government construction data"""
        print("📊 Downloading Government Data...")
        
        # CPWD Data URLs
        urls = {
            "cpwd_plinth_rates": "https://www.jayaramanr.com/assets/docs/CPWD_1976_to_2021_210909_113612.pdf",
            "karnataka_pwd": "https://kpwd.karnataka.gov.in/storage/pdf-files/Vol 2 Schedule of Rates for Buildings - Final 18-03-2022.pdf"
        }
        
        for name, url in urls.items():
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    filename = f"{name}.pdf"
                    with open(self.data_dir / filename, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Downloaded: {filename}")
                else:
                    print(f"❌ Failed to download: {name}")
            except Exception as e:
                print(f"❌ Error downloading {name}: {e}")
    
    def download_kaggle_datasets(self):
        """Download Kaggle datasets (requires kaggle API setup)"""
        print("🏗️ Downloading Kaggle Construction Datasets...")
        
        datasets = [
            "sasakitetsuya/construction-estimation-data",
            "teejgomez/construction-data", 
            "mohamedafsal007/house-price-dataset-of-india",
            "devitachi/building-dataset"
        ]
        
        try:
            # Check if kaggle API is configured
            kaggle.api.authenticate()
            
            for dataset in datasets:
                try:
                    dataset_name = dataset.split('/')[-1]
                    download_path = self.data_dir / dataset_name
                    download_path.mkdir(exist_ok=True)
                    
                    kaggle.api.dataset_download_files(
                        dataset, 
                        path=str(download_path), 
                        unzip=True
                    )
                    print(f"✅ Downloaded: {dataset_name}")
                except Exception as e:
                    print(f"❌ Error downloading {dataset}: {e}")
                    
        except OSError:
            print("❌ Kaggle API not configured. Please run:")
            print("   1. pip install kaggle")
            print("   2. Get API token from kaggle.com/account")
            print("   3. Place kaggle.json in ~/.kaggle/")
    
    def download_sample_datasets(self):
        """Download publicly available sample datasets"""
        print("📈 Downloading Sample Datasets...")
        
        # Create sample construction cost data
        sample_data = {
            'area_sqft': [800, 1200, 1500, 2000, 2500],
            'floors': [1, 1, 2, 2, 3],
            'location': ['Rural', 'Semi-Urban', 'Urban', 'Metro', 'Metro'],
            'quality_tier': ['Basic', 'Standard', 'Standard', 'Premium', 'Luxury'],
            'cement_cost': [32000, 54000, 67500, 90000, 112500],
            'steel_cost': [28000, 48000, 60000, 80000, 100000],
            'labor_cost': [80000, 120000, 150000, 200000, 250000],
            'total_cost': [140000, 222000, 277500, 370000, 462500]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(self.data_dir / 'sample_construction_costs.csv', index=False)
        print("✅ Created: sample_construction_costs.csv")
        
        # Create sample material prices
        material_prices = {
            'material': ['Cement (50kg bag)', 'Steel (per kg)', 'Bricks (per 1000)', 
                        'Sand (per cubic ft)', 'Paint (per liter)'],
            'basic_price': [350, 65, 8000, 1200, 180],
            'premium_price': [420, 75, 12000, 1500, 280],
            'location_factor': [1.0, 1.15, 1.25, 1.4, 1.6]  # Rural to Metro multiplier
        }
        
        materials_df = pd.DataFrame(material_prices)
        materials_df.to_csv(self.data_dir / 'material_prices.csv', index=False)
        print("✅ Created: material_prices.csv")

def main():
    """Main function to run data collection"""
    print("🏗️ Smart Home Planner - Data Collection Started")
    print("=" * 50)
    
    collector = DataCollector()
    
    # Download different types of data
    collector.download_sample_datasets()
    collector.download_government_data()  
    collector.download_kaggle_datasets()
    
    print("=" * 50)
    print("✅ Data collection completed!")
    print(f"📁 Check the 'data' folder for downloaded files")
    
    # List downloaded files
    data_files = list(Path("data").glob("*"))
    if data_files:
        print("\n📋 Downloaded files:")
        for file in data_files:
            print(f"   • {file.name}")

if __name__ == "__main__":
    main()