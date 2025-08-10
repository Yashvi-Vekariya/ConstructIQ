#!/usr/bin/env python3
"""
Smart Home Planner - Comprehensive Data Cleaning Pipeline
Cleans and standardizes raw construction data for ML models
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir  # No "raw" folder, use data/ directly
        self.cleaned_dir = self.data_dir / "cleaned"
        self.cleaned_dir.mkdir(exist_ok=True)
        
        self.cleaning_stats = {
            "files_processed": 0,
            "total_rows_before": 0,
            "total_rows_after": 0,
            "columns_standardized": 0,
            "missing_values_handled": 0,
            "outliers_removed": 0
        }
        
        print("🧹 Data Cleaning Pipeline Initialized")
        print(f"📁 Raw data: {self.raw_dir}")
        print(f"📁 Cleaned data: {self.cleaned_dir}")

    def clean_sample_data(self):
        """Clean the sample construction cost data"""
        print("\n🏗️ CLEANING SAMPLE DATA")
        print("=" * 40)
        
        sample_file = self.raw_dir / "sample_construction_costs.csv"  # Correct path
        if not sample_file.exists():
            print(f"❌ Sample data not found at {sample_file}")
            return None
        
        df = pd.read_csv(sample_file)
        ...
    
    def clean_kaggle_data(self):
        """Clean all datasets (Kaggle-style folders)"""
        print("\n🏗️ CLEANING DATASETS")
        print("=" * 40)
        
        cleaned_datasets = []
        
        for dataset_folder in self.raw_dir.iterdir():
            if dataset_folder.is_dir() and dataset_folder.name != "cleaned":
                print(f"\n📂 Processing {dataset_folder.name}")
                
                csv_files = list(dataset_folder.glob("*.csv"))
                for file in csv_files:
                    try:
                        df = pd.read_csv(file)
                        print(f"  📊 {file.name}: {len(df)} rows, {len(df.columns)} columns")
                        
                        # Dataset-specific cleaning
                        if "construction-estimation" in dataset_folder.name:
                            df_clean = self.clean_construction_estimation(df, file.name)
                        elif "house-price" in dataset_folder.name:
                            df_clean = self.clean_house_price_data(df, file.name)
                        elif "building-dataset" in dataset_folder.name:
                            df_clean = self.clean_building_data(df, file.name)
                        else:
                            df_clean = self.clean_generic_construction_data(df, file.name)
                        
                        if df_clean is not None and len(df_clean) > 0:
                            output_name = f"{dataset_folder.name}_{file.stem}_cleaned.csv"
                            df_clean.to_csv(self.cleaned_dir / output_name, index=False)
                            
                            cleaned_datasets.append({
                                "original_file": f"{dataset_folder.name}/{file.name}",
                                "cleaned_file": output_name,
                                "original_rows": len(df),
                                "cleaned_rows": len(df_clean),
                                "columns": list(df_clean.columns)
                            })
                            
                            self.cleaning_stats["files_processed"] += 1
                            self.cleaning_stats["total_rows_before"] += len(df)
                            self.cleaning_stats["total_rows_after"] += len(df_clean)
                            
                            print(f"  ✅ Cleaned: {len(df)} → {len(df_clean)} rows")
                    
                    except Exception as e:
                        print(f"  ❌ Error cleaning {file.name}: {e}")
        """Clean all Kaggle datasets"""
        print("\n🏗️ CLEANING KAGGLE DATA")
        print("=" * 40)
        
        kaggle_dir = self.raw_dir / "kaggle"
        if not kaggle_dir.exists():
            print("❌ Kaggle data not found")
            return []
        
        cleaned_datasets = []
        
        for dataset_folder in kaggle_dir.iterdir():
            if dataset_folder.is_dir():
                print(f"\n📂 Processing {dataset_folder.name}")
                
                csv_files = list(dataset_folder.glob("*.csv"))
                for file in csv_files:
                    try:
                        df = pd.read_csv(file)
                        print(f"  📊 {file.name}: {len(df)} rows, {len(df.columns)} columns")
                        
                        # Clean dataset based on its type
                        if "construction-estimation" in dataset_folder.name:
                            df_clean = self.clean_construction_estimation(df, file.name)
                        elif "house-price" in dataset_folder.name:
                            df_clean = self.clean_house_price_data(df, file.name)
                        elif "building-dataset" in dataset_folder.name:
                            df_clean = self.clean_building_data(df, file.name)
                        else:
                            df_clean = self.clean_generic_construction_data(df, file.name)
                        
                        if df_clean is not None and len(df_clean) > 0:
                            # Save cleaned dataset
                            output_name = f"{dataset_folder.name}_{file.stem}_cleaned.csv"
                            df_clean.to_csv(self.cleaned_dir / output_name, index=False)
                            
                            cleaned_datasets.append({
                                "original_file": f"{dataset_folder.name}/{file.name}",
                                "cleaned_file": output_name,
                                "original_rows": len(df),
                                "cleaned_rows": len(df_clean),
                                "columns": list(df_clean.columns)
                            })
                            
                            self.cleaning_stats["files_processed"] += 1
                            self.cleaning_stats["total_rows_before"] += len(df)
                            self.cleaning_stats["total_rows_after"] += len(df_clean)
                            
                            print(f"  ✅ Cleaned: {len(df)} → {len(df_clean)} rows")
                        
                    except Exception as e:
                        print(f"  ❌ Error cleaning {file.name}: {e}")
        
        return cleaned_datasets
    
    def clean_construction_estimation(self, df, filename):
        """Clean construction estimation specific data"""
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                # Fill with median for numeric columns
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                self.cleaning_stats["missing_values_handled"] += df_clean[col].isnull().sum()
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            if len(df_clean[col].unique()) > 10:  # Only for continuous variables
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = len(df_clean)
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                outliers_removed = outliers_before - len(df_clean)
                self.cleaning_stats["outliers_removed"] += outliers_removed
        
        # Add data source identifier
        df_clean['data_source'] = 'construction_estimation'
        
        return df_clean
    
    def clean_house_price_data(self, df, filename):
        """Clean house price dataset"""
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Common column mappings for house price data
        column_mapping = {
            'price': 'total_cost',
            'area': 'area_sqft',
            'size': 'area_sqft',
            'city': 'location',
            'bhk': 'bedrooms',
            'bathroom': 'bathrooms',
            'balcony': 'balconies'
        }
        
        # Rename columns
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Clean price/cost data
        if 'total_cost' in df_clean.columns:
            # Remove non-numeric characters and convert to numeric
            df_clean['total_cost'] = pd.to_numeric(
                df_clean['total_cost'].astype(str).str.replace(r'[^\d.]', '', regex=True), 
                errors='coerce'
            )
            
            # Remove unrealistic prices (too low or too high)
            df_clean = df_clean[
                (df_clean['total_cost'] > 50000) & 
                (df_clean['total_cost'] < 50000000)  # Between 50K and 5 Crores
            ]
        
        # Clean area data
        if 'area_sqft' in df_clean.columns:
            df_clean['area_sqft'] = pd.to_numeric(
                df_clean['area_sqft'].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )
            
            # Remove unrealistic areas
            df_clean = df_clean[
                (df_clean['area_sqft'] > 200) & 
                (df_clean['area_sqft'] < 10000)  # Between 200 and 10,000 sqft
            ]
        
        # Calculate cost per sqft if both columns exist
        if 'total_cost' in df_clean.columns and 'area_sqft' in df_clean.columns:
            df_clean['cost_per_sqft'] = df_clean['total_cost'] / df_clean['area_sqft']
            
            # Remove unrealistic cost per sqft
            df_clean = df_clean[
                (df_clean['cost_per_sqft'] > 500) & 
                (df_clean['cost_per_sqft'] < 15000)  # Between ₹500 and ₹15,000 per sqft
            ]
        
        # Clean location data
        if 'location' in df_clean.columns:
            # Standardize location names
            df_clean['location'] = df_clean['location'].str.title().str.strip()
            
            # Create location tiers
            metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune']
            tier1_cities = ['Ahmedabad', 'Surat', 'Vadodara', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur']
            
            def categorize_location(location):
                if any(city in str(location) for city in metro_cities):
                    return 'Metro'
                elif any(city in str(location) for city in tier1_cities):
                    return 'Tier1'
                else:
                    return 'Tier2'
            
            df_clean['location_tier'] = df_clean['location'].apply(categorize_location)
            
            # Encode location tiers
            tier_mapping = {'Tier2': 1, 'Tier1': 2, 'Metro': 3}
            df_clean['location_encoded'] = df_clean['location_tier'].map(tier_mapping)
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Remove rows with too many missing values
        df_clean = df_clean.dropna(thresh=len(df_clean.columns) * 0.7)
        
        # Add data source
        df_clean['data_source'] = 'house_price'
        
        return df_clean
    
    def clean_building_data(self, df, filename):
        """Clean building/construction dataset"""
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Handle numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Remove negative values where they don't make sense
            if col in ['area', 'size', 'price', 'cost', 'rooms', 'floors']:
                df_clean = df_clean[df_clean[col] >= 0]
            
            # Fill missing values with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Handle categorical columns
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            # Clean text data
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace('Nan', 'Unknown')
            df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Add data source
        df_clean['data_source'] = 'building_data'
        
        return df_clean
    
    def clean_generic_construction_data(self, df, filename):
        """Clean generic construction data"""
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Basic cleaning
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Remove rows with too many missing values
        df_clean = df_clean.dropna(thresh=len(df_clean.columns) * 0.5)
        
        # Add data source
        df_clean['data_source'] = 'generic_construction'
        
        return df_clean
    
    def standardize_all_datasets(self):
        """Standardize all cleaned datasets to common schema"""
        print("\n🔗 STANDARDIZING ALL DATASETS")
        print("=" * 40)
        
        cleaned_files = list(self.cleaned_dir.glob("*_cleaned.csv"))
        if not cleaned_files:
            print("❌ No cleaned files found")
            return None
        
        standardized_datasets = []
        
        # Define common schema
        common_schema = {
            'project_id': 'string',
            'area_sqft': 'float',
            'floors': 'int',
            'bedrooms': 'int',
            'bathrooms': 'int',
            'location': 'string',
            'location_tier': 'string',
            'location_encoded': 'int',
            'quality_tier': 'string',
            'quality_encoded': 'int',
            'construction_type': 'string',
            'total_cost': 'float',
            'cost_per_sqft': 'float',
            'cement_cost': 'float',
            'steel_cost': 'float',
            'labor_cost': 'float',
            'data_source': 'string'
        }
        
        for file in cleaned_files:
            try:
                df = pd.read_csv(file)
                print(f"📊 Standardizing {file.name}: {len(df)} rows")
                
                df_std = pd.DataFrame()
                
                # Map existing columns to standard schema
                column_mappings = self.get_column_mappings(df.columns)
                
                for std_col, dtype in common_schema.items():
                    if std_col in column_mappings:
                        source_col = column_mappings[std_col]
                        if source_col in df.columns:
                            df_std[std_col] = df[source_col]
                        else:
                            df_std[std_col] = self.get_default_value(std_col, dtype, len(df))
                    else:
                        df_std[std_col] = self.get_default_value(std_col, dtype, len(df))
                
                # Generate project IDs if not present
                if 'project_id' not in df.columns:
                    df_std['project_id'] = [f"PROJ_{file.stem}_{i:04d}" for i in range(len(df_std))]
                
                # Calculate missing derived columns
                if 'cost_per_sqft' not in df_std.columns or df_std['cost_per_sqft'].isna().all():
                    if 'total_cost' in df_std.columns and 'area_sqft' in df_std.columns:
                        df_std['cost_per_sqft'] = df_std['total_cost'] / df_std['area_sqft']
                
                # Remove invalid rows
                df_std = df_std.dropna(subset=['area_sqft', 'total_cost'])
                df_std = df_std[df_std['area_sqft'] > 0]
                df_std = df_std[df_std['total_cost'] > 0]
                
                if len(df_std) > 0:
                    standardized_datasets.append(df_std)
                    print(f"  ✅ Standardized: {len(df)} → {len(df_std)} rows")
                else:
                    print(f"  ❌ No valid data after standardization")
                
            except Exception as e:
                print(f"  ❌ Error standardizing {file.name}: {e}")
        
        if standardized_datasets:
            # Combine all standardized datasets
            master_df = pd.concat(standardized_datasets, ignore_index=True)
            
            # Final cleaning
            master_df = self.final_cleanup(master_df)
            
            # Save master dataset
            master_df.to_csv(self.cleaned_dir / "master_construction_dataset.csv", index=False)
            
            print(f"\n✅ Master dataset created: {len(master_df)} rows, {len(master_df.columns)} columns")
            print(f"📊 Data sources: {master_df['data_source'].value_counts().to_dict()}")
            
            return master_df
        
        return None
    
    def get_column_mappings(self, columns):
        """Get mapping from source columns to standard columns"""
        mappings = {}
        
        column_patterns = {
            'area_sqft': ['area', 'size', 'sqft', 'square_feet', 'built_area', 'super_area'],
            'total_cost': ['price', 'cost', 'amount', 'total_price', 'property_price'],
            'floors': ['floor', 'floors', 'stories', 'levels'],
            'bedrooms': ['bedroom', 'bedrooms', 'bhk', 'rooms'],
            'bathrooms': ['bathroom', 'bathrooms', 'baths'],
            'location': ['city', 'location', 'area', 'locality', 'region'],
            'quality_tier': ['quality', 'grade', 'class', 'category'],
            'construction_type': ['type', 'property_type', 'building_type']
        }
        
        for std_col, patterns in column_patterns.items():
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    mappings[std_col] = col
                    break
        
        return mappings
    
    def get_default_value(self, column, dtype, length):
        """Get default value for missing columns"""
        defaults = {
            'project_id': [f"PROJ_DEFAULT_{i:04d}" for i in range(length)],
            'floors': 1,
            'bedrooms': 2,
            'bathrooms': 1,
            'location': 'Unknown',
            'location_tier': 'Tier2',
            'location_encoded': 1,
            'quality_tier': 'Standard',
            'quality_encoded': 2,
            'construction_type': 'Residential'
        }
        
        if column in defaults:
            default = defaults[column]
            if isinstance(default, list):
                return default
            else:
                return [default] * length
        
        # Generate default based on dtype
        if 'int' in dtype:
            return [0] * length
        elif 'float' in dtype:
            return [0.0] * length
        else:
            return ['Unknown'] * length
    
    def final_cleanup(self, df):
        """Final cleanup of master dataset"""
        print("\n🧹 Final cleanup of master dataset...")
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"  Removed duplicates: {initial_rows - len(df)} rows")
        
        # Remove unrealistic data
        df = df[
            (df['area_sqft'] > 100) & (df['area_sqft'] < 20000) &
            (df['total_cost'] > 10000) & (df['total_cost'] < 100000000)
        ]
        print(f"  Removed unrealistic values: {len(df)} rows remaining")
        
        # Recalculate cost_per_sqft
        df['cost_per_sqft'] = df['total_cost'] / df['area_sqft']
        
        # Remove unrealistic cost per sqft
        df = df[
            (df['cost_per_sqft'] > 200) & (df['cost_per_sqft'] < 20000)
        ]
        print(f"  Final dataset: {len(df)} rows")
        
        return df
    
    def create_data_splits(self, df):
        """Create train/validation/test splits"""
        print("\n📊 Creating data splits...")
        
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create splits
        n = len(df_shuffled)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]
        
        # Save splits
        train_df.to_csv(self.cleaned_dir / "train_dataset.csv", index=False)
        val_df.to_csv(self.cleaned_dir / "validation_dataset.csv", index=False)
        test_df.to_csv(self.cleaned_dir / "test_dataset.csv", index=False)
        
        print(f"✅ Train: {len(train_df)} rows")
        print(f"✅ Validation: {len(val_df)} rows") 
        print(f"✅ Test: {len(test_df)} rows")
        
        # Save data dictionary
        self.create_data_dictionary(df)
    
    def create_data_dictionary(self, df):
        """Create comprehensive data dictionary"""
        data_dict = {
            "dataset_info": {
                "name": "Smart Home Planner - Clean Construction Dataset",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "total_records": len(df),
                "total_features": len(df.columns),
                "cleaning_stats": self.cleaning_stats
            },
            "features": {}
        }
        
        for col in df.columns:
            col_info = {
                "description": self.get_column_description(col),
                "data_type": str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "missing_percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
                "unique_values": int(df[col].nunique())
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "quartiles": {
                        "25%": float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
                        "50%": float(df[col].quantile(0.50)) if not df[col].isnull().all() else None,
                        "75%": float(df[col].quantile(0.75)) if not df[col].isnull().all() else None
                    }
                })
            else:
                col_info.update({
                    "value_counts": df[col].value_counts().head(10).to_dict(),
                    "sample_values": df[col].dropna().head(5).tolist()
                })
            
            data_dict["features"][col] = col_info
        
        # Save data dictionary
        with open(self.cleaned_dir / "data_dictionary.json", "w") as f:
            json.dump(data_dict, f, indent=2)
        
        print("✅ Data dictionary created")
    
    def get_column_description(self, column):
        """Get description for each column"""
        descriptions = {
            'project_id': 'Unique identifier for each construction project',
            'area_sqft': 'Built-up area in square feet',
            'floors': 'Number of floors in the building',
            'bedrooms': 'Number of bedrooms',
            'bathrooms': 'Number of bathrooms',
            'location': 'City or area location',
            'location_tier': 'Location categorization (Metro/Tier1/Tier2)',
            'location_encoded': 'Numeric encoding of location tier',
            'quality_tier': 'Construction quality level',
            'quality_encoded': 'Numeric encoding of quality tier',
            'construction_type': 'Type of construction (Residential/Commercial)',
            'total_cost': 'Total construction cost in INR',
            'cost_per_sqft': 'Cost per square foot in INR',
            'cement_cost': 'Cost of cement materials in INR',
            'steel_cost': 'Cost of steel materials in INR',
            'labor_cost': 'Labor cost in INR',
            'data_source': 'Source of the data record'
        }
        
        return descriptions.get(column, f"Feature: {column}")

def main():
    """Main data cleaning pipeline"""
    print("🧹 Smart Home Planner - Data Cleaning Pipeline")
    print("=" * 50)
    
    cleaner = DataCleaner()
    
    # Step 1: Clean sample data
    sample_df = cleaner.clean_sample_data()
    
    # Step 2: Clean Kaggle datasets
    kaggle_datasets = cleaner.clean_kaggle_data()
    
    # Step 3: Standardize all datasets
    master_df = cleaner.standardize_all_datasets()
    
    if master_df is not None:
        # Step 4: Create data splits
        cleaner.create_data_splits(master_df)
        
        print("\n" + "=" * 50)
        print("✅ DATA CLEANING COMPLETED!")
        print(f"\n📊 Cleaning Summary:")
        print(f"   • Files processed: {cleaner.cleaning_stats['files_processed']}")
        print(f"   • Rows before: {cleaner.cleaning_stats['total_rows_before']:,}")
        print(f"   • Rows after: {cleaner.cleaning_stats['total_rows_after']:,}")
        print(f"   • Data retention: {(cleaner.cleaning_stats['total_rows_after']/max(cleaner.cleaning_stats['total_rows_before'],1)*100):.1f}%")
        
        print(f"\n📁 Check 'data/cleaned' folder for:")
        print("   • master_construction_dataset.csv")
        print("   • train_dataset.csv")
        print("   • validation_dataset.csv")
        print("   • test_dataset.csv")
        print("   • data_dictionary.json")
        
        print("\n🚀 Ready for ML model training!")
    else:
        print("❌ No data available after cleaning")

if __name__ == "__main__":
    main()