import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_cache = {}
        
        self.components = ["gearbox", "leftaxlebox", "motor", "rightaxlebox"]
        
        self.sample_to_condition_map = {
            # 训练集
            ("Training", "Sample1"): {"frequency": 20, "load": -1},  # 20Hz-1kN
            ("Training", "Sample2"): {"frequency": 40, "load": -1},  # 40Hz-1kN
            ("Training", "Sample3"): {"frequency": 60, "load": -1},  # 60Hz-1kN
            # 测试集
            ("Test", "Sample1"): {"frequency": 20, "load": 0},  # 20Hz 0kN
            ("Test", "Sample2"): {"frequency": 20, "load": 1},  # 20Hz 1kN
            ("Test", "Sample3"): {"frequency": 40, "load": 0},  # 40Hz 0kN
            ("Test", "Sample4"): {"frequency": 40, "load": 1},  # 40Hz 1kN
            ("Test", "Sample5"): {"frequency": 60, "load": 0},  # 60Hz 0kN
            ("Test", "Sample6"): {"frequency": 60, "load": 1},  # 60Hz 1kN
        }

    def get_fault_types(self, dataset="Training"):
        path = self.base_dir / dataset
        if not path.exists():
            print(f"警告: 目录 {path} 不存在")
            return []
        return [d.name for d in path.iterdir() if d.is_dir()]

    def get_samples(self, fault_type, dataset="Training"):
        path = self.base_dir / dataset / fault_type
        if not path.exists():
            print(f"警告: 路径 {path} 不存在")
            return []
        return [d.name for d in path.iterdir() if d.is_dir()]

    def load_component_data(self, fault_type, sample_name, component, dataset="Training"):
        cache_key = f"{dataset}_{fault_type}_{sample_name}_{component}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        file_path = self.base_dir / dataset / fault_type / sample_name / f"data_{component}.csv"
        
        if not file_path.exists():
            print(f"警告: 文件 {file_path} 不存在")
            return None
        
        data = pd.read_csv(file_path)
        self.data_cache[cache_key] = data
        return data

    def get_processed_dataset(self, include_types=None, dataset="Training"):
        fault_types = include_types or self.get_fault_types(dataset)
        
        component_data = {comp: [] for comp in self.components}
        labels = []
        conditions = []
        
        for fault_type in fault_types:
            label = int(fault_type.replace('TYPE', ''))
            
            for sample_name in self.get_samples(fault_type, dataset):
                all_components_valid = True
                sample_components = {}
                
                for component in self.components:
                    comp_data = self.load_component_data(fault_type, sample_name, component, dataset)
                    
                    if comp_data is None:
                        all_components_valid = False
                        print(f"跳过样本 {dataset}/{fault_type}/{sample_name}: 缺少 {component} 数据")
                        break
                    
                    numeric_data = comp_data.select_dtypes(include=[np.number])
                    sample_components[component] = numeric_data.values
                
                if not all_components_valid:
                    continue
                
                labels.append(label)
                
                condition = self.sample_to_condition_map.get((dataset, sample_name), {"frequency": 0, "load": 0})
                conditions.append([condition["frequency"], condition["load"]])
                
                for component in self.components:
                    component_data[component].append(sample_components[component])
        
        if not labels:
            print("警告: 没有找到符合条件的数据")
            return None, None, None
        
        labels_array = np.array(labels)
        conditions_array = np.array(conditions)
        
        component_arrays = {}
        for component, samples_list in component_data.items():
            component_arrays[component] = np.array(samples_list)
        
        return component_arrays, labels_array, conditions_array