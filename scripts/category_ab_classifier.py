#!/usr/bin/env python3
"""
Category A/B Answer Generation Module

Contains GBDT classifier for Category A (600Mbps) and rule-based for Category B (100Mbps).
Used by step_2_data_enrichment.py to generate answers on-the-fly.
"""

import re
import warnings
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# Semantic patterns for answer remapping (C1-C8 to option labels)
SEMANTIC_PATTERNS = [
    (r'downtilt\s+angle.*too\s+large', 'C1'),
    (r'weak\s+coverage\s+at\s+the\s+far\s+end', 'C1'),
    (r'coverage\s+distance\s+exceeds?\s+1\s*km', 'C2'),
    (r'over-?shooting', 'C2'),
    (r'neighbor.*cell.*provides?\s+higher\s+throughput', 'C3'),
    (r'neighboring\s+cell\s+provides?\s+higher', 'C3'),
    (r'non-?colocated.*co-?frequency.*neighboring', 'C4'),
    (r'overlapping\s+coverage', 'C4'),
    (r'severe\s+overlap', 'C4'),
    (r'frequent\s+handovers?', 'C5'),
    (r'handovers?\s+degrade', 'C5'),
    (r'same\s+pci\s+mod\s+30', 'C6'),
    (r'pci\s+mod\s+30', 'C6'),
    (r'speed\s+exceeds?\s+40\s*km', 'C7'),
    (r'vehicle\s+speed\s+exceeds?\s+40', 'C7'),
    (r'scheduled\s+rbs?\s+.*below\s+160', 'C8'),
    (r'rbs?\s+(are\s+)?below\s+160', 'C8'),
]


def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points in meters."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371000 * 2 * asin(sqrt(a))


def get_beam_val(scenario):
    if scenario == 'DEFAULT':
        return 0
    m = re.search(r'SCENARIO_(\d+)', scenario)
    return int(m.group(1)) if m else 0


def map_option_to_c(option_text):
    """Map option text to C-category using semantic patterns."""
    text_lower = option_text.lower()
    for pattern, c_val in SEMANTIC_PATTERNS:
        if re.search(pattern, text_lower):
            return c_val
    return None


def extract_option_mapping(question):
    """Extract option label → C-category mapping from question."""
    mapping = {}
    option_pattern = re.compile(r'^([A-Z]?\d+):\s*(.+)$')
    for line in question.split('\n'):
        match = option_pattern.match(line.strip())
        if match:
            c_val = map_option_to_c(match.group(2))
            if c_val:
                mapping[match.group(1)] = c_val
    return mapping


def is_5g_600mbps(question):
    """Check if question is 5G 600Mbps type."""
    return '600Mbps' in question or '600 Mbps' in question


def is_5g_100mbps(question):
    """Check if question is 5G 100Mbps type."""
    return '100Mbps' in question or '100 Mbps' in question


def parse_question_features(question: str) -> dict:
    """Extract ML features from question data."""
    features = {}
    lines = question.split('\n')
    
    speeds, rsrps, sinrs, throughputs, rbs = [], [], [], [], []
    serving_pcis, lons, lats = [], [], []
    neighbor_data = []
    
    data_started = False
    for line in lines:
        if 'Timestamp|' in line:
            data_started = True
            continue
        if not data_started:
            continue
        if 'gNodeB ID|' in line or 'Engeneering' in line:
            break
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 19:
                try:
                    lons.append(float(parts[1]) if parts[1] != '-' else 0)
                    lats.append(float(parts[2]) if parts[2] != '-' else 0)
                    speeds.append(float(parts[3]) if parts[3] != '-' else 0)
                    serving_pcis.append(int(parts[4]) if parts[4] != '-' else 0)
                    rsrps.append(float(parts[5]) if parts[5] != '-' else -100)
                    sinrs.append(float(parts[6]) if parts[6] != '-' else 0)
                    throughputs.append(float(parts[7]) if parts[7] != '-' else 0)
                    rbs.append(float(parts[18]) if len(parts) > 18 and parts[18] != '-' else 0)
                    
                    row_neighbors = []
                    for i in range(5):
                        pci_idx, rsrp_idx = 8 + i, 13 + i
                        if pci_idx < len(parts) and rsrp_idx < len(parts):
                            pci_str, rsrp_str = parts[pci_idx].strip(), parts[rsrp_idx].strip()
                            if pci_str and pci_str != '-' and rsrp_str and rsrp_str != '-':
                                try:
                                    row_neighbors.append({'pci': int(pci_str), 'rsrp': float(rsrp_str)})
                                except:
                                    pass
                    neighbor_data.append(row_neighbors)
                except:
                    continue
    
    # Parse engineering parameters
    cells = []
    eng_started = False
    for line in lines:
        if 'gNodeB ID|' in line:
            eng_started = True
            continue
        if eng_started and '|' in line:
            parts = line.split('|')
            if len(parts) >= 11:
                try:
                    digi_tilt = float(parts[6]) if parts[6] and parts[6] != '-' else 0
                    if digi_tilt == 255:
                        digi_tilt = 6
                    cells.append({
                        'gnodeb': parts[0].strip(),
                        'lon': float(parts[2]) if parts[2] else 0,
                        'lat': float(parts[3]) if parts[3] else 0,
                        'mech_dt': float(parts[5]) if parts[5] else 0,
                        'digi_dt': digi_tilt,
                        'beam': parts[8] if len(parts) > 8 else 'DEFAULT',
                        'height': float(parts[9]) if len(parts) > 9 and parts[9] else 0,
                        'pci': int(parts[10]) if len(parts) > 10 and parts[10] else 0,
                    })
                except:
                    continue
    
    for c in cells:
        c['total_dt'] = c['mech_dt'] + c['digi_dt']
        c['beam_val'] = get_beam_val(c['beam'])
    
    pci_to_cell = {c['pci']: c for c in cells}
    
    # Speed features (C7)
    if speeds:
        features['speed_mean'] = np.mean(speeds)
        features['speed_max'] = np.max(speeds)
        features['speed_over_40_ratio'] = sum(1 for s in speeds if s > 40) / len(speeds)
    else:
        features['speed_mean'] = features['speed_max'] = features['speed_over_40_ratio'] = 0
    
    # RB features (C8)
    if rbs:
        features['rb_mean'] = np.mean(rbs)
        features['rb_below_160_ratio'] = sum(1 for r in rbs if r < 160) / len(rbs)
    else:
        features['rb_mean'] = 200
        features['rb_below_160_ratio'] = 0
    
    # Handover features (C5)
    ho_count = sum(1 for i in range(1, len(serving_pcis)) if serving_pcis[i] != serving_pcis[i-1])
    features['handover_count'] = ho_count
    features['handover_rate'] = ho_count / max(len(serving_pcis), 1)
    
    # PCI mod 30 features (C6)
    mod30_collision = 0
    for i, pci in enumerate(serving_pcis):
        serving_mod = pci % 30
        if i < len(neighbor_data):
            for n in neighbor_data[i]:
                if n['pci'] % 30 == serving_mod:
                    mod30_collision += 1
                    break
    features['mod30_collision_ratio'] = mod30_collision / max(len(serving_pcis), 1)
    
    # RSRP/SINR features
    if rsrps:
        features['rsrp_mean'] = np.mean(rsrps)
        features['rsrp_weak_ratio'] = sum(1 for r in rsrps if r < -90) / len(rsrps)
    else:
        features['rsrp_mean'] = -85
        features['rsrp_weak_ratio'] = 0
    
    if sinrs:
        features['sinr_mean'] = np.mean(sinrs)
    else:
        features['sinr_mean'] = 10
    
    # Throughput features
    if throughputs:
        features['tp_mean'] = np.mean(throughputs)
        features['tp_below_100_ratio'] = sum(1 for t in throughputs if t < 100) / len(throughputs)
    else:
        features['tp_mean'] = features['tp_below_100_ratio'] = 0
    
    # Distance features (C2)
    distances = []
    for i, pci in enumerate(serving_pcis):
        if pci in pci_to_cell and i < len(lons) and lons[i] > 0:
            cell = pci_to_cell[pci]
            distances.append(haversine(lons[i], lats[i], cell['lon'], cell['lat']))
    
    if distances:
        features['dist_mean'] = np.mean(distances)
        features['dist_over_1km_ratio'] = sum(1 for d in distances if d > 1000) / len(distances)
    else:
        features['dist_mean'] = features['dist_over_1km_ratio'] = 0
    
    # Downtilt features (C1)
    if cells:
        serving_dts = [pci_to_cell[pci]['total_dt'] for pci in set(serving_pcis) if pci in pci_to_cell]
        features['serving_dt_max'] = np.max(serving_dts) if serving_dts else 0
    else:
        features['serving_dt_max'] = 0
    
    # Neighbor features (C3)
    neighbor_stronger = 0
    for i, rsrp in enumerate(rsrps):
        if i < len(neighbor_data) and neighbor_data[i]:
            best_neighbor = max(n['rsrp'] for n in neighbor_data[i])
            if best_neighbor > rsrp:
                neighbor_stronger += 1
    features['neighbor_stronger_ratio'] = neighbor_stronger / max(len(rsrps), 1)
    
    return features


class CategoryAClassifier:
    """GBDT Classifier for Category A (600Mbps) questions."""
    
    def __init__(self):
        self.clf = None
        self.le = None
        self.feature_cols = None
    
    def train(self, train_df: pd.DataFrame, phase1_df: pd.DataFrame = None):
        """Train on train.csv and optionally phase1 data."""
        # Filter 600Mbps questions from train
        train_600 = train_df[train_df['question'].apply(is_5g_600mbps)].reset_index(drop=True)
        
        combined = train_600.copy()
        if phase1_df is not None and len(phase1_df) > 0:
            phase1_600 = phase1_df[phase1_df['question'].apply(is_5g_600mbps)].reset_index(drop=True)
            combined = pd.concat([train_600, phase1_600], ignore_index=True)
        
        if len(combined) == 0:
            return False
        
        # Extract features
        train_feats = [parse_question_features(q) for q in combined['question']]
        X_train = pd.DataFrame(train_feats).fillna(0)
        self.feature_cols = sorted(X_train.columns)
        
        self.le = LabelEncoder()
        y_train = self.le.fit_transform(combined['answer'].values)
        
        # Train classifier
        self.clf = HistGradientBoostingClassifier(
            max_iter=500, max_depth=12, learning_rate=0.05, random_state=42
        )
        self.clf.fit(X_train[self.feature_cols], y_train)
        return True
    
    def predict(self, question: str) -> str:
        """Predict answer for a Category A question."""
        if self.clf is None:
            return '1'  # Fallback
        
        f = parse_question_features(question)
        X = pd.DataFrame([f]).fillna(0)
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[self.feature_cols]
        
        pred_enc = self.clf.predict(X)[0]
        pred_c = self.le.inverse_transform([pred_enc])[0]
        
        # Map C-category back to option label
        mapping = extract_option_mapping(question)
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping.get(pred_c, pred_c)


def generate_category_b_answer(question: str) -> str:
    """Category B (100Mbps): Rule-based, always return 'A'."""
    return 'A'
