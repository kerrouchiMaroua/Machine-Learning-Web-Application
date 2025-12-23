import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report)
from sklearn.datasets import make_blobs, load_diabetes, load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats

try:
    from sklearn_extra.cluster import KMedoids
    HAS_MEDOIDS = True
except:
    HAS_MEDOIDS = False

st.set_page_config(page_title="ML Explorer", layout="wide", initial_sidebar_state="collapsed")

# --- session_state SAFE initialization ---
for v, d in [
    ('raw_data', None), ('clean_data', None), ('features_used', []),
    ('algo_results', {}), ('comparison_kpis', {}), ('pca_data', {}),
    ('dendro', {}), ('current_comparison', []), ('outliers_count', 0),
    ('preprocessing_config', {}), ('analysis_history', []),
    ('target_column', None), ('X_train', None), ('X_test', None),
    ('y_train', None), ('y_test', None), ('classification_results', {}),
    ('best_k', None), ('knn_results', []), ('elbow_data', None),
    ('suggested_clusters', 3), ('current_tab', 'preprocess')
]:
    if v not in st.session_state:
        st.session_state[v] = d

# ------------ CSS - WEKA MODERN STYLE --------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

:root {
    --primary-dark: #1a237e;
    --primary: #2f3e9e;
    --primary-light: #4a5bd4;

    --secondary: #455a64;

    --accent-blue: #4f8bd6;
    --accent-green: #5fa87a;
    --accent-orange: #e39a4c;
    --accent-purple: #8e6bbf;

    --bg-dark: #263238;
    --bg-main: #eceff1;
    --bg-panel: #ffffff;

    --border: #b0bec5;
    --border-light: #cfd8dc;

    --text-dark: #212121;
    --text-muted: #546e7a;

    --success: #2e7d32;
    --danger: #c62828;
    --warning: #f57c00;
}

/* GLOBAL */
html, body, [data-testid="stAppViewContainer"] {
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.stApp {
    background: var(--bg-main) !important;
    font-family: 'Segoe UI', Roboto, sans-serif !important;
}

[data-testid="stSidebar"] { display: none !important; }

.main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}
body {
    margin: 0 !important;
    padding: 0 !important;
}
header { 
    visibility: hidden;
    height: 0px;
}

/* HEADER */
.weka-header {
    background: linear-gradient(135deg, var(--primary-dark), var(--primary));
    color: white;
    padding: 8px 20px;
    border-radius: 4px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(26,35,126,0.3);
}

.weka-header h1 {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
}

/* PANELS */
.weka-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
}

.panel-header {
    background: linear-gradient(180deg, #e8eaf6, #c5cae9);
    padding: 6px 12px;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--primary-dark);
    border-bottom: 1px solid var(--border);
}

.panel-content {
    padding: 8px 12px;
    font-size: 0.82rem;
    overflow-y: auto;
}

/* METRICS */
.metric-box {
    background: linear-gradient(135deg, #4a6b8c, #5d82a5);
    border: 1px solid #517292;
    border-radius: 4px;
    padding: 8px 12px;
    text-align: center;
    flex: 1;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.15),
                0 1px 2px rgba(0,0,0,0.12);
}

.metric-box.success {
    background: linear-gradient(135deg, #4fa67d, #65bb91);
    border-color: #4fa67d;
}

.metric-box.warning {
    background: linear-gradient(135deg, #f0a75b, #f3b76f);
    border-color: #ea9d52;
}

.metric-box.purple {
    background: linear-gradient(135deg, #9b72b1, #ac87c4);
    border-color: #9b72b1;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.25);
}

.metric-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.85);
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(180deg, var(--primary-light), var(--primary));
    color: white !important;
    border: 1px solid var(--primary-dark) !important;
    border-radius: 3px !important;
    font-size: 0.82rem !important;
}

.stButton > button:hover {
    background: linear-gradient(180deg, #6b7ce8, var(--primary-light));
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(180deg, var(--bg-dark), #37474f);
    border-bottom: 2px solid var(--primary);
}

.stTabs [data-baseweb="tab"] {
    color: #b0bec5 !important;
    font-size: 0.88rem;
}

.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* TABLES */
.results-table th {
    background: linear-gradient(180deg, #e8eaf6, #c5cae9);
    color: var(--primary-dark);
}

/* CONFUSION MATRIX */
.cm-tp { background: #b7dfc2; }
.cm-tn { background: #b8d3f0; }
.cm-fp { background: #e6b0b7; }
.cm-fn { background: #f0c48a; }

/* STATUS BAR */
.status-bar {
    background: linear-gradient(180deg, #f1f3f6, #dfe4ea);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 6px 12px;
    font-size: 0.78rem;
}

.status-ok { color: var(--success); font-weight: 600; }
.status-warning { color: var(--warning); font-weight: 600; }

/* INFO BOXES */
.info-box {
    background: #e3f2fd;
    border-left: 4px solid var(--accent-blue);
}

.success-box {
    background: #e8f5e9;
    border-left: 4px solid var(--accent-green);
}

.warning-box {
    background: #fff3e0;
    border-left: 4px solid var(--accent-orange);
}

/* SCROLLBAR */
::-webkit-scrollbar-thumb {
    background: #90a4ae;
    border-radius: 4px;
}


</style>
""", unsafe_allow_html=True)



def load_demo_data():
    X, y = make_blobs(n_samples=400, centers=5, cluster_std=[1.0, 1.5, 0.8, 1.2, 1.0], 
                      n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E'])
    df.loc[::25, 'Feature_C'] = np.nan
    df.loc[::30, 'Feature_D'] = np.nan
    df['Category'] = np.random.choice(['Type_1', 'Type_2', 'Type_3', 'Type_4', np.nan], size=len(df))
    df['Region'] = np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], size=len(df))
    outlier_indices = np.random.choice(len(df), size=15, replace=False)
    df.loc[outlier_indices, 'Feature_A'] = df['Feature_A'].mean() + 4 * df['Feature_A'].std()
    return df

def load_benchmark_dataset(name):
    """Load sklearn benchmark datasets for classification"""
    try:
        if name == "Diabetes":
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            # Convert to classification (binary)
            df['target'] = (data.target > data.target.mean()).astype(int)
        elif name == "Iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif name == "Wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif name == "Breast Cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        else:
            return load_demo_data()
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset '{name}'. Please try another dataset.")
        return None

def detect_outliers(df, method='IQR', contamination=0.1):
    try:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            st.warning("⚠️ No numeric columns found for outlier detection.")
            return np.zeros(len(df), dtype=bool)
        df_num = df[num_cols].dropna()
        if df_num.empty:
            st.warning("⚠️ No valid data available after removing missing values for outlier detection.")
            return np.zeros(len(df), dtype=bool)
        if method == 'IQR':
            Q1 = df_num.quantile(0.25)
            Q3 = df_num.quantile(0.75)
            IQR = Q3 - Q1
            mask = ((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)
        elif method == 'Z-Score':
            z_scores = np.abs(stats.zscore(df_num, nan_policy='omit'))
            mask = (z_scores > 3).any(axis=1)
        elif method == 'Isolation Forest':
            if len(df_num) < 10:
                st.warning("⚠️ Not enough samples for Isolation Forest. Using IQR instead.")
                Q1 = df_num.quantile(0.25)
                Q3 = df_num.quantile(0.75)
                IQR = Q3 - Q1
                mask = ((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)
            else:
                iso = IsolationForest(contamination=contamination, random_state=42)
                preds = iso.fit_predict(df_num)
                mask = preds == -1
        else:
            mask = np.zeros(len(df_num), dtype=bool)
        full_mask = np.zeros(len(df), dtype=bool)
        full_mask[df_num.index] = mask
        return full_mask
    except Exception as e:
        st.warning(f"⚠️ Outlier detection failed: Could not analyze data. Proceeding without outlier removal.")
        return np.zeros(len(df), dtype=bool)

def preprocess(df, selected_cols, missing_num, missing_cat, scaler_choice, remove_outliers=False, outlier_method='IQR'):
    errors = []
    if not selected_cols:
        errors.append("Sélectionnez au moins une colonne !")
        return None, errors, 0
    dfc = df[selected_cols].copy()
    outliers_removed_count = 0
    if remove_outliers:
        outliers_mask = detect_outliers(dfc, method=outlier_method)
        outliers_removed_count = outliers_mask.sum()
        dfc = dfc[~outliers_mask].reset_index(drop=True)
    num_cols = dfc.select_dtypes(include=np.number).columns
    cat_cols = dfc.select_dtypes(exclude=np.number).columns
    if len(num_cols):
        strat = {"Moyenne": "mean", "Médiane": "median", "Zéro": "constant"}[missing_num]
        imp = SimpleImputer(strategy=strat, fill_value=0)
        try:
            dfc[num_cols] = imp.fit_transform(dfc[num_cols])
        except Exception as e:
            errors.append(f"Erreur imputation numérique : {e}")
    if len(cat_cols):
        if missing_cat == "Supprimer":
            dfc = dfc.dropna(subset=cat_cols).reset_index(drop=True)
        else:
            strat_map = {"Mode": "most_frequent", "Valeur 'missing'": "constant"}
            fill_val = "missing" if missing_cat == "Valeur 'missing'" else None
            try:
                imp_cat = SimpleImputer(strategy=strat_map[missing_cat], fill_value=fill_val)
                dfc[cat_cols] = imp_cat.fit_transform(dfc[cat_cols].astype(str))
            except Exception as e:
                errors.append(f"Erreur imputation catégorielle : {e}")
    for c in cat_cols:
        try:
            le = LabelEncoder()
            dfc[c] = le.fit_transform(dfc[c].astype(str))
        except Exception as e:
            errors.append(f"Label encoding {c}: {e}")
    if scaler_choice != "Aucun":
        scaler_cls = {"StandardScaler": StandardScaler(),"MinMaxScaler": MinMaxScaler(),"RobustScaler": RobustScaler()}[scaler_choice]
        try:
            dfc[dfc.columns] = scaler_cls.fit_transform(dfc[dfc.columns])
        except Exception as e:
            errors.append(f"Erreur scaling : {e}")
    if dfc.empty or not dfc.shape[0]:
        errors.append("Données vides après nettoyage !")
        return None, errors, outliers_removed_count
    return dfc, errors, outliers_removed_count

def run_clustering(algo, params, X):
    try:
        if algo == "K-Means":
            model = KMeans(n_clusters=params.get('n_clusters', 3), random_state=42, n_init=10)
            labels = model.fit_predict(X)
            centers = model.cluster_centers_
            z = None
        elif algo == "K-Medoids" and HAS_MEDOIDS:
            model = KMedoids(n_clusters=params.get('n_clusters', 3), random_state=42, method='alternate')
            labels = model.fit_predict(X)
            centers = model.cluster_centers_
            z = None
        elif algo == "DBSCAN":
            model = DBSCAN(eps=params.get('eps', 0.5), min_samples=params.get('min_samples', 5))
            labels = model.fit_predict(X)
            centers = None
            z = None
        elif algo == "AGNES":
            model = AgglomerativeClustering(n_clusters=params.get('n_clusters', 3), 
                                            linkage=params.get('linkage', 'ward'))
            labels = model.fit_predict(X)
            z = linkage(X, method=params.get('linkage', 'ward'))
            centers = None
        elif algo == "DIANA":
            labels = diana_clustering(X, params.get('n_clusters', 3))
            centers = None
            z = linkage(X, method='complete')
        else:
            return None, None, None
        return labels, centers, z
    except Exception as e:
        st.error(f"Erreur clustering {algo}: {e}")
        return None, None, None

def diana_clustering(X, n_clusters):
    Z = linkage(X, method='complete')
    return AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit_predict(X)

def kpi_metrics(X, labels):
    try:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        else:
            sil, db, ch = None, None, None
        return {"clusters": n_clusters, "silhouette": sil, "davies-bouldin": db, "calinski": ch}
    except Exception:
        return {"clusters": None, "silhouette": None, "davies-bouldin": None, "calinski": None}

def get_recommendation(kpis):
    if not kpis:
        return None
    best_algo = None; best_score = -1
    for algo, metrics in kpis.items():
        if metrics['silhouette'] is not None:
            score = metrics['silhouette']
            if metrics['davies-bouldin'] is not None:
                score -= metrics['davies-bouldin'] * 0.1
            if score > best_score:
                best_score = score
                best_algo = algo
    return best_algo

def compute_elbow(X, max_k=10):
    """Compute elbow curve data for K-Means"""
    try:
        if X is None or len(X) == 0:
            st.error("❌ No data available for elbow analysis.")
            return None
        if len(X) < max_k:
            st.warning(f"⚠️ Dataset has only {len(X)} samples. Reducing max k to {len(X) - 1}.")
            max_k = max(2, len(X) - 1)
        
        inertias = []
        silhouettes = []
        K_range = range(2, max_k + 1)
        for k in K_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X, kmeans.labels_))
            except Exception:
                continue
        
        if len(inertias) < 2:
            st.error("❌ Could not compute enough clusters for elbow analysis.")
            return None
        
        # Find optimal k using elbow method (second derivative)
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            optimal_k = np.argmax(diffs2) + 3  # +3 because we start at k=2 and lose 2 points from diffs
            optimal_k = min(max(optimal_k, 2), max_k)
        else:
            optimal_k = 3
        
        return {
            'k_range': list(K_range)[:len(inertias)],
            'inertias': inertias,
            'silhouettes': silhouettes,
            'optimal_k': optimal_k
        }
    except Exception as e:
        st.error(f"❌ Elbow analysis failed. Please check your data and try again.")
        return None

def run_knn_experiment(X_train, X_test, y_train, y_test, max_k=10):
    """Run k-NN for k from 1 to max_k"""
    try:
        if X_train is None or len(X_train) == 0:
            st.error("❌ Training data is empty. Please split data first.")
            return []
        if len(X_train) < max_k:
            st.warning(f"⚠️ Training set has only {len(X_train)} samples. Reducing max k to {len(X_train)}.")
            max_k = len(X_train)
        
        results = []
        for k in range(1, max_k + 1):
            try:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                
                # Compute metrics
                cm = confusion_matrix(y_test, y_pred)
                if len(cm) == 2:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tp = np.diag(cm).sum()
                    fp = cm.sum() - np.diag(cm).sum()
                    fn = fp
                    tn = 0
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'k': k,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'confusion_matrix': cm
                })
            except Exception as e:
                st.warning(f"⚠️ k-NN failed for k={k}. Skipping...")
                continue
        
        if not results:
            st.error("❌ k-NN experiment failed for all values of k. Check your data.")
        return results
    except Exception as e:
        st.error(f"❌ k-NN experiment failed. Please ensure data is properly prepared.")
        return []

def run_classifier(clf_name, X_train, X_test, y_train, y_test, params=None):
    """Run a classifier and return metrics"""
    try:
        if X_train is None or len(X_train) == 0:
            st.error(f"❌ Cannot train {clf_name}: Training data is empty.")
            return None
        
        if clf_name == "Naive Bayes":
            clf = GaussianNB()
        elif clf_name == "C4.5 (Decision Tree)":
            clf = DecisionTreeClassifier(criterion='entropy', random_state=42, 
                                         max_depth=params.get('max_depth') if params else None)
        elif clf_name == "SVM":
            clf = SVC(kernel=params.get('kernel', 'rbf') if params else 'rbf',
                      C=params.get('C', 1.0) if params else 1.0,
                      random_state=42)
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=params.get('n_estimators', 100) if params else 100,
                                         random_state=42)
        else:
            st.warning(f"⚠️ Unknown classifier: {clf_name}")
            return None
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'model': clf
        }
    except Exception as e:
        st.error(f"❌ {clf_name} training failed. This may be due to incompatible data or insufficient samples.")
        return None

# ------------------ HEADER - WEKA STYLE ------------------
st.markdown("""
<div class="weka-header">
    <div>
        <h1>🔬 ML Explorer</h1>
        <p class="subtitle">Machine Learning Workbench • Classification & Clustering</p>
    </div>
    <div style="text-align: right; font-size: 0.8rem;">
        <span style="opacity: 0.8;">Session: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ MAIN TABS - WEKA STYLE ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Preprocess", "🎯 Classify", "🔮 Cluster", "📊 Visualize", "📋 Report"
])

# ============ TAB 1: PREPROCESS (WEKA STYLE) ============
with tab1:
    col_left, col_right = st.columns([1, 2])
    
    # LEFT PANEL - Data Source & Options
    with col_left:
        st.markdown("""<div class="panel-header">📂 Open File / Dataset</div>""", unsafe_allow_html=True)
        
        data_source = st.radio("Source:", ["Benchmark Dataset", "Upload File"], horizontal=True, key="data_src")
        
        # --- Benchmark Dataset ---
        if data_source == "Benchmark Dataset":
            dataset_name = st.selectbox("Dataset:", ["Diabetes", "Iris", "Wine", "Breast Cancer", "Demo (Clustering)"])
            if st.button("📥 Load Dataset", use_container_width=True):
                try:
                    with st.spinner("Loading dataset..."):
                        if dataset_name == "Demo (Clustering)":
                            st.session_state.raw_data = load_demo_data()
                        else:
                            st.session_state.raw_data = load_benchmark_dataset(dataset_name)
                        if st.session_state.raw_data is not None:
                            st.success(f"✓ {dataset_name} loaded!")
                        else:
                            st.error("❌ Failed to load dataset. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error loading dataset: Could not load {dataset_name}. Please try another dataset.")
        
        # --- File Upload ---
        else:
            # Restore previously uploaded file if exists
            if "uploaded_file" not in st.session_state:
                st.session_state.uploaded_file = None

            uploaded = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed", key="file_upload")
            if uploaded:
                st.session_state.uploaded_file = uploaded  # save uploaded file in session
                try:
                    with st.spinner("Reading file..."):
                        if uploaded.name.lower().endswith('.csv'):
                            # Read with '?' as NA values
                            st.session_state.raw_data = pd.read_csv(uploaded, na_values=['?', '??', 'NA', 'N/A', 'na', 'n/a', ''])
                        else:
                            st.session_state.raw_data = pd.read_excel(uploaded, na_values=['?', '??', 'NA', 'N/A', 'na', 'n/a', ''])
                        
                        # Try to convert object columns that should be numeric
                        for col in st.session_state.raw_data.columns:
                            if st.session_state.raw_data[col].dtype == 'object':
                                try:
                                    st.session_state.raw_data[col] = pd.to_numeric(st.session_state.raw_data[col], errors='ignore')
                                except:
                                    pass
                        
                        if st.session_state.raw_data is None or st.session_state.raw_data.empty:
                            st.error("❌ The file appears to be empty. Please upload a file with data.")
                        elif len(st.session_state.raw_data.columns) == 0:
                            st.error("❌ No columns detected in the file. Please check the file format.")
                        else:
                            st.success("✓ File loaded!")
                except pd.errors.EmptyDataError:
                    st.error("❌ The file is empty. Please upload a file with data.")
                except pd.errors.ParserError:
                    st.error("❌ Could not parse the file. Please ensure it's a valid CSV or Excel file.")
                except Exception as e:
                    st.error(f"❌ Error reading file: Please ensure the file format is correct (CSV or Excel).")
            # Load previously uploaded file after rerun
            elif st.session_state.uploaded_file:
                try:
                    uploaded = st.session_state.uploaded_file
                    if uploaded.name.lower().endswith('.csv'):
                        st.session_state.raw_data = pd.read_csv(uploaded, na_values=['?', '??', 'NA', 'N/A', 'na', 'n/a', ''])
                    else:
                        st.session_state.raw_data = pd.read_excel(uploaded, na_values=['?', '??', 'NA', 'N/A', 'na', 'n/a', ''])
                    
                    # Try to convert object columns that should be numeric
                    for col in st.session_state.raw_data.columns:
                        if st.session_state.raw_data[col].dtype == 'object':
                            try:
                                st.session_state.raw_data[col] = pd.to_numeric(st.session_state.raw_data[col], errors='ignore')
                            except:
                                pass
                except Exception:
                    st.warning("⚠️ Could not reload previous file. Please upload again.")
                    st.session_state.uploaded_file = None

        st.markdown("---")
        st.markdown("""<div class="panel-header">⚙️ Preprocessing Options</div>""", unsafe_allow_html=True)
        
        # --- Preprocessing Section ---
        if st.session_state.raw_data is not None:
            df = st.session_state.raw_data
            all_cols = df.columns.tolist()
            
            cols_to_use = st.multiselect("Attributes:", all_cols, default=all_cols[:min(8, len(all_cols))])
            
            col_a, col_b = st.columns(2)
            with col_a:
                num_strat = st.selectbox("Missing (Num):", ["Mean", "Median", "Zero"], key="num_miss")
            with col_b:
                cat_strat = st.selectbox("Missing (Cat):", ["Mode", "'missing'", "Remove"], key="cat_miss")
            
            scaler = st.selectbox("Normalization:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
            
            remove_out = st.checkbox("Remove Outliers", value=False)
            if remove_out:
                out_method = st.selectbox("Method:", ["IQR", "Z-Score", "Isolation Forest"])
            else:
                out_method = "IQR"
            
            if st.button("▶ Apply Preprocessing", use_container_width=True):
                try:
                    if not cols_to_use:
                        st.warning("⚠️ Please select at least one attribute to preprocess.")
                    else:
                        with st.spinner("Preprocessing data..."):
                            # Map strategy names
                            num_map = {"Mean": "Moyenne", "Median": "Médiane", "Zero": "Zéro"}
                            cat_map = {"Mode": "Mode", "'missing'": "Valeur 'missing'", "Remove": "Supprimer"}
                            scaler_map = {"None": "Aucun"}
                            
                            clean, errors, out_count = preprocess(
                                df, cols_to_use, 
                                num_map.get(num_strat, num_strat),
                                cat_map.get(cat_strat, cat_strat),
                                scaler_map.get(scaler, scaler),
                                remove_out, out_method
                            )
                            if errors:
                                for e in errors:
                                    st.error(f"❌ {e}")
                            elif clean is None or clean.empty:
                                st.error("❌ Preprocessing resulted in empty data. Try different settings.")
                            else:
                                st.session_state.clean_data = clean
                                st.session_state.features_used = cols_to_use
                                st.session_state.outliers_count = out_count
                                st.success(f"✓ Preprocessed! {out_count} outliers removed." if out_count else "✓ Preprocessed!")
                except Exception as e:
                    st.error("❌ Preprocessing failed. Please check your data and settings.")
        else:
            st.markdown('<div class="info-box">Load a dataset to configure preprocessing.</div>', unsafe_allow_html=True)

    # RIGHT PANEL - Data View
    with col_right:
        if st.session_state.raw_data is not None:
            df = st.session_state.raw_data
            
            # Metrics Row
            st.markdown("""<div class="panel-header">📊 Current Relation</div>""", unsafe_allow_html=True)
            
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{df.shape[0]:,}</div><div class="metric-label">Instances</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box success"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Attributes</div></div>', unsafe_allow_html=True)
            with m3:
                num_c = len(df.select_dtypes(include=np.number).columns)
                st.markdown(f'<div class="metric-box"><div class="metric-value">{num_c}</div><div class="metric-label">Numeric</div></div>', unsafe_allow_html=True)
            with m4:
                cat_c = len(df.select_dtypes(exclude=np.number).columns)
                st.markdown(f'<div class="metric-box warning"><div class="metric-value">{cat_c}</div><div class="metric-label">Nominal</div></div>', unsafe_allow_html=True)
            with m5:
                na_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                st.markdown(f'<div class="metric-box purple"><div class="metric-value">{na_pct:.1f}%</div><div class="metric-label">Missing</div></div>', unsafe_allow_html=True)
            
            # Data Tables
            col_tbl1, col_tbl2 = st.columns([1.2, 1])
            
            with col_tbl1:
                st.markdown("""<div class="panel-header" style="margin-top:10px;">📋 Data Preview</div>""", unsafe_allow_html=True)
                st.dataframe(df.head(12), use_container_width=True, height=280)
            
            with col_tbl2:
                st.markdown("""<div class="panel-header" style="margin-top:10px;">📈 Attribute Summary</div>""", unsafe_allow_html=True)
                summary_data = []
                for col in df.columns[:15]:
                    # Determine if continuous or discrete
                    if pd.api.types.is_numeric_dtype(df[col]):
                        unique_count = df[col].nunique()
                        # If numeric with many unique values relative to size, it's continuous
                        if unique_count > 10 or (df[col].dtype in ['float64', 'float32']):
                            dtype = "Continuous"
                        else:
                            dtype = "Discrete (Num)"
                    else:
                        dtype = "Discrete (Cat)"
                    missing = df[col].isna().sum()
                    unique = df[col].nunique()
                    missing_pct = f"{(missing/len(df)*100):.1f}%" if len(df) > 0 else "0%"
                    summary_data.append({"Attribute": col[:18], "Type": dtype, "Missing": f"{missing} ({missing_pct})", "Unique": unique})
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, height=280)
            
            # Distribution
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                st.markdown("""<div class="panel-header" style="margin-top:10px;">📉 Distribution</div>""", unsafe_allow_html=True)
                sel_col = st.selectbox("Attribute:", num_cols, key="dist_attr")
                c1, c2 = st.columns(2)
                try:
                    with c1:
                        fig = px.histogram(df, x=sel_col, nbins=25, color_discrete_sequence=['#5c6bc0'])
                        fig.update_layout(height=220, margin=dict(l=20,r=20,t=30,b=20), title=f"Histogram: {sel_col}")
                        st.plotly_chart(fig, use_container_width=True, key="hist_pre")
                    with c2:
                        fig = px.box(df, y=sel_col, color_discrete_sequence=['#66bb6a'])
                        fig.update_layout(height=220, margin=dict(l=20,r=20,t=30,b=20), title=f"Box Plot: {sel_col}")
                        st.plotly_chart(fig, use_container_width=True, key="box_pre")
                except Exception:
                    st.warning("⚠️ Could not render distribution charts for this attribute.")
                    st.plotly_chart(fig, use_container_width=True, key="box_pre")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px; color:#546e7a;">
                <div style="font-size:3rem; margin-bottom:15px;">📂</div>
                <h3 style="color:#1a237e;">No Data Loaded</h3>
                <p>Select a benchmark dataset or upload a file from the left panel.</p>
            </div>
            """, unsafe_allow_html=True)

# ============ TAB 2: CLASSIFY (WEKA STYLE) ============
with tab2:
    if st.session_state.raw_data is None:
        st.markdown('<div class="info-box">⚠️ Load data in the Preprocess tab first.</div>', unsafe_allow_html=True)
    elif st.session_state.clean_data is None:
        st.markdown('<div class="info-box">⚠️ Please preprocess your data in the Preprocess tab first before classification.<br><br>Go to <b>📂 Preprocess</b> → Select attributes → Click <b>▶ Apply Preprocessing</b></div>', unsafe_allow_html=True)
    else:
        df = st.session_state.clean_data
        
        col_left, col_right = st.columns([1, 2])
        
        # LEFT PANEL - Classifier Settings
        with col_left:
            st.markdown("""<div class="panel-header">🎯 Classifier Configuration</div>""", unsafe_allow_html=True)
            
            # Target Selection
            all_cols = df.columns.tolist()
            target_col = st.selectbox("Target (Class):", all_cols, index=len(all_cols)-1 if 'target' in all_cols else 0)
            
            # Train/Test Split
            st.markdown("**Data Partitioning:**")
            test_size = st.slider("Test Size %:", 10, 40, 20, 5)
            
            feature_cols = [c for c in all_cols if c != target_col]
            
            if st.button("🔀 Split Data", use_container_width=True):
                try:
                    X = df[feature_cols]
                    if X.empty:
                        st.error("❌ No features available! Please ensure your data has columns other than target.")
                    elif len(df) < 10:
                        st.error("❌ Dataset is too small (minimum 10 samples required for splitting).")
                    else:
                        y = df[target_col]
                        
                        if y.isna().all():
                            st.error("❌ Target column contains only missing values. Please select a different target.")
                        else:
                            y = pd.Series(y).fillna(0).astype(int)
                            
                            n_classes = len(np.unique(y))
                            if n_classes < 2:
                                st.error("❌ Target column must have at least 2 classes for classification.")
                            else:
                                # Check if stratification is possible
                                min_class_count = pd.Series(y).value_counts().min()
                                if min_class_count < 2:
                                    st.warning("⚠️ Some classes have very few samples. Stratification disabled.")
                                    stratify = None
                                else:
                                    stratify = y
                                
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size/100, random_state=42, stratify=stratify
                                )
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                st.session_state.target_column = target_col
                                st.success(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")
                except ValueError as e:
                    if "least populated class" in str(e).lower():
                        st.error("❌ Some classes have too few samples. Try increasing the test size or use a larger dataset.")
                    else:
                        st.error(f"❌ Data split failed. Please check your data and target column.")
                except Exception as e:
                    st.error(f"❌ Data split failed. Please ensure your data is properly formatted.")
            
            st.markdown("---")
            st.markdown("""<div class="panel-header">🔧 Algorithms</div>""", unsafe_allow_html=True)
            
            # k-NN Section
            st.markdown("**k-NN (k = 1 to 10):**")
            if st.button("▶ Run k-NN Experiment", use_container_width=True):
                if st.session_state.X_train is not None:
                    try:
                        with st.spinner("Running k-NN experiment..."):
                            results = run_knn_experiment(
                                st.session_state.X_train, st.session_state.X_test,
                                st.session_state.y_train, st.session_state.y_test, max_k=10
                            )
                            if results:
                                st.session_state.knn_results = results
                                best = max(results, key=lambda x: x['precision'])
                                st.session_state.best_k = best['k']
                                st.success(f"✓ Best k = {best['k']} (Precision: {best['precision']:.3f})")
                            else:
                                st.error("❌ k-NN experiment produced no results. Check your data.")
                    except Exception as e:
                        st.error("❌ k-NN experiment failed. Please check your training data.")
                else:
                    st.warning("⚠️ Please split data first before running k-NN!")
            
            st.markdown("---")
            
            # Other Classifiers
            st.markdown("**Other Classifiers:**")
            
            clf_options = ["Naive Bayes", "C4.5 (Decision Tree)", "SVM", "Random Forest"]
            
            # SVM Parameters
            with st.expander("SVM Parameters"):
                svm_kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly", "sigmoid"])
                svm_c = st.slider("C:", 0.1, 10.0, 1.0, 0.1)
            
            # Decision Tree Parameters
            with st.expander("Decision Tree Parameters"):
                dt_depth = st.slider("Max Depth:", 1, 20, 5)
            
            selected_clfs = st.multiselect("Select:", clf_options, default=["Naive Bayes"])
            
            if st.button("▶ Run Classifiers", use_container_width=True):
                if st.session_state.X_train is not None:
                    if not selected_clfs:
                        st.warning("⚠️ Please select at least one classifier to run.")
                    else:
                        try:
                            with st.spinner("Training classifiers..."):
                                successful = 0
                                for clf_name in selected_clfs:
                                    params = {}
                                    if clf_name == "SVM":
                                        params = {'kernel': svm_kernel, 'C': svm_c}
                                    elif clf_name == "C4.5 (Decision Tree)":
                                        params = {'max_depth': dt_depth}
                                    
                                    result = run_classifier(
                                        clf_name,
                                        st.session_state.X_train, st.session_state.X_test,
                                        st.session_state.y_train, st.session_state.y_test,
                                        params
                                    )
                                    if result:
                                        st.session_state.classification_results[clf_name] = result
                                        successful += 1
                                
                                if successful > 0:
                                    st.success(f"✓ {successful} classifier(s) trained successfully!")
                                else:
                                    st.error("❌ All classifiers failed. Please check your data.")
                        except Exception as e:
                            st.error("❌ Classification failed. Please ensure your data is properly prepared.")
                else:
                    st.warning("⚠️ Please split data first before running classifiers!")
        
        # RIGHT PANEL - Results
        with col_right:
            st.markdown("""<div class="panel-header">📊 Classification Results</div>""", unsafe_allow_html=True)
            
            # Dataset Info
            if st.session_state.X_train is not None:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{len(st.session_state.X_train)}</div><div class="metric-label">Train Size</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-box warning"><div class="metric-value">{len(st.session_state.X_test)}</div><div class="metric-label">Test Size</div></div>', unsafe_allow_html=True)
                with m3:
                    n_classes = len(np.unique(st.session_state.y_train))
                    st.markdown(f'<div class="metric-box success"><div class="metric-value">{n_classes}</div><div class="metric-label">Classes</div></div>', unsafe_allow_html=True)
                with m4:
                    n_features = st.session_state.X_train.shape[1]
                    st.markdown(f'<div class="metric-box purple"><div class="metric-value">{n_features}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
            
            # k-NN Results Grid
            if st.session_state.knn_results:
                st.markdown("""<div class="panel-header" style="margin-top:10px;">🎯 k-NN Results (k = 1 to 10)</div>""", unsafe_allow_html=True)
                
                col_tbl, col_chart = st.columns([1, 1])
                
                with col_tbl:
                    knn_df = pd.DataFrame([{
                        'k': r['k'],
                        'Accuracy': f"{r['accuracy']:.3f}",
                        'Precision': f"{r['precision']:.3f}",
                        'Recall': f"{r['recall']:.3f}",
                        'F1': f"{r['f1']:.3f}"
                    } for r in st.session_state.knn_results])
                    st.dataframe(knn_df, use_container_width=True, height=250)
                
                with col_chart:
                    knn_plot_df = pd.DataFrame(st.session_state.knn_results)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=knn_plot_df['k'], y=knn_plot_df['precision'], 
                                            mode='lines+markers', name='Precision', 
                                            line=dict(color='#5c6bc0', width=3)))
                    fig.add_trace(go.Scatter(x=knn_plot_df['k'], y=knn_plot_df['accuracy'],
                                            mode='lines+markers', name='Accuracy',
                                            line=dict(color='#66bb6a', width=2, dash='dash')))
                    
                    # Mark best k
                    if st.session_state.best_k:
                        best_prec = knn_plot_df[knn_plot_df['k'] == st.session_state.best_k]['precision'].values[0]
                        fig.add_vline(x=st.session_state.best_k, line_dash="dot", line_color="red")
                        fig.add_annotation(x=st.session_state.best_k, y=best_prec, 
                                          text=f"Best k={st.session_state.best_k}", showarrow=True)
                    
                    fig.update_layout(
                        title="Precision vs k", height=250,
                        margin=dict(l=20,r=20,t=40,b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True, key="knn_curve")
                
                # Confusion Matrix for best k
                if st.session_state.best_k:
                    best_result = next(r for r in st.session_state.knn_results if r['k'] == st.session_state.best_k)
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>🏆 Optimal k = {st.session_state.best_k}</strong> | 
                        Precision: {best_result['precision']:.4f} | 
                        Recall: {best_result['recall']:.4f} | 
                        F1: {best_result['f1']:.4f}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Other Classifier Results
            if st.session_state.classification_results:
                st.markdown("""<div class="panel-header" style="margin-top:10px;">📈 Classifier Comparison</div>""", unsafe_allow_html=True)
                
                results_df = pd.DataFrame([{
                    'Classifier': name,
                    'Accuracy': f"{res['accuracy']:.4f}",
                    'Precision': f"{res['precision']:.4f}",
                    'Recall': f"{res['recall']:.4f}",
                    'F1-Score': f"{res['f1']:.4f}"
                } for name, res in st.session_state.classification_results.items()])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Confusion Matrices
                st.markdown("""<div class="panel-header" style="margin-top:10px;">🔢 Confusion Matrices</div>""", unsafe_allow_html=True)
                
                cm_cols = st.columns(min(3, len(st.session_state.classification_results)))
                for idx, (name, res) in enumerate(st.session_state.classification_results.items()):
                    with cm_cols[idx % len(cm_cols)]:
                        cm = res['confusion_matrix']
                        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                       title=f"{name}")
                        fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20))
                        st.plotly_chart(fig, use_container_width=True, key=f"cm_{name}")

# ============ TAB 3: CLUSTER (WEKA STYLE) ============
with tab3:
    if st.session_state.raw_data is None:
        st.markdown('<div class="info-box">⚠️ Load data in the Preprocess tab first.</div>', unsafe_allow_html=True)
    elif st.session_state.clean_data is None:
        st.markdown('<div class="info-box">⚠️ Please preprocess your data in the Preprocess tab first before clustering.<br><br>Go to <b>📂 Preprocess</b> → Select attributes → Click <b>▶ Apply Preprocessing</b></div>', unsafe_allow_html=True)
    else:
        df = st.session_state.clean_data
        X_cluster = df.values
        
        col_left, col_right = st.columns([1, 2])
        
        # LEFT PANEL - Clustering Settings
        with col_left:
            st.markdown("""<div class="panel-header">🔮 Clusterer Configuration</div>""", unsafe_allow_html=True)
            
            # Algorithm Selection
            algos_avail = ["K-Means"]
            if HAS_MEDOIDS:
                algos_avail.append("K-Medoids")
            algos_avail += ["DBSCAN", "AGNES", "DIANA"]
            
            selected_algos = st.multiselect("Algorithms:", algos_avail, default=["K-Means"])
            
            st.markdown("---")
            
            # Elbow Method
            st.markdown("""<div class="panel-header">📈 Elbow Method</div>""", unsafe_allow_html=True)
            max_k_elbow = st.slider("Max k for Elbow:", 5, 15, 10)
            
            if st.button("📊 Compute Elbow Curve", use_container_width=True):
                try:
                    with st.spinner("Computing elbow curve..."):
                        if X_cluster is None or len(X_cluster) == 0:
                            st.error("❌ No data available for elbow analysis.")
                        elif len(X_cluster) < 3:
                            st.error("❌ Need at least 3 samples for elbow analysis.")
                        else:
                            elbow = compute_elbow(X_cluster, max_k_elbow)
                            if elbow:
                                st.session_state.elbow_data = elbow
                                st.session_state.suggested_clusters = elbow['optimal_k']
                                st.success(f"✓ Suggested k = {elbow['optimal_k']}")
                            else:
                                st.error("❌ Could not compute elbow curve.")
                except Exception as e:
                    st.error("❌ Elbow curve computation failed. Please check your data.")
            
            # Show suggested k
            if st.session_state.elbow_data:
                st.markdown(f"""
                <div class="success-box">
                    <strong>🎯 Suggested k = {st.session_state.suggested_clusters}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                accept_suggested = st.checkbox("Use suggested k", value=True)
                if accept_suggested:
                    n_clusters = st.session_state.suggested_clusters
                else:
                    n_clusters = st.slider("Manual k:", 2, 12, 4, key="manual_k")
            else:
                n_clusters = st.slider("Number of clusters (k):", 2, 12, 4, key="default_k")
            
            st.markdown("---")
            
            # Algorithm-specific params
            algo_params = {}
            for algo in selected_algos:
                if algo in ["K-Means", "K-Medoids", "AGNES", "DIANA"]:
                    algo_params[algo] = {"n_clusters": n_clusters}
                    if algo == "AGNES":
                        with st.expander(f"{algo} Options"):
                            linkage_m = st.selectbox("Linkage:", ["ward", "complete", "average", "single"], key=f"link_{algo}")
                            algo_params[algo]["linkage"] = linkage_m
                elif algo == "DBSCAN":
                    with st.expander("DBSCAN Options"):
                        eps = st.slider("Epsilon:", 0.1, 3.0, 0.5, 0.1)
                        min_samples = st.slider("Min Samples:", 2, 15, 5)
                        algo_params[algo] = {"eps": eps, "min_samples": min_samples}
            
            st.markdown("---")
            
            if st.button("▶ Run Clustering", use_container_width=True):
                if not selected_algos:
                    st.warning("⚠️ Please select at least one algorithm!")
                elif X_cluster is None or len(X_cluster) == 0:
                    st.error("❌ No data available for clustering.")
                elif len(X_cluster) < n_clusters:
                    st.error(f"❌ Not enough samples ({len(X_cluster)}) for {n_clusters} clusters.")
                else:
                    try:
                        st.session_state.algo_results.clear()
                        st.session_state.pca_data.clear()
                        st.session_state.dendro.clear()
                        st.session_state.comparison_kpis.clear()
                        st.session_state.current_comparison = selected_algos
                        
                        progress = st.progress(0)
                        successful = 0
                        for i, algo in enumerate(selected_algos):
                            try:
                                labels, centers, dendro_z = run_clustering(algo, algo_params.get(algo, {}), X_cluster)
                                if labels is not None:
                                    st.session_state.algo_results[algo] = labels
                                    st.session_state.comparison_kpis[algo] = kpi_metrics(X_cluster, labels)
                                    successful += 1
                                    
                                    n_components = min(3, X_cluster.shape[1])
                                    if n_components >= 1:
                                        try:
                                            pca_result = PCA(n_components=n_components).fit_transform(X_cluster)
                                            st.session_state.pca_data[algo] = {
                                                'pca': pca_result, 'labels': labels, 'centers': centers,
                                                'n_features': X_cluster.shape[1], 'n_components': n_components
                                            }
                                        except Exception:
                                            pass  # PCA visualization failed but clustering succeeded
                                    if dendro_z is not None:
                                        st.session_state.dendro[algo] = dendro_z
                            except Exception as e:
                                st.warning(f"⚠️ {algo} failed. Skipping...")
                            progress.progress((i + 1) / len(selected_algos))
                        progress.empty()
                        
                        if successful > 0:
                            st.success(f"✓ Clustering complete! ({successful}/{len(selected_algos)} algorithms succeeded)")
                        else:
                            st.error("❌ All clustering algorithms failed. Please check your data.")
                    except Exception as e:
                        st.error("❌ Clustering failed. Please ensure your data is properly prepared.")
        
        # RIGHT PANEL - Results
        with col_right:
            st.markdown("""<div class="panel-header">📊 Clustering Results</div>""", unsafe_allow_html=True)
            
            # Elbow Curve
            if st.session_state.elbow_data:
                elbow = st.session_state.elbow_data
                
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=elbow['k_range'], y=elbow['inertias'],
                        mode='lines+markers', name='Inertia',
                        line=dict(color='#5c6bc0', width=3)
                    ))
                    fig.add_vline(x=elbow['optimal_k'], line_dash="dash", line_color="red")
                    fig.add_annotation(x=elbow['optimal_k'], y=elbow['inertias'][elbow['optimal_k']-2],
                                      text=f"Optimal k={elbow['optimal_k']}", showarrow=True)
                    fig.update_layout(title="Elbow Curve (Inertia)", height=220, 
                                     margin=dict(l=20,r=20,t=40,b=20))
                    st.plotly_chart(fig, use_container_width=True, key="elbow_inertia")
                
                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=elbow['k_range'], y=elbow['silhouettes'],
                        mode='lines+markers', name='Silhouette',
                        line=dict(color='#66bb6a', width=3)
                    ))
                    best_sil_k = elbow['k_range'][np.argmax(elbow['silhouettes'])]
                    fig.add_vline(x=best_sil_k, line_dash="dash", line_color="green")
                    fig.update_layout(title="Silhouette Score vs k", height=220,
                                     margin=dict(l=20,r=20,t=40,b=20))
                    st.plotly_chart(fig, use_container_width=True, key="elbow_sil")
            
            # Clustering Results Table
            if st.session_state.comparison_kpis:
                st.markdown("""<div class="panel-header" style="margin-top:10px;">📈 Algorithm Comparison</div>""", unsafe_allow_html=True)
                
                cmp_df = pd.DataFrame([{
                    "Algorithm": a,
                    "Clusters": st.session_state.comparison_kpis[a]['clusters'],
                    "Silhouette": f"{st.session_state.comparison_kpis[a]['silhouette']:.4f}" if st.session_state.comparison_kpis[a]['silhouette'] else "N/A",
                    "Davies-Bouldin": f"{st.session_state.comparison_kpis[a]['davies-bouldin']:.4f}" if st.session_state.comparison_kpis[a]['davies-bouldin'] else "N/A",
                    "Calinski-H.": f"{st.session_state.comparison_kpis[a]['calinski']:.1f}" if st.session_state.comparison_kpis[a]['calinski'] else "N/A"
                } for a in st.session_state.comparison_kpis])
                st.dataframe(cmp_df, use_container_width=True)
                
                # Best Algorithm
                best = get_recommendation(st.session_state.comparison_kpis)
                if best:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>🏆 Recommended: {best}</strong> (Best silhouette/DB ratio)
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cluster Visualizations
                st.markdown("""<div class="panel-header" style="margin-top:10px;">🔍 Cluster Visualization</div>""", unsafe_allow_html=True)
                
                viz_cols = st.columns(min(2, len(st.session_state.pca_data)))
                for idx, (algo, pca_info) in enumerate(st.session_state.pca_data.items()):
                    with viz_cols[idx % len(viz_cols)]:
                        pca_data = pca_info['pca']
                        labels = pca_info['labels']
                        
                        if pca_info['n_components'] >= 2:
                            df_pca = pd.DataFrame({'PC1': pca_data[:, 0], 'PC2': pca_data[:, 1], 
                                                  'Cluster': labels.astype(str)})
                            fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                                           title=f"{algo}", opacity=0.7,
                                           color_discrete_sequence=px.colors.qualitative.Set2)
                            fig.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
                            st.plotly_chart(fig, use_container_width=True, key=f"pca_{algo}_{idx}")

# ============ TAB 4: VISUALIZE (WEKA STYLE) ============
with tab4:
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.markdown("""<div class="panel-header">🎨 Visualization Options</div>""", unsafe_allow_html=True)
        
        viz_type = st.radio("View:", ["Clustering", "Classification", "Data"], key="viz_type")
        
        if viz_type == "Clustering" and st.session_state.algo_results:
            selected_viz_algo = st.selectbox("Algorithm:", list(st.session_state.algo_results.keys()))
            dim_option = st.radio("Projection:", ["2D", "3D"], horizontal=True)
            show_centers = st.checkbox("Show Centroids", value=True)
            show_dendro = st.checkbox("Show Dendrogram", value=True)
        
        elif viz_type == "Classification":
            if st.session_state.knn_results:
                show_knn_curve = st.checkbox("k-NN Curve", value=True)
            if st.session_state.classification_results:
                show_cm = st.checkbox("Confusion Matrices", value=True)
                show_comparison = st.checkbox("Metrics Comparison", value=True)
        
        elif viz_type == "Data" and st.session_state.raw_data is not None:
            df_viz = st.session_state.raw_data
            num_cols_viz = df_viz.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols_viz) >= 2:
                x_col = st.selectbox("X-axis:", num_cols_viz, key="x_viz")
                y_col = st.selectbox("Y-axis:", num_cols_viz, index=min(1, len(num_cols_viz)-1), key="y_viz")
                color_col = st.selectbox("Color by:", ["None"] + df_viz.columns.tolist(), key="color_viz")
    
    with col_right:
        st.markdown("""<div class="panel-header">📊 Visualization Panel</div>""", unsafe_allow_html=True)
        
        if viz_type == "Clustering" and st.session_state.algo_results:
            if selected_viz_algo in st.session_state.pca_data:
                try:
                    pca_info = st.session_state.pca_data[selected_viz_algo]
                    pca_data = pca_info['pca']
                    labels = pca_info['labels']
                    centers = pca_info['centers']
                    n_components = pca_info['n_components']
                    
                    c1, c2 = st.columns([2, 1])
                    
                    with c1:
                        if n_components >= 2:
                            try:
                                if dim_option == "3D" and n_components >= 3:
                                    df_pca = pd.DataFrame({'PC1': pca_data[:, 0], 'PC2': pca_data[:, 1], 
                                                          'PC3': pca_data[:, 2], 'Cluster': labels.astype(str)})
                                    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster',
                                                       title=f"{selected_viz_algo} - 3D PCA", opacity=0.7,
                                                       color_discrete_sequence=px.colors.qualitative.Set2)
                                    if show_centers and centers is not None:
                                        try:
                                            pca_m = PCA(n_components=3).fit(st.session_state.clean_data.values if st.session_state.clean_data is not None else st.session_state.raw_data.select_dtypes(include=np.number).fillna(0).values)
                                            c_pca = pca_m.transform(centers)
                                            fig.add_trace(go.Scatter3d(x=c_pca[:, 0], y=c_pca[:, 1], z=c_pca[:, 2],
                                                                      mode='markers', marker=dict(symbol='diamond', size=8, color='red'),
                                                                      name='Centroids'))
                                        except Exception:
                                            pass  # Centroids failed but main viz works
                                    fig.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20))
                                else:
                                    df_pca = pd.DataFrame({'PC1': pca_data[:, 0], 'PC2': pca_data[:, 1], 
                                                          'Cluster': labels.astype(str)})
                                    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                                                   title=f"{selected_viz_algo} - 2D PCA", opacity=0.7,
                                                   color_discrete_sequence=px.colors.qualitative.Set2)
                                    if show_centers and centers is not None:
                                        try:
                                            X_data = st.session_state.clean_data.values if st.session_state.clean_data is not None else st.session_state.raw_data.select_dtypes(include=np.number).fillna(0).values
                                            pca_m = PCA(n_components=2).fit(X_data)
                                            c_pca = pca_m.transform(centers)
                                            fig.add_trace(go.Scatter(x=c_pca[:, 0], y=c_pca[:, 1],
                                                                    mode='markers', marker=dict(symbol='star', size=15, color='red', 
                                                                                               line=dict(width=2, color='black')),
                                                                    name='Centroids'))
                                        except Exception:
                                            pass  # Centroids failed but main viz works
                                    fig.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20))
                                st.plotly_chart(fig, use_container_width=True, key="main_pca_viz")
                            except Exception:
                                st.warning("⚠️ Could not render cluster visualization.")
                        else:
                            st.warning("⚠️ Not enough components for visualization")
                    
                    with c2:
                        try:
                            # Cluster distribution
                            cluster_counts = pd.Series(labels).value_counts().sort_index()
                            fig = px.pie(values=cluster_counts.values, names=[f"C{i}" for i in cluster_counts.index],
                                        title="Cluster Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
                            fig.update_layout(height=200, margin=dict(l=10,r=10,t=40,b=10))
                            st.plotly_chart(fig, use_container_width=True, key="cluster_pie")
                        except Exception:
                            st.info("Cluster distribution not available.")
                        
                        # Metrics
                        kpi = st.session_state.comparison_kpis.get(selected_viz_algo, {})
                        if kpi.get('silhouette'):
                            st.markdown(f"""
                            <div class="metric-box" style="margin-top:10px;">
                                <div class="metric-value">{kpi['silhouette']:.3f}</div>
                                <div class="metric-label">Silhouette</div>
                            </div>
                            """, unsafe_allow_html=True)
                        if kpi.get('davies-bouldin'):
                            st.markdown(f"""
                            <div class="metric-box warning" style="margin-top:5px;">
                                <div class="metric-value">{kpi['davies-bouldin']:.3f}</div>
                                <div class="metric-label">Davies-Bouldin</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Dendrogram
                    if show_dendro and selected_viz_algo in st.session_state.dendro:
                        try:
                            st.markdown("---")
                            dendro_data = dendrogram(st.session_state.dendro[selected_viz_algo], no_plot=True)
                            fig = go.Figure()
                            for i in range(len(dendro_data['icoord'])):
                                fig.add_trace(go.Scatter(
                                    x=dendro_data['icoord'][i], y=dendro_data['dcoord'][i],
                                    mode='lines', line=dict(color='#5c6bc0', width=2), hoverinfo='skip'
                                ))
                            fig.update_layout(title=f"Dendrogram - {selected_viz_algo}", height=250,
                                             margin=dict(l=20,r=20,t=40,b=20), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True, key="dendro_main")
                        except Exception:
                            st.warning("⚠️ Could not render dendrogram.")
                except Exception:
                    st.error("❌ Could not display clustering visualization. Please try running clustering again.")
        
        elif viz_type == "Classification":
            try:
                if st.session_state.knn_results and 'show_knn_curve' in dir() and show_knn_curve:
                    try:
                        knn_df = pd.DataFrame(st.session_state.knn_results)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=knn_df['k'], y=knn_df['precision'], mode='lines+markers', 
                                                name='Precision', line=dict(color='#5c6bc0', width=3)))
                        fig.add_trace(go.Scatter(x=knn_df['k'], y=knn_df['recall'], mode='lines+markers',
                                                name='Recall', line=dict(color='#66bb6a', width=2)))
                        fig.add_trace(go.Scatter(x=knn_df['k'], y=knn_df['f1'], mode='lines+markers',
                                                name='F1', line=dict(color='#ffa726', width=2)))
                        if st.session_state.best_k:
                            fig.add_vline(x=st.session_state.best_k, line_dash="dash", line_color="red")
                        fig.update_layout(title="k-NN Performance Metrics", height=300,
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
                        st.plotly_chart(fig, use_container_width=True, key="knn_full_curve")
                    except Exception:
                        st.warning("⚠️ Could not render k-NN curve.")
                
                if st.session_state.classification_results and 'show_comparison' in dir() and show_comparison:
                    try:
                        metrics_df = pd.DataFrame([{
                            'Classifier': name,
                            'Accuracy': res['accuracy'],
                            'Precision': res['precision'],
                            'Recall': res['recall'],
                            'F1': res['f1']
                        } for name, res in st.session_state.classification_results.items()])
                        
                        fig = go.Figure()
                        for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                            fig.add_trace(go.Bar(name=metric, x=metrics_df['Classifier'], y=metrics_df[metric]))
                        fig.update_layout(barmode='group', title="Classifier Comparison", height=300)
                        st.plotly_chart(fig, use_container_width=True, key="clf_comparison")
                    except Exception:
                        st.warning("⚠️ Could not render classifier comparison chart.")
            except Exception:
                st.error("❌ Could not display classification visualization.")
        
        elif viz_type == "Data" and st.session_state.raw_data is not None:
            try:
                df_viz = st.session_state.raw_data
                num_cols_viz = df_viz.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols_viz) >= 2:
                    color = None if color_col == "None" else color_col
                    try:
                        fig = px.scatter(df_viz, x=x_col, y=y_col, color=color, opacity=0.7,
                                       title=f"{x_col} vs {y_col}")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True, key="data_scatter")
                    except Exception:
                        st.warning("⚠️ Could not render scatter plot for selected columns.")
                    
                    # Correlation Matrix
                    if len(num_cols_viz) > 2:
                        try:
                            corr = df_viz[num_cols_viz].corr()
                            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                                           title="Correlation Matrix")
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True, key="corr_matrix")
                        except Exception:
                            st.warning("⚠️ Could not render correlation matrix.")
                else:
                    st.info("ℹ️ Need at least 2 numeric columns for scatter plot visualization.")
            except Exception:
                st.error("❌ Could not display data visualization.")
        
        else:
            st.markdown("""
            <div style="text-align:center; padding:40px; color:#546e7a;">
                <div style="font-size:2.5rem; margin-bottom:15px;">📊</div>
                <p>Run analysis first to see visualizations.</p>
            </div>
            """, unsafe_allow_html=True)

# ============ TAB 5: REPORT (WEKA STYLE) ============
with tab5:
    st.markdown("""<div class="panel-header">📋 Analysis Report</div>""", unsafe_allow_html=True)
    
    if st.session_state.raw_data is None:
        st.markdown('<div class="info-box">No analysis performed yet. Load data to begin.</div>', unsafe_allow_html=True)
    else:
        # Summary Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""<div class="panel-header">📂 Dataset Info</div>""", unsafe_allow_html=True)
            df = st.session_state.raw_data
            st.markdown(f"""
            <div class="data-info-row"><span class="data-label">Instances:</span><span class="data-value">{df.shape[0]:,}</span></div>
            <div class="data-info-row"><span class="data-label">Attributes:</span><span class="data-value">{df.shape[1]}</span></div>
            <div class="data-info-row"><span class="data-label">Numeric:</span><span class="data-value">{len(df.select_dtypes(include=np.number).columns)}</span></div>
            <div class="data-info-row"><span class="data-label">Nominal:</span><span class="data-value">{len(df.select_dtypes(exclude=np.number).columns)}</span></div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""<div class="panel-header">⚙️ Preprocessing</div>""", unsafe_allow_html=True)
            if st.session_state.clean_data is not None:
                st.markdown(f"""
                <div class="data-info-row"><span class="data-label">Status:</span><span class="data-value status-ok">✓ Applied</span></div>
                <div class="data-info-row"><span class="data-label">Features:</span><span class="data-value">{len(st.session_state.features_used)}</span></div>
                <div class="data-info-row"><span class="data-label">Outliers Removed:</span><span class="data-value">{st.session_state.outliers_count}</span></div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="data-info-row"><span class="data-label">Status:</span><span class="data-value status-warning">Pending</span></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("""<div class="panel-header">🎯 Classification</div>""", unsafe_allow_html=True)
            if st.session_state.classification_results:
                best_clf = max(st.session_state.classification_results.items(), key=lambda x: x[1]['f1'])
                st.markdown(f"""
                <div class="data-info-row"><span class="data-label">Models Trained:</span><span class="data-value">{len(st.session_state.classification_results)}</span></div>
                <div class="data-info-row"><span class="data-label">Best Model:</span><span class="data-value">{best_clf[0][:15]}</span></div>
                <div class="data-info-row"><span class="data-label">Best F1:</span><span class="data-value">{best_clf[1]['f1']:.4f}</span></div>
                """, unsafe_allow_html=True)
                if st.session_state.best_k:
                    st.markdown(f'<div class="data-info-row"><span class="data-label">Best k (k-NN):</span><span class="data-value">{st.session_state.best_k}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="data-info-row"><span class="data-label">Status:</span><span class="data-value status-warning">Not Run</span></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown("""<div class="panel-header">🔮 Clustering</div>""", unsafe_allow_html=True)
            if st.session_state.comparison_kpis:
                best = get_recommendation(st.session_state.comparison_kpis)
                st.markdown(f"""
                <div class="data-info-row"><span class="data-label">Algorithms:</span><span class="data-value">{len(st.session_state.comparison_kpis)}</span></div>
                <div class="data-info-row"><span class="data-label">Recommended:</span><span class="data-value">{best or 'N/A'}</span></div>
                """, unsafe_allow_html=True)
                if st.session_state.elbow_data:
                    st.markdown(f'<div class="data-info-row"><span class="data-label">Suggested k:</span><span class="data-value">{st.session_state.suggested_clusters}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="data-info-row"><span class="data-label">Status:</span><span class="data-value status-warning">Not Run</span></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Results
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.session_state.classification_results or st.session_state.knn_results:
                st.markdown("""<div class="panel-header">📊 Classification Results</div>""", unsafe_allow_html=True)
                
                all_clf_results = []
                
                # k-NN results
                if st.session_state.knn_results and st.session_state.best_k:
                    best_knn = next(r for r in st.session_state.knn_results if r['k'] == st.session_state.best_k)
                    all_clf_results.append({
                        'Classifier': f"k-NN (k={st.session_state.best_k})",
                        'Accuracy': best_knn['accuracy'],
                        'Precision': best_knn['precision'],
                        'Recall': best_knn['recall'],
                        'F1-Score': best_knn['f1']
                    })
                
                # Other classifiers
                for name, res in st.session_state.classification_results.items():
                    all_clf_results.append({
                        'Classifier': name,
                        'Accuracy': res['accuracy'],
                        'Precision': res['precision'],
                        'Recall': res['recall'],
                        'F1-Score': res['f1']
                    })
                
                if all_clf_results:
                    results_df = pd.DataFrame(all_clf_results)
                    results_df = results_df.round(4)
                    st.dataframe(results_df, use_container_width=True)
        
        with col_right:
            if st.session_state.comparison_kpis:
                st.markdown("""<div class="panel-header">📈 Clustering Results</div>""", unsafe_allow_html=True)
                
                cluster_df = pd.DataFrame([{
                    'Algorithm': algo,
                    'Clusters': kpi['clusters'],
                    'Silhouette': round(kpi['silhouette'], 4) if kpi['silhouette'] else None,
                    'Davies-Bouldin': round(kpi['davies-bouldin'], 4) if kpi['davies-bouldin'] else None,
                    'Calinski-H': round(kpi['calinski'], 1) if kpi['calinski'] else None
                } for algo, kpi in st.session_state.comparison_kpis.items()])
                st.dataframe(cluster_df, use_container_width=True)
        
        # Export Options
        st.markdown("---")
        st.markdown("""<div class="panel-header">💾 Export Results</div>""", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.classification_results or st.session_state.comparison_kpis:
                try:
                    # Prepare export data
                    export_data = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'dataset_shape': st.session_state.raw_data.shape if st.session_state.raw_data is not None else None
                    }
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        if st.session_state.classification_results:
                            clf_df = pd.DataFrame([{
                                'Classifier': name, **{k: v for k, v in res.items() if k != 'confusion_matrix' and k != 'model'}
                            } for name, res in st.session_state.classification_results.items()])
                            clf_df.to_excel(writer, sheet_name='Classification', index=False)
                        
                        if st.session_state.comparison_kpis:
                            cluster_df = pd.DataFrame([{
                                'Algorithm': algo, **kpi
                            } for algo, kpi in st.session_state.comparison_kpis.items()])
                            cluster_df.to_excel(writer, sheet_name='Clustering', index=False)
                    
                    st.download_button(
                        "📥 Download Excel Report",
                        buffer.getvalue(),
                        f"ml_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception:
                    st.warning("⚠️ Could not prepare Excel report. Some data may be incompatible.")
        
        with col2:
            if st.session_state.raw_data is not None:
                try:
                    csv = st.session_state.raw_data.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Data (CSV)", csv, "data_export.csv", "text/csv", use_container_width=True)
                except Exception:
                    st.warning("⚠️ Could not prepare CSV export.")
        
        with col3:
            if st.button("🔄 Reset Session", use_container_width=True):
                try:
                    for k in list(st.session_state.keys()):
                        st.session_state.pop(k)
                    st.rerun()
                except Exception:
                    st.error("❌ Could not reset session. Please refresh the page.")

# Status Bar
st.markdown(f"""
<div class="status-bar">
    <span>ML Explorer  </span>
    <span>Session: {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
</div>
""", unsafe_allow_html=True)
