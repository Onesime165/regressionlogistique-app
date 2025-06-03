import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Econometric Logistic Regression",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark and Technological Theme ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    color: #e0e6ed !important;
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
}

.main .block-container {
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%) !important;
    color: #e0e6ed !important;
    padding-top: 2rem;
}

.stSidebar {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
    border-right: 2px solid #00ffff;
    box-shadow: 5px 0 15px rgba(0, 255, 255, 0.1);
}
.stSidebar .stFileUploader label,
.stSidebar .stSelectbox label,
.stSidebar .stSlider label,
.stSidebar .stButton button {
    color: #b0c4de !important;
    font-family: 'Roboto', sans-serif;
}
.stSidebar .stButton button {
    background: linear-gradient(45deg, #00ffff, #00ff88) !important;
    border: none !important;
    color: #0f0f23 !important;
    font-weight: bold;
    border-radius: 25px;
    padding: 10px 25px;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    width: 100%;
}
.stSidebar .stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 255, 255, 0.4);
    background: linear-gradient(45deg, #00ff88, #00ffff) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #00ffff;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}
h4 {
    margin: 20px 0 15px 0;
    font-size: 1.5em;
}

.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, #0f3460 0%, #16537e 100%);
    border-bottom: 2px solid #00ffff;
    border-radius: 10px 10px 0 0;
}
.stTabs [data-baseweb="tab"] {
    color: #b0c4de !important;
    background: transparent;
    border: none;
    transition: all 0.3s ease;
    font-family: 'Roboto', sans-serif;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0, 255, 255, 0.2) !important;
    color: #00ffff !important;
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.7);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #00ffff !important;
    color: #0f0f23 !important;
    font-weight: bold;
    border: none;
    text-shadow: none;
}

.stDataFrame {
    background: rgba(15, 15, 35, 0.8) !important;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid rgba(0, 255, 255, 0.3);
}
.stDataFrame table thead th {
    background: linear-gradient(90deg, #0f3460 0%, #16537e 100%) !important;
    color: #00ffff !important;
    border: 1px solid #00ffff !important;
    text-align: center;
    font-weight: bold;
}
.stDataFrame table tbody td {
    background: rgba(15, 15, 35, 0.6) !important;
    color: #e0e6ed !important;
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
}
.stDataFrame table tbody tr:hover td {
    background: rgba(0, 255, 255, 0.1) !important;
}

.stCodeBlock, pre {
    background: rgba(0, 0, 0, 0.8) !important;
    color: #00ff88 !important;
    border: 1px solid #00ffff;
    border-radius: 8px;
    padding: 15px;
    font-family: 'Courier New', monospace;
    box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.1);
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(15, 15, 35, 0.9) !important;
    border: 1px solid #00ffff !important;
    color: #e0e6ed !important;
    border-radius: 5px;
    transition: all 0.3s ease;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #00ff88 !important;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.5) !important;
    background: rgba(15, 15, 35, 1) !important;
}

.author-info {
    background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
    border: 2px solid #00ffff;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
}
.author-info h4 {
    text-align: center;
    margin-bottom: 15px;
    font-size: 18px;
    color: #00ffff;
    font-family: 'Orbitron', monospace;
}
.author-info p {
    margin: 8px 0;
    margin-left: 15px;
    font-weight: 400;
    color: #b0c4de;
}
.author-info a {
    color: #00ff88 !important;
    text-decoration: none;
    transition: all 0.3s ease;
}
.author-info a:hover {
    color: #00ffff !important;
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.7);
}
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
    margin: 20px 0;
}

.plotly-graph-div {
    background: transparent !important;
}

.metric-card {
    background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
    border: 1px solid #00ffff;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# --- Global Plotly Dark Theme for consistency ---
plotly_dark_template = go.layout.Template()
plotly_dark_template.layout.plot_bgcolor = '#1a1a2e'
plotly_dark_template.layout.paper_bgcolor = '#0c0c0c'
plotly_dark_template.layout.font.color = '#e0e6ed'
plotly_dark_template.layout.xaxis.gridcolor = '#4a5568'
plotly_dark_template.layout.yaxis.gridcolor = '#4a5568'
plotly_dark_template.layout.xaxis.linecolor = '#e0e6ed'
plotly_dark_template.layout.yaxis.linecolor = '#e0e6ed'
plotly_dark_template.layout.title.font.color = '#00ffff'
plotly_dark_template.layout.xaxis.title.font.color = '#00ffff'
plotly_dark_template.layout.yaxis.title.font.color = '#00ffff'
plotly_dark_template.layout.legend.bgcolor = 'rgba(15,15,35,0.8)'
plotly_dark_template.layout.legend.bordercolor = '#00ffff'

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext == "csv":
                df = pd.read_csv(uploaded_file)
            elif ext in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return None
            for col in df.select_dtypes(include=['bool', 'object']).columns:
                if df[col].nunique() == 2:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        pass
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def get_lrt_tests(df, target_var, explanatory_vars_list, full_model_llf, full_model_df_model):
    lrt_results = []
    if not explanatory_vars_list:
        return pd.DataFrame(columns=['Variable', 'Statistic', 'p-value'])

    for var_to_remove in explanatory_vars_list:
        reduced_vars_list = [v for v in explanatory_vars_list if v != var_to_remove]
        
        if not reduced_vars_list:
            reduced_formula_str = f"{target_var} ~ 1"
        else:
            reduced_formula_parts = []
            for var in reduced_vars_list:
                if df[var].dtype == 'object' or df[var].dtype.name == 'category' or \
                   (df[var].nunique() < 10 and pd.api.types.is_integer_dtype(df[var].dtype)):
                    reduced_formula_parts.append(f"C({var})")
                else:
                    reduced_formula_parts.append(var)
            reduced_formula_str = f"{target_var} ~ {' + '.join(reduced_formula_parts)}"
        
        try:
            reduced_model = smf.logit(formula=reduced_formula_str, data=df).fit(disp=0)
            lr_statistic = 2 * (full_model_llf - reduced_model.llf)
            df_diff = full_model_df_model - reduced_model.df_model
            if df_diff <= 0:
                p_value = np.nan
            else:
                p_value = stats.chi2.sf(lr_statistic, df_diff)
            
            lrt_results.append({
                'Variable': var_to_remove,
                'Statistic': lr_statistic,
                'p-value': p_value
            })
        except Exception as e:
            st.warning(f"Could not fit reduced model or run LRT for variable '{var_to_remove}': {e}")
            lrt_results.append({
                'Variable': var_to_remove,
                'Statistic': np.nan,
                'p-value': np.nan
            })
            
    return pd.DataFrame(lrt_results)

def create_prediction_input_form(data, explanatory_vars):
    """Create input form for individual predictions"""
    input_values = {}
    
    col1, col2 = st.columns(2)
    
    for i, var in enumerate(explanatory_vars):
        with col1 if i % 2 == 0 else col2:
            if data[var].dtype == 'object' or data[var].dtype.name == 'category':
                unique_vals = data[var].unique()
                input_values[var] = st.selectbox(f"{var}:", unique_vals, key=f"pred_{var}")
            elif data[var].nunique() < 10 and pd.api.types.is_integer_dtype(data[var].dtype):
                unique_vals = sorted(data[var].unique())
                input_values[var] = st.selectbox(f"{var}:", unique_vals, key=f"pred_{var}")
            else:
                min_val = float(data[var].min())
                max_val = float(data[var].max())
                mean_val = float(data[var].mean())
                input_values[var] = st.number_input(
                    f"{var}:", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=mean_val,
                    key=f"pred_{var}"
                )
    
    return input_values

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx", "xls"])

data = load_data(uploaded_file)
target_var = None
explanatory_vars = []
sig_vars_original = []

if data is not None:
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    potential_target_cols = [col for col in data.columns if data[col].nunique() == 2 and set(data[col].dropna().unique()).issubset({0,1,True,False})]
    
    if not potential_target_cols:
        st.sidebar.warning("No binary (0/1) columns found for target. Showing all columns as potential targets.")
        potential_target_cols = data.columns.tolist()
    
    if potential_target_cols:
         target_var = st.sidebar.selectbox("Target Variable (binary, 0/1)", potential_target_cols, index=0 if potential_target_cols else None, key="target_var_select")
    else:
        st.sidebar.error("No columns found in the uploaded data.")

    if target_var:
        available_explanatory = [col for col in data.columns if col != target_var]
        explanatory_vars = st.sidebar.multiselect("Explanatory Variables", available_explanatory, default = available_explanatory[:min(len(available_explanatory), 5)], key="explanatory_vars_multiselect")

threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01, key="threshold_slider")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="author-info">
    <h4>🧾 About the Author</h4>
    <p>Name: N'dri</p>
    <p>First Name: Abo Onesime</p>
    <p>Role: Data Analyst / Scientist</p>
    <p>Phone: 07-68-05-98-87 / 01-01-75-11-81</p>
    <p>Email: <a href="mailto:ndriablatie123@gmail.com">ndriablatie123@gmail.com</a></p>
    <p>LinkedIn: <a href="https://www.linkedin.com/in/abo-onesime-n-dri-54a537200/" target="_blank">LinkedIn Profile</a></p>
    <p>GitHub: <a href="https://github.com/Aboonesime" target="_blank">My GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# --- Main Panel with Tabs ---
st.title("📊 Econometric Logistic Regression Dashboard")

if data is not None and target_var and explanatory_vars:
    if not (data[target_var].nunique() == 2 and set(data[target_var].dropna().unique()).issubset({0, 1, True, False})):
        st.error(f"Target variable '{target_var}' must be binary (0/1). Please select an appropriate column or preprocess your data.")
        st.stop()
    
    _data_processed = data.copy()
    _data_processed[target_var] = _data_processed[target_var].astype(int)

    cols_to_check = [target_var] + explanatory_vars
    nan_rows_before = _data_processed.shape[0]
    _data_processed = _data_processed.dropna(subset=cols_to_check)
    nan_rows_removed = nan_rows_before - _data_processed.shape[0]

    if nan_rows_removed > 0:
        st.warning(f"Removed {nan_rows_removed} rows with missing values in selected columns.")
    if _data_processed.empty:
        st.error("No data remaining after handling missing values in key columns. Cannot proceed.")
        st.stop()

    formula_parts = []
    for var in explanatory_vars:
        if _data_processed[var].dtype == 'object' or _data_processed[var].dtype.name == 'category' or \
           (_data_processed[var].nunique() < 10 and pd.api.types.is_integer_dtype(_data_processed[var].dtype)):
            formula_parts.append(f"C({var})")
        else:
            formula_parts.append(var)
    formula_str = f"{target_var} ~ {' + '.join(formula_parts)}"

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Data Preview", "📉 Initial Model", "📊 Final Model",
        "📈 Marginal Effects", "🎯 Predictions", "👤 Individual Prediction"
    ])

    with tab1:
        st.subheader("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>Total Rows</h3><h2>{}</h2></div>'.format(len(data)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>Total Columns</h3><h2>{}</h2></div>'.format(len(data.columns)), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>Target Variable</h3><h2>{}</h2></div>'.format(target_var), unsafe_allow_html=True)
        with col4:
            target_dist = _data_processed[target_var].value_counts()
            st.markdown('<div class="metric-card"><h3>Target Distribution</h3><h2>{}:{}</h2></div>'.format(target_dist.index[0], target_dist.iloc[0]), unsafe_allow_html=True)
        
        st.subheader("Data Preview (First 100 rows)")
        st.dataframe(data.head(100))
        
        # Target variable distribution
        st.subheader("Target Variable Distribution")
        fig_target = px.histogram(
            _data_processed, 
            x=target_var, 
            title=f"Distribution of {target_var}",
            template="plotly_dark",
            color_discrete_sequence=['#00ffff']
        )
        fig_target.update_layout(template=plotly_dark_template)
        st.plotly_chart(fig_target, use_container_width=True)

    initial_model = None
    lrt_df = pd.DataFrame()
    sig_vars_original = []

    try:
        initial_model = smf.logit(formula=formula_str, data=_data_processed).fit(disp=0)
        
        with tab2:
            st.subheader("Initial Model Summary")
            st.code(initial_model.summary().as_text())

            st.markdown("<h4>🧪 Likelihood Ratio Tests (vs Full Model)</h4>", unsafe_allow_html=True)
            lrt_df = get_lrt_tests(_data_processed, target_var, explanatory_vars, initial_model.llf, initial_model.df_model)
            st.dataframe(lrt_df.style.format({'Statistic': '{:.3f}', 'p-value': '{:.4f}'}))
            
            if not lrt_df.empty:
                sig_vars_original = lrt_df[lrt_df['p-value'] < 0.05]['Variable'].tolist()

    except Exception as e:
        st.error(f"Error processing initial model or LRT: {e}")
        if initial_model is None:
             st.error("Initial model fitting failed. Cannot proceed with further analysis.")

    final_model = None
    if initial_model:
        if sig_vars_original:
            final_formula_parts = []
            for var in sig_vars_original:
                if _data_processed[var].dtype == 'object' or _data_processed[var].dtype.name == 'category' or \
                   (_data_processed[var].nunique() < 10 and pd.api.types.is_integer_dtype(_data_processed[var].dtype)):
                    final_formula_parts.append(f"C({var})")
                else:
                    final_formula_parts.append(var)
            
            if final_formula_parts:
                final_formula_str = f"{target_var} ~ {' + '.join(final_formula_parts)}"
                try:
                    final_model = smf.logit(formula=final_formula_str, data=_data_processed).fit(disp=0)
                except Exception as e:
                    st.error(f"Error fitting final model with significant variables: {e}. Falling back to initial model.")
                    final_model = initial_model
                    sig_vars_original = explanatory_vars
            else:
                st.warning("No variables were found to be significant by LRT to build a different final model. Using the initial model as the final model.")
                final_model = initial_model
                sig_vars_original = explanatory_vars
        else:
            st.warning("No variables were found to be significant by LRT. Using the initial model as the final model.")
            final_model = initial_model
            sig_vars_original = explanatory_vars

    if final_model:
        with tab3:
            st.subheader("Final Model Summary")
            st.code(final_model.summary().as_text())

            st.markdown("<h4>📊 Odds Ratios</h4>", unsafe_allow_html=True)
            try:
                odds_ratios = pd.DataFrame({
                    "Odds Ratio": np.exp(final_model.params),
                    "p-value": final_model.pvalues,
                    "Conf. Low": np.exp(final_model.conf_int().iloc[:, 0]),
                    "Conf. High": np.exp(final_model.conf_int().iloc[:, 1])
                })
                odds_ratios_display = odds_ratios[odds_ratios.index != "Intercept"].copy()
                odds_ratios_display.index.name = "Variable"
                odds_ratios_display = odds_ratios_display.reset_index()
                
                st.dataframe(odds_ratios_display.style.format({
                    'Odds Ratio': '{:.3f}', 'p-value': '{:.3f}',
                    'Conf. Low': '{:.3f}', 'Conf. High': '{:.3f}'
                }))

                st.markdown("<h4>📈 Odds Ratios Plot</h4>", unsafe_allow_html=True)
                if not odds_ratios_display.empty:
                    fig_odds_custom = go.Figure()
                    fig_odds_custom.add_trace(go.Bar(
                        y=odds_ratios_display['Variable'],
                        x=odds_ratios_display['Odds Ratio'],
                        orientation='h',
                        name='Odds Ratio',
                        marker_color='#00ffff',
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=odds_ratios_display['Conf. High'] - odds_ratios_display['Odds Ratio'],
                            arrayminus=odds_ratios_display['Odds Ratio'] - odds_ratios_display['Conf. Low'],
                            color='#00ff88',
                            thickness=1.5,
                            width=5
                        )
                    ))
                    fig_odds_custom.add_vline(x=1, line_dash="dash", line_color="#ff6b6b", line_width=2)
                    fig_odds_custom.update_layout(
                        title="Odds Ratios with 95% Confidence Intervals",
                        xaxis_title="Odds Ratio",
                        yaxis_title="Variable",
                        template=plotly_dark_template,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_odds_custom, use_container_width=True)
                else:
                    st.info("No variables in the final model to plot odds ratios (e.g. intercept-only model).")
            except Exception as e:
                st.error(f"Could not generate odds ratios: {e}")

        with tab4:
            st.subheader("Marginal Effects")
            try:
                mfx = final_model.get_margeff(at='overall', method='dydx')
                mfx_summary_frame = mfx.summary_frame()
                
                st.code(mfx.summary().as_text())

                st.markdown("<h4>📈 Marginal Effects Plot</h4>", unsafe_allow_html=True)
                
                mfx_df = mfx_summary_frame.reset_index()
                mfx_df = mfx_df.rename(columns={'index': 'Variable'})
                
                # Find appropriate column names
                effect_col = 'dy/dx' if 'dy/dx' in mfx_df.columns else mfx_df.columns[1]
                
                if not mfx_df.empty and 'Variable' in mfx_df.columns:
                    fig_mfx = go.Figure()
                    
                    # Check for confidence intervals
                    conf_cols = [col for col in mfx_df.columns if '[' in col or 'Conf' in col]
                    
                    if len(conf_cols) >= 2:
                        # With confidence intervals
                        conf_low_col = conf_cols[0]
                        conf_high_col = conf_cols[1]
                        
                        fig_mfx.add_trace(go.Bar(
                            y=mfx_df['Variable'],
                            x=mfx_df[effect_col],
                            orientation='h',
                            name='Marginal Effect (dy/dx)',
                            marker_color='#00ffff',
                            error_x=dict(
                                type='data',
                                symmetric=False,
                                array=mfx_df[conf_high_col] - mfx_df[effect_col],
                                arrayminus=mfx_df[effect_col] - mfx_df[conf_low_col],
                                color='#00ff88',
                                thickness=1.5,
                                width=5
                            )
                        ))
                    else:
                        # Without confidence intervals
                        fig_mfx.add_trace(go.Bar(
                            y=mfx_df['Variable'],
                            x=mfx_df[effect_col],
                            orientation='h',
                            name='Marginal Effect (dy/dx)',
                            marker_color='#00ffff'
                        ))
                    
                    fig_mfx.add_vline(x=0, line_dash="dash", line_color="#ff6b6b", line_width=2)
                    fig_mfx.update_layout(
                        title="Average Marginal Effects (AME)",
                        xaxis_title="Marginal Effect (Change in Pr(Y=1))",
                        yaxis_title="Variable",
                        template=plotly_dark_template,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_mfx, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error calculating marginal effects: {e}")

        with tab5:
            st.subheader("Model Predictions & Evaluation")
            try:
                predicted_prob = final_model.predict(_data_processed)
                predicted_class = (predicted_prob >= threshold).astype(int)
                actual_values = _data_processed[target_var]

                # Model performance metrics
                col1, col2, col3, col4 = st.columns(4)
                accuracy = (predicted_class == actual_values).mean()
                precision = (predicted_class[predicted_class == 1] == actual_values[predicted_class == 1]).mean() if (predicted_class == 1).sum() > 0 else 0
                recall = (predicted_class[actual_values == 1] == actual_values[actual_values == 1]).mean() if (actual_values == 1).sum() > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><h2>{accuracy:.2%}</h2></div>'.format(accuracy), unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>Precision</h3><h2>{precision:.2%}</h2></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h3>Recall</h3><h2>{recall:.2%}</h2></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="metric-card"><h3>F1 Score</h3><h2>{f1:.2%}</h2></div>', unsafe_allow_html=True)

                # Confusion matrix
                cm = confusion_matrix(actual_values, predicted_class)
                st.markdown("<h4>🧮 Confusion Matrix</h4>", unsafe_allow_html=True)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                 x=['Predicted 0', 'Predicted 1'],
                                 y=['Actual 0', 'Actual 1'])
                fig_cm.update_layout(template=plotly_dark_template)
                st.plotly_chart(fig_cm, use_container_width=True)

                # ROC Curve
                fpr, tpr, _ = roc_curve(actual_values, predicted_prob)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, 
                                         name=f'ROC Curve (AUC = {roc_auc:.2f})',
                                         mode='lines',
                                         line=dict(color='#00ffff', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                         mode='lines',
                                         line=dict(color='#ff6b6b', width=2, dash='dash'),
                                         showlegend=False))
                fig_roc.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template=plotly_dark_template,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )
                st.plotly_chart(fig_roc, use_container_width=True)

                # Download predictions
                download_df = pd.DataFrame({
                    'Actual': actual_values,
                    'Predicted_Probability': predicted_prob,
                    'Predicted_Class': predicted_class
                })
                csv = download_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="💾 Download Predictions",
                    data=csv,
                    file_name='logistic_regression_predictions.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"Error in prediction and evaluation: {e}")
        
        with tab6:
            st.subheader("🔮 Individual Prediction")
            try:
                input_values = create_prediction_input_form(_data_processed, sig_vars_original)
                if st.button("Make Prediction"):
                    input_df = pd.DataFrame([input_values])
                    for col in input_df.columns:
                        if _data_processed[col].dtype != 'object' and not (_data_processed[col].nunique() < 10 and pd.api.types.is_integer_dtype(_data_processed[col].dtype)):
                            input_df[col] = pd.to_numeric(input_df[col])
                    
                    # Ensure correct categorical encoding
                    for var in sig_vars_original:
                        if _data_processed[var].dtype == 'object' or _data_processed[var].dtype.name == 'category':
                            input_df[var] = input_df[var].astype('category')
                    
                    prediction = final_model.predict(input_df)[0]
                    predicted_class = 1 if prediction >= threshold else 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Predicted Probability</h3>
                        <h2>{prediction:.2%}</h2>
                        <p>For threshold of {threshold:.2f}: {'Class 1' if predicted_class else 'Class 0'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Feature importance plot for individual prediction
                    params = final_model.params
                    feature_importance = {}
                    for var in sig_vars_original:
                        value = input_df[var].iloc[0]
                        if var in params.index:
                            coef = params[var]
                            if _data_processed[var].dtype == 'object' or _data_processed[var].dtype.name == 'category':
                                # For categorical variables, use the coefficient directly
                                feature_importance[var] = coef * (value == _data_processed[var].unique())
                            else:
                                # For numerical variables, multiply by value
                                feature_importance[var] = coef * value
                    
                    importance_df = pd.DataFrame({
                        'Variable': list(feature_importance.keys()),
                        'Importance': [feature_importance[k] for k in feature_importance]
                    }).sort_values(by='Importance', key=abs, ascending=True)
                    
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(
                        y=importance_df['Variable'],
                        x=importance_df['Importance'],
                        orientation='h',
                        marker_color=importance_df['Importance'].apply(lambda x: '#00ff88' if x > 0 else '#ff6b6b'),
                    ))
                    fig_imp.add_vline(x=0, line_dash="dash", line_color="#e0e6ed", line_width=1)
                    fig_imp.update_layout(
                        title="Feature Contribution to Prediction",
                        xaxis_title="Contribution to Log-Odds",
                        yaxis_title="Variable",
                        template=plotly_dark_template
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please ensure all inputs are correctly set.")
else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>📂 No Data Uploaded</h2>
        <p>Please upload a CSV or Excel file using the sidebar on the left</p>
        <p>Requirements:</p>
        <ul style="list-style: none; padding-left: 0; display: inline-block; text-align: left;">
            <li>• Must contain at least one binary target variable (0/1)</li>
            <li>• Can include numeric and categorical explanatory variables</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b0c4de; padding: 20px;">
    <p>© 2023 Econometric Logistic Regression Dashboard | Created with ❤️ using Streamlit & Plotly</p>
    <p>Made by Abo Onesime N'dri - Data Analyst / Scientist</p>
</div>
""", unsafe_allow_html=True)