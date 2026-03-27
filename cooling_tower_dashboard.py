import streamlit as st
import pandas as pd
import numpy as np
from report_generator import generate_pdf_report
import anthropic
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import shap


st.title('Water Intelligence Dashboard')
tab1, tab2 = st.tabs(["🌊 Cooling Tower", "🔥 Boiler Feedwater"])

uploaded_file = st.sidebar.file_uploader(
    "Upload plant data (CSV) \n Required Columns: date, TDS_circ, TDS_makeup, pH, temp_C, flow_rate, calcium_hardness, alkalinity", 
    type="csv"
)
st.sidebar.header('Plant Inputs')
plant_name = st.sidebar.text_input("Plant Name", value="My Cooling Tower Plant")
contact_name = st.sidebar.text_input("Contact Name", value="Plant Manager")
contact_email = st.sidebar.text_input("Contact Email", value="manager@plant.com")
#uploaded_file = st.sidebar.file_uploader("Upload plant data (CSV)", type="csv")


@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def LSI_status(lsi_value):
        if lsi_value > 0.5:
            return st.error("High Scaling Risk")
        elif lsi_value < -0.5:
            return st.warning("Corrosive conditions")
        else:
            return st.success("Balanced water")

def detect_anomalies(df, features):
    X = df[features]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    df = df.copy()
    df['anomaly'] = model.predict(X)
    df['anomaly_score'] = model.decision_function(X)
    return df, model, X

def get_LSI_label(lsi_value):
    if lsi_value > 0.5:
        return "High Scaling Risk"
    elif lsi_value < -0.5:
        return "Corrosive"
    else:
        return "Balanced"
    
def get_ai_recommendation(current_CoC, max_safe_CoC, 
                           current_makeup, optimised_makeup,
                           monthly_savings_m3, monthly_savings_Rs, LSI, dominant_cause):

    load_dotenv()

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a water intelligence assistant with 40 years of experience for an industrial cooling tower optimization system. Your only job is to explain what the engineering formulas have computed in plain English for a plant operator. You never invent numbers. You never make recommendations outside the safe operating envelope provided to you. Always mention the specific numbers you were given.",
        messages=[{
            "role": "user", "content": f"Current CoC: {current_CoC}. Max safe CoC: {max_safe_CoC}. Current makeup water: {current_makeup} m3/month. Optimised makeup water: {optimised_makeup} m3/month. Monthly savings: {monthly_savings_m3} m3. Money saved: Rs {monthly_savings_Rs}. LSI: {LSI} {get_LSI_label(LSI)}. Primary anomaly driver: {dominant_cause}. Explain this to the plant operator in 3-4 sentences."
        }]
    )
    return message.content[0].text


def get_shap_values(model, X, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=features)
    return shap_df


with tab1:
    st.info("""
📊 **Sample mode** — values shown are based on typical Indian industrial cooling tower parameters for display purposes. Upload your plant's CSV file using the sidebar for a realistic simulation.

**Required CSV columns:** `date, TDS_circ, TDS_makeup, pH, temp_C, flow_rate, calcium_hardness, alkalinity`
""")
    if uploaded_file is not None:
        df_file = load_data(uploaded_file)
        required_columns = ["date", "TDS_circ", "TDS_makeup", "pH", "temp_C", 
                     "flow_rate", "calcium_hardness", "alkalinity"]
        missing_cols = [col for col in required_columns if col not in df_file.columns]
        if missing_cols:
            st.error(f"❌ Uploaded CSV is missing required columns or has differently named columns. Missing columns are: {missing_cols}. Please check your file format.")
            st.stop()
        df_file = df_file.reset_index(drop=True)  # ensure clean integer index
        TDS_circ = df_file["TDS_circ"].mean()
        TDS_makeup = df_file["TDS_makeup"].mean()
        #water temp ranges from 32 to 37 degrees celsius, so the values are at avg. temp of 34.5 degree
        m_circ = df_file["flow_rate"].mean() #circulation rate of water (kg/hr)
        current_CoC = TDS_circ/TDS_makeup 
        pH = df_file["pH"].mean()
        calcium_hardness = df_file["calcium_hardness"].mean()
        alkalinity = df_file["alkalinity"].mean()
        temp_C = df_file["temp_C"].mean()

    else:
        report_start = st.sidebar.date_input("Report Start Date", value=pd.to_datetime("2024-01-01"))
        report_end = st.sidebar.date_input("Report End Date", value=pd.to_datetime("2024-01-31"))
        TDS_circ = st.sidebar.number_input("Circulating Water TDS (mg/L)", value=500)
        TDS_makeup = st.sidebar.number_input("Makeup water TDS (mg/L)", value=200)
        m_circ = st.sidebar.number_input("Circulation Rate (kg/hr)", value=3300) #circulation rate of water (kg/hr)
        current_CoC = st.sidebar.number_input("Current CoC", value=1.5, step=0.1)
        pH = st.sidebar.number_input("pH of circulating water", value=7.5, step=0.1)
        calcium_hardness = st.sidebar.number_input("Calcium Hardness (mg/L as CaCO3)", value=300)
        alkalinity = st.sidebar.number_input("Alkalinity (mg/L as CaCO3)", value=200)
        temp_C = st.sidebar.number_input("Water Temperature (°C)", value=34.5)


    # GENERATE CoC TREND OVER 30 DAYS

    st.subheader("CoC Trend — Last 30 Days")
    np.random.seed(42)
    hours = 720
    buildup_rate = 2  # TDS increases by 2 mg/L every hour

    if uploaded_file is not None:
        CoC_series = np.array(df_file["TDS_circ"]/df_file["TDS_makeup"])
        df_trend = pd.DataFrame({'CoC': CoC_series}, index=pd.to_datetime(df_file["date"]))
    else:
        TDS_circ_series = [TDS_makeup + (i % 120) * 2 for i in range(hours)]
        CoC_series = np.array(TDS_circ_series) / TDS_makeup
        df_trend = pd.DataFrame({'CoC': CoC_series},
            index=pd.date_range(start='2024-01-01', periods=hours, freq='h'))

    st.line_chart(df_trend['CoC'])

    #current_CoC = st.sidebar.number_input("Current CoC", value=3)

    #WATER METRICS

    #water temp ranges from 32 to 37 degrees celsius, so the values are at avg. temp of 34.5 degree
    T1 = st.sidebar.number_input("Minimum Temperature (°C)", value=32)
    T2 = st.sidebar.number_input("Maximum Temperature (°C)", value=37)
    delta_T = T2-T1
    T_avg = (T1+T2)/2  
    Cp = 4.18        # kJ/kg°C — specific heat of water
    Q = m_circ * Cp * delta_T  # scale heat duty with circulation rate because Q = m*Cp*delta T ; so Q directly prop to mass flow rate(aka m_circ)
    lambda_v = 2409.2 #calorific value at T_avg  (kJ/kg)
    E_prime = Q / lambda_v #Evapouration rate converted to kg/hr
    #LSI parameters for 
    A = (np.log10(TDS_circ) - 1)/10
    B = -13.12 * np.log10(temp_C + 273) + 34.55
    C = np.log10(calcium_hardness) - 0.4
    D = np.log10(alkalinity)
    pHs = 9.3 + A + B - C - D #Saturation pH
    LSI = pH - pHs
    #Max LSI safe limit is 0.5. so replacing that with 0.5 for max safe CoC aka optimised CoC value.
    #pH − 0.5 = 9.3 + A + B − C − D so A = pH − 0.5 − 9.3 − B + C + D
    #(log10(TDS_circ) − 1) / 10 = pH − 0.5 − 9.3 − B + C + D
    max_safe_TDS_circ = 10 ** (10 * (pH - 0.5 - 9.3 - B + C + D) + 1)
    max_safe_CoC = max_safe_TDS_circ / TDS_makeup
    if max_safe_CoC < 1.5:
        st.error(f"⚠️ Max safe CoC = {max_safe_CoC:.2f} — water chemistry inputs may be unrealistic or plant water is already severely scale-forming. Check calcium hardness, alkalinity, and pH values.")
    elif max_safe_CoC > 8.0:
        st.warning(f"⚠️ Max safe CoC = {max_safe_CoC:.2f} — unusually high. Verify water chemistry inputs.")
    else:
        st.metric("Max Safe CoC (LSI limit)", f"{max_safe_CoC:.2f}")
    W_prime = 0.003* m_circ #0.3% of circulation rate

    optimised_CoC = max_safe_CoC #our system's target based on LSI = 0.5 (max safe limit before scaling risk)
    current_B_prime = (E_prime + W_prime)/(current_CoC-1) #blowdown rate
    current_makeup = (E_prime + W_prime + current_B_prime)*720/1000 #kg/hr converted to m3/month
    optimised_B_prime = (E_prime + W_prime)/(optimised_CoC-1)
    optimised_makeup = (E_prime + W_prime + optimised_B_prime)*720/1000 #kg/hr converted to m3/month
    if current_CoC > max_safe_CoC:
        st.error(f"🚨 SCALING RISK — Current CoC ({current_CoC:.2f}) exceeds max safe limit ({max_safe_CoC:.2f}). Increase blowdown immediately.")
    else:
        #DISPLAYING WATER METRICS ON DASHBOARD
        st.metric("Avg Water Temperature (°C)", f"{T_avg:.1f}")
        st.metric("Current Cycles of Concentration(CoC)",f"{current_CoC:.2f}")
        st.metric("Optimised Cycles of Concentration(CoC)",f"{optimised_CoC:.2f}")
        st.metric("Current Blowdown (kg/hr)",f"{current_B_prime:.2f}")
        st.metric("Optimised Blowdown (kg/hr)",f"{optimised_B_prime:.2f}")
        st.metric("Current Makeup Water",f"{current_makeup:.2f}")
        st.metric("Optimised Makeup Water",f"{optimised_makeup:.2f}")


    #LSI - Langelier Saturation Index = to determine whether water is scale-forming or corrosive
    # LSI > 0 → scale-forming (calcium carbonate deposits on heat transfer surfaces)
    # LSI = 0 → balanced
    # LSI < 0 → corrosive (attacking metal surfaces)


    # LSI = pH - pHs

    st.subheader("Scaling Risk — Langelier Saturation Index")
    st.metric("LSI", f"{LSI:.2f}")

    LSI_status(LSI)

    st.subheader("Unusual Tower Patterns Over 720 Hours")

    #Anomaly detection:
    #using IsolationForest to detect anomalies
    if uploaded_file is not None:
        features = ['TDS_circ', 'pH', 'flow_rate']
        df_result, model, X = detect_anomalies(df_file, features)
        flagged = (df_result['anomaly'] == -1).sum()
        st.metric("Anomalous hours detected", flagged)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        normal = df_result[df_result['anomaly'] == 1]
        anomalous = df_result[df_result['anomaly'] == -1]
        ax.plot(normal.index, normal['TDS_circ'], 
                'b.', markersize=4, label='Normal', alpha=0.6)
        ax.scatter(anomalous.index, anomalous['TDS_circ'],
                color='red', s=60, label='Anomaly', zorder=5)
        ax.set_xlabel('Hour')
        ax.set_ylabel('TDS Circulating (mg/L)')
        ax.set_title('Isolation Forest Anomaly Detection')
        ax.legend()
        st.pyplot(fig)

        # Show SHAP values for anomalous hours only
        shap_df = get_shap_values(model, X, features)
        anomalous_shap = shap_df[df_result['anomaly'] == -1].copy()
        anomalous_shap['hour'] = df_result[df_result['anomaly'] == -1].index
        anomalous_shap['dominant_cause'] = anomalous_shap[features].abs().idxmax(axis=1)
        st.write("**What caused each anomaly:**")
        st.dataframe(anomalous_shap[['hour', 'dominant_cause'] + features].head(10))

    else: #in case a file is not uploaded. this is a sample anomaly create by Claude just for demo. Real plant will need real data 
        # Synthetic fault simulation for demo mode
        # Blowdown failur essentially means that CoC in circulating water is high. So high CoC than a certain limit = blowdown anomaly
    # The CoC limit depends on water chemistry of the plant and the LSI
    # In the dashboard, after uploading file:
        demo_CoC = max_safe_CoC
        E_series = E_prime * (1 + np.random.uniform(-0.10, 0.10, hours))
        W_series = W_prime * (1 + np.random.uniform(-0.05, 0.05, hours))
        B_series = (E_series + W_series) / (demo_CoC - 1)
        
        for i in range(500, 600):
            CoC_fault = demo_CoC + (i - 500) * 0.05
            B_series[i] = (E_series[i] + W_series[i]) / (CoC_fault - 1)
        
        df_fault = pd.DataFrame({'Blowdown': B_series}, index=pd.date_range(start='2024-01-01', periods=hours, freq='h'))
        blowdown_mean = B_series.mean()
        df_fault['fault'] = df_fault['Blowdown'] < 0.8 * blowdown_mean
        
        st.line_chart(df_fault['Blowdown'])
        flagged = df_fault['fault'].sum()
        st.metric("Hours flagged as anomalous", flagged)


    st.subheader("Savings Calculator")
    water_cost = st.sidebar.number_input("Water Cost (Rs/m3)", value=50)
    monthly_savings_m3 = current_makeup - optimised_makeup #m3/month
    monthly_savings_Rs = monthly_savings_m3 * water_cost #m3/month * Rs/m3
    my_fees = 0.2*monthly_savings_Rs  #20% of total savings per month

    #DISPLAYING MONTHLY SAVINGS IF NEEDED
    if monthly_savings_m3 < 0:
        st.warning("Current CoC is already above the safe limit. Recommend reducing CoC to prevent scaling damage.")
    else:
        st.metric("Monthly Savings of Water (m3)",f"{monthly_savings_m3:.2f}")
        st.metric("Monthly Savings (Rs.)",f"{monthly_savings_Rs:.2f}")
        st.metric("My Fees (Rs.)",f"{my_fees:.2f}")
        #DISPLAYING ANNUAL SAVINGS
        st.subheader("Annual Savings")
        annual_savings_Rs = monthly_savings_Rs * 12
        st.metric("Annual Savings (Rs.)", f"{annual_savings_Rs:,.0f}")
        st.metric("Annual Fee to You (Rs.)", f"{my_fees * 12:,.0f}")



    #Baseline Selector
    if uploaded_file is not None:
        baseline_start = pd.to_datetime(st.sidebar.date_input("Baseline start date", value=pd.to_datetime("2024-01-01")))
        baseline_end = pd.to_datetime(st.sidebar.date_input("Baseline end date", value=pd.to_datetime("2024-01-15")))
        #df_baseline = rows where date is within the selected baseline period
        df_baseline = df_file[(pd.to_datetime(df_file["date"]) >= baseline_start) & (pd.to_datetime(df_file["date"]) <= baseline_end)]
        baseline_CoC = df_baseline["TDS_circ"].mean()/df_baseline["TDS_makeup"].mean()
        baseline_m_circ = df_baseline["flow_rate"].mean()
        baseline_Q = baseline_m_circ * Cp * delta_T
        baseline_E_prime = baseline_Q/ lambda_v
        baseline_W_prime = 0.003 * baseline_m_circ
        baseline_B_prime = (baseline_E_prime+baseline_W_prime)/(baseline_CoC-1)
        baseline_M_prime = baseline_E_prime + baseline_W_prime + baseline_B_prime #baseline makeup water hourly
        baseline_daily_makeup = baseline_M_prime*24
        report_start = baseline_start.date()
        report_end = baseline_end.date()


        #df_optimised = rows where date is after the baseline period
        df_optimised = df_file[pd.to_datetime(df_file["date"]) > baseline_end]
        actual_post_baseline_CoC = df_optimised["TDS_circ"].mean()/df_optimised["TDS_makeup"].mean()
        optimised_m_circ = df_optimised["flow_rate"].mean()
        optimised_Q = optimised_m_circ * Cp * delta_T
        optimised_E_prime = optimised_Q/ lambda_v
        optimised_W_prime = 0.003 * optimised_m_circ
        optimised_B_prime = (optimised_E_prime+optimised_W_prime)/(actual_post_baseline_CoC-1)
        optimised_M_prime = optimised_E_prime + optimised_W_prime + optimised_B_prime #optimised makeup water hourly
        df_optimised["optimised makeup"] = optimised_M_prime

        daily = df_optimised.copy()
        daily["baseline makeup"] = baseline_M_prime
        daily["date"] = pd.to_datetime(df_optimised["date"])
        daily = daily.groupby(pd.to_datetime(df_optimised["date"]).dt.date).mean()
        daily["daily_optimised_makeup_m3"] = daily["optimised makeup"]/1000

        daily["daily_saving_m3"] = (daily["baseline makeup"] / 1000) - daily["daily_optimised_makeup_m3"]
        daily["cumulative_saving_m3"] = daily["daily_saving_m3"].cumsum()
        daily["cumulative_saving_Rs"] = daily["cumulative_saving_m3"] * water_cost

        st.subheader("Cumulative Savings vs Baseline")
        st.line_chart(daily["cumulative_saving_Rs"])
        st.metric("Total Verified Savings (m³)", f"{daily['cumulative_saving_m3'].iloc[-1]:.1f}")
        st.metric("Total Verified Savings (₹)", f"{daily['cumulative_saving_Rs'].iloc[-1]:,.0f}")
        

    # PDF DOWNLOAD
    st.subheader("Download Report")
    bs = str(baseline_start.date()) if uploaded_file is not None else "N/A"
    be = str(baseline_end.date()) if uploaded_file is not None else "N/A"
    dominant_cause = anomalous_shap['dominant_cause'].mode()[0] if len(anomalous_shap) > 0 and uploaded_file is not None else "No anomalies detected"

    pdf_bytes = generate_pdf_report(
        plant_name=plant_name,
        report_start=str(report_start),
        report_end=str(report_end),
        current_CoC=current_CoC,
        optimised_CoC=optimised_CoC,
        current_makeup=current_makeup,
        optimised_makeup=optimised_makeup,
        monthly_savings_m3=max(monthly_savings_m3, 0),
        monthly_savings_Rs=max(monthly_savings_Rs, 0),
        my_fees=max(my_fees, 0),
        avg_LSI=LSI,
        flagged_hours=int(flagged),
        baseline_start=bs,
        baseline_end=be,
        contact_name=contact_name,
        contact_email=contact_email,
        dominant_cause=dominant_cause
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"water_report_{report_start}_{report_end}.pdf",
        mime="application/pdf",
    )



if st.button("🤖 Get AI Recommendation"):
    with st.spinner("Analysing your water data..."):
        dominant_cause = anomalous_shap['dominant_cause'].mode()[0] if len(anomalous_shap) > 0 else "none"
        recommendation = get_ai_recommendation(
            current_CoC, max_safe_CoC, 
            current_makeup, optimised_makeup,
            monthly_savings_m3, monthly_savings_Rs, LSI, dominant_cause
        )
    st.subheader("AI Recommendation")
    st.markdown(recommendation)




with tab2:
    st.info("Boiler feedwater monitoring module — coming soon")