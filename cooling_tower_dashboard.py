import streamlit as st
import pandas as pd
import numpy as np

st.title('Cooling Tower Water Intelligence Dashboard')
st.sidebar.header('Plant Inputs')
uploaded_file = st.sidebar.file_uploader("Upload plant data (CSV)", type="csv")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df_file = load_data(uploaded_file)
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
    #st.dialog("File format not valid or file not uploaded. Please upload a .csv file")
    TDS_circ = st.sidebar.number_input("Circulating Water TDS (mg/L)", value=1000)
    TDS_makeup = st.sidebar.number_input("Makeup water TDS (mg/L)", value=200)
    m_circ = st.sidebar.number_input("Circulation Rate (kg/hr)", value=3300) #circulation rate of water (kg/hr)
    current_CoC = st.sidebar.number_input("Current CoC", value=3)
    pH = st.sidebar.number_input("pH of circulating water", value=7.5, step=0.1)
    calcium_hardness = st.sidebar.number_input("Calcium Hardness (mg/L as CaCO3)", value=200)
    alkalinity = st.sidebar.number_input("Alkalinity (mg/L as CaCO3)", value=150)
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
W_prime = 0.003* m_circ #0.3% of circulation rate

optimised_CoC = max_safe_CoC #our system's target based on LSI = 0.5 (max safe limit before scaling risk)
current_B_prime = (E_prime + W_prime)/(current_CoC-1) #blowdown rate
current_makeup = (E_prime + W_prime + current_B_prime)*720/1000 #kg/hr converted to m3/month
optimised_B_prime = (E_prime + W_prime)/(optimised_CoC-1)
optimised_makeup = (E_prime + W_prime + optimised_B_prime)*720/1000 #kg/hr converted to m3/month
st.write("pH:", pH)
st.write("B:", B)
st.write("C:", C)
st.write("D:", D)
st.write("max_safe_TDS_circ:", max_safe_TDS_circ)
st.write("TDS_makeup:", TDS_makeup)
st.write("max_safe_CoC:", max_safe_CoC)

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
if LSI > 0.5:
    st.error("High Scaling risk")
elif -0.5 <= LSI <= 0.5:
    st.success("Balanced water")   #ideally it supposed be in range 0 to 0.5, but operationally -0.5 to 0.5 is acceptable
elif LSI< -0.5:
    st.warning("Corrosive conditions")

st.subheader("Fault Detection — Blowdown Anomaly")

#Anomaly detection:
# Blowdown failur essentially means that CoC in circulating water is high. So high CoC than a certain limit = blowdown anomaly
# The CoC limit depends on water chemistry of the plant and the LSI
if uploaded_file is not None:
    # Real anomaly detection on uploaded data
    CoC_real = np.array(df_file["TDS_circ"] / df_file["TDS_makeup"])
    fault_index = pd.to_datetime(df_file["date"])
    
    # Flag hours where CoC exceeds 2.5 — scaling risk threshold for this plant
    # (threshold should eventually be set based on LSI, not hardcoded)
    CoC_threshold = max_safe_CoC
    faults = CoC_real > CoC_threshold
    df_fault = pd.DataFrame({
        'CoC': CoC_real,
        'fault': faults}, index=fault_index)
    df_fault['threshold'] = CoC_threshold
    st.line_chart(df_fault[['CoC', 'threshold']])
    flagged = np.array(df_fault['fault']).sum()
    st.metric("Hours above scaling threshold", flagged)

else: #in case a file is not uploaded. this is a sample anomaly create by Claude just for demo. Real plant will need real data 
    # Synthetic fault simulation for demo mode
    demo_CoC = 3.0
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

#DISPLAYING MONTHLY SAVINGS

st.metric("Monthly Savings of Water (m3)",f"{monthly_savings_m3:.2f}")
st.metric("Monthly Savings (Rs.)",f"{monthly_savings_Rs:.2f}")
st.metric("My Fees (Rs.)",f"{my_fees:.2f}")



#Baseline Selector
if uploaded_file is not None:
    baseline_start = pd.to_datetime(st.sidebar.date_input("Baseline start date", value=pd.to_datetime("2024-01-01")))
    baseline_end = pd.to_datetime(st.sidebar.date_input("Baseline end date", value=pd.to_datetime("2024-01-30")))
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
    
#DISPLAYING ANNUAL SAVINGS
st.subheader("Annual Savings")
annual_savings_Rs = monthly_savings_Rs * 12
st.metric("Annual Savings (Rs.)", f"{annual_savings_Rs:,.0f}")
st.metric("Annual Fee to You (Rs.)", f"{my_fees * 12:,.0f}")

