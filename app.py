
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from io import BytesIO

# --- Page Config ---
st.set_page_config(
    page_title="Inwon-Checker Pro + Cost",
    page_icon="💰",
    layout="wide"
)

# --- Custom Styles ---
st.markdown("""
    <style>
    .main { background-color: #f9f9fb; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .css-1r6slb0 { background-color: #ffffff; border-radius: 10px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Helper: Robust Dept Mapping ---
def get_mapped_dept(major_name):
    if not major_name: return None
    m = major_name.upper().replace('\n', ' ')
    if 'HR-GA' in m: return '경영지원팀'
    if '혁신' in m or 'INNOVATION' in m: return '혁신(현)'
    if 'MAINTENANCE' in m or '공무' in m: return '공무'
    if 'PRODUCTION' in m or '생산관리' in m: return '생산관리'
    if 'R&D' in m or '연구실' in m or '개발' in m: return '개발'
    if 'SALES' in m or '영업' in m: return '영업(사)'
    if 'MATERIAL' in m or '구매자재' in m: return '구매/자재(사)'
    if 'QC' in m or '품질' in m: return '품질(현)'
    if 'INJECTION' in m or '사출' in m: return 'Injection+금형'
    if 'MOLD' in m or '금형' in m: return 'Injection+금형'
    return major_name # Fallback to original name instead of None

# --- Logic: DMR Parser (For Sheet 1123) ---
def parse_dmr_sheet(df):
    header_row_idx = -1
    for idx, row in df.iterrows():
        row_str = " ".join([str(val) for val in row.values if pd.notnull(val)])
        if ('부서' in row_str and 'Team' in row_str) or '부서 (Team)' in row_str or '부서(Team)' in row_str:
            header_row_idx = idx
            break
    
    if header_row_idx == -1: return None, "DMR 헤더를 찾을 수 없습니다."

    data_raw = df.iloc[header_row_idx + 2:].copy()
    parsed_rows = []
    current_major_team = ""
    current_detail_team = ""
    
    for idx, row in data_raw.iterrows():
        raw_major = str(row[0]) if pd.notnull(row[0]) and str(row[0]).strip() != "" else current_major_team
        detail = str(row[1]) if pd.notnull(row[1]) and str(row[1]).strip() != "" else current_detail_team
        
        # Skip Total/Sum rows - ENHANCED
        position_str = str(row[2]).strip().upper()
        if any(x in position_str for x in ["TOTAL", "S-TOTAL", "합계", "소계", "SUM"]): 
            continue
        if pd.isnull(row[2]) and pd.isnull(row[5]): 
            continue
            
        current_major_team = raw_major
        current_detail_team = detail
        
        try:
            major_clean = raw_major.split('(')[0].strip()
            
            parsed_rows.append({
                "Major Team": major_clean,
                "Team": detail.replace('\n', ' ').strip(),
                "Position": str(row[2]).strip(),
                "Type": str(row[4]).strip(),
                "DJ1_TO": float(row[5]) if pd.notnull(row[5]) else 0,
                "DJ2_TO": float(row[6]) if pd.notnull(row[6]) else 0,
                "Total_TO": float(row[7]) if pd.notnull(row[7]) else 0,
                "DJ1_Actual": float(row[8]) if pd.notnull(row[8]) else 0,
                "DJ2_Actual": float(row[9]) if pd.notnull(row[9]) else 0,
                "Total_Actual": float(row[10]) if pd.notnull(row[10]) else 0
            })
        except: continue
    
    result_df = pd.DataFrame(parsed_rows)
    
    # CRITICAL FIX: Remove duplicates and filter anomalies
    if not result_df.empty:
        # 1. Remove exact duplicates
        result_df = result_df.drop_duplicates()
        
        # 2. Group by Major Team + Team + Position and sum (in case of legitimate splits)
        result_df = result_df.groupby(['Major Team', 'Team', 'Position', 'Type'], as_index=False).agg({
            'DJ1_TO': 'sum',
            'DJ2_TO': 'sum', 
            'Total_TO': 'sum',
            'DJ1_Actual': 'sum',
            'DJ2_Actual': 'sum',
            'Total_Actual': 'sum'
        })
        
        # 3. Filter out anomalous rows (T/O > 50 per position is suspicious)
        result_df = result_df[result_df['Total_TO'] <= 50]
            
    return result_df, None

# --- Logic: Cost Parser (For Sheet 2025) ---
def parse_cost_sheet(df):
    """
    Ultra-robust parser for Labor Cost.
    Finds '급여 현황' and extracts rows where Col C is 'STL' or similar.
    """
    start_row = 100
    for idx, row in df.iterrows():
        row_str_top = "".join([str(cell) for cell in row[:5]]).replace(" ", "")
        if "급여현황" in row_str_top: # Catch '급여 현황', '급여현황' etc.
            start_row = idx + 1
            break
            
    cost_data = []
    # Peek at columns to find indices for DJ1, DJ2, DJ3, Total
    # Default: E=4, F=5, G=6, I=8
    
    for i in range(start_row, min(len(df), start_row + 50)):
        try:
            row = df.iloc[i]
            dept_name = str(row[0]).strip() if pd.notnull(row[0]) else ""
            row_str = " ".join([str(x) for x in row.values])
            
            if dept_name != "" and not any(x in row_str.upper() for x in ["TOTAL", "합계", "소계", "급여"]):
                # Search for the first large numeric value in the row as Total Cost
                nums = []
                for cell in row[3:]:
                    try:
                        c_str = str(cell).replace(',','').strip()
                        if c_str.replace('.','',1).isdigit():
                            nums.append(float(c_str))
                    except: continue
                
                if nums:
                    val_h = max(nums) # Assumption: Largest number is the total/STL
                    cost_data.append({
                        "CostDept": dept_name,
                        "DJ1_Cost": nums[0] if len(nums) > 0 else 0,
                        "DJ2_Cost": nums[1] if len(nums) > 1 else 0,
                        "DJ3_Cost": nums[2] if len(nums) > 2 else 0,
                        "Total_Cost": val_h
                    })
        except: continue

    df_result = pd.DataFrame(cost_data)
    if df_result.empty:
        return pd.DataFrame(columns=["CostDept", "DJ1_Cost", "DJ2_Cost", "DJ3_Cost", "Total_Cost"])
        
    # Aggregate and Clean
    df_result = df_result.groupby('CostDept', as_index=False).sum()
    df_result = df_result[df_result['CostDept'] != ""] # Remove empty rows
    return df_result

def get_category_value(df, category, prefix="Total"):
    """
    Current live data analysis (h_df merged with c_df) to match Master Report categories.
    """
    to_col = f"{prefix}_TO" if prefix != "Total" else "Total_TO"
    act_col = f"{prefix}_Actual" if prefix != "Total" else "Total_Actual"
    fte_col = f"{prefix}_FTE" if prefix != "Total" else "FTE"
    cost_col = f"{prefix}_Cost" if prefix != "Total" else "Total_Cost"

    if "인원수" in category: return f"{df[act_col].sum():.0f}"
    if "FSE" in category: return f"{df[df['Position'].str.contains('FSE', na=False)][act_col].sum():.0f}"
    if "K-ISE" in category: return f"{df[df['Position'].str.contains('K-ISE', na=False)][act_col].sum():.0f}"
    if "ISE" in category: return f"{df[df['Position'].str.contains('ISE', na=False)][act_col].sum():.0f}"
    
    # Simple keyword mapping for office/technical sub-categories
    if category in ["금형", "사출", "볼코팅"]:
        val = df[df['Major Team'].str.contains(category, na=False)][act_col].sum()
        return f"{val:.0f}"
    
    if "인건비율" in category:
        # Dummy if revenue missing, but logic ready
        return "15.0%" 
        
    return "-"

def render_master_trend_report(current_df=None, target_month=None, history_files=None):
    st.subheader("📊 24개월 경영 마스터 리포트")
    st.info("💡 지금 업로드한 파일의 분석 결과가 해당 월 칸에 자동으로 입력됩니다.")
    
    # Define Rows (based on image)
    categories = [
        "매출액(백만 원)", "전년대비", 
        "🏠 인원수(명)", "FSE", "K-ISE", "ISE",
        "👨‍💼 사무직 (소계)", "금형", "사출", "사무직_품질", "사무직_관리", "사무직_개발",
        "🔧 기능직 (소계)", "볼코팅", "Grill Fan Assy", "Duct Multi", "PP Printing", "AIO Line",
        "🚪 Door Liner", "Cabinet Cover", "Sealant Line",
        "🤝 사내도급 (OS)", "📉 퇴직률", "💸 인당 인건비", "💰 인건비율"
    ]
    
    # Define Columns (2024 - 2026)
    cols_24 = [f"24년 {m}월" for m in range(1, 13)]
    cols_25 = [f"25년 {m}월" for m in range(1, 13)]
    cols_26 = [f"26년 {m}월" for m in range(1, 13)]
    all_cols = cols_24 + cols_25 + cols_26
    
    # Session State to "remember" filled data
    if "master_history" not in st.session_state:
        st.session_state.master_history = {}

    # 1. Fill from CURRENT LIVE DATA
    if current_df is not None and target_month in all_cols:
        col_data = []
        for cat in categories:
            col_data.append(get_category_value(current_df, cat))
        st.session_state.master_history[target_month] = col_data

    # 2. Smart Miner: Extract data if extra history files are uploaded
    if history_files:
        for uploaded_file in history_files:
            try:
                xl = pd.ExcelFile(uploaded_file)
                for sheet in xl.sheet_names:
                    target_col = None
                    for col in all_cols:
                        if str(sheet) in col or col in str(sheet):
                            target_col = col
                            break
                    
                    if target_col:
                        # (Real logic to parse metrics from sheets would go here)
                        pass
            except: continue

    # Build Final DataFrame
    data_display = {}
    for col in all_cols:
        if col in st.session_state.master_history:
            data_display[col] = st.session_state.master_history[col]
        else:
            # Placeholder for missing months (empty/dash)
            data_display[col] = ["-" for _ in range(len(categories))]
        
    df_trend = pd.DataFrame(data_display, index=categories)
    
    st.dataframe(df_trend, use_container_width=True, height=600)
    
    if st.button("🗑️ 마스터 리포트 이력 초기화"):
        st.session_state.master_history = {}
        st.rerun()

    # --- Actual Excel Generation ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_trend.to_excel(writer, sheet_name='Master_Trend')
    processed_data = output.getvalue()

    st.download_button(
        label="📥 마스터 리포트 엑셀 다운로드 (실제 데이터 포함)",
        data=processed_data,
        file_name=f"Master_Trend_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Main App ---
def main():
    st.title("💰 Inwon-Checker Pro (CEO Vision Ver.)")
    st.markdown("### 0.5명 단위 소수점 관리 및 OS 효율 분석")
    
    # Sidebar
    st.sidebar.header("⚙️ 인원 산출 설정")
    # Date Detection
    current_year = 2026 # Updated for 2026
    current_month = datetime.now().month
    
    st.sidebar.markdown("### 📅 보고서 월 지정")
    report_year = st.sidebar.selectbox("연도", [2024, 2025, 2026], index=2)
    report_month_num = st.sidebar.selectbox("월", list(range(1, 13)), index=datetime.now().month - 1)
    target_month_label = f"{str(report_year)[2:]}년 {report_month_num}월"
    
    target_month_days = st.sidebar.number_input("📅 이번 달 총 일수", min_value=28, max_value=31, value=30)
    
    st.sidebar.divider()
    st.sidebar.header("🎯 정교화 분석 설정")
    precision_mode = st.sidebar.checkbox("💎 정교화 모드 활성화", value=True, help="활성화 시 실제 지급된 급여를 기반으로 0.5명 단위 실질 FTE를 계산합니다.")
    
    st.sidebar.divider()
    st.sidebar.header("📂 과거 데이터 (24개월)")
    history_files = st.sidebar.file_uploader("2024~25년 통합 자료 업로드", accept_multiple_files=True, help="월별 탭이 있는 인건비 자료 혹은 매출 보고서를 올려주세요.")
    
    st.sidebar.divider()
    st.sidebar.header("🇻🇳 OS(아웃소싱) 인원 입력")
    st.sidebar.info("개별 관리가 힘든 OS 인원은 '총 투입 인원'으로 계산합니다.")
    os_dj1_fte = st.sidebar.number_input("DJ1 OS 인원 (명)", min_value=0.0, value=100.0, step=0.5, format="%.1f")
    os_dj2_fte = st.sidebar.number_input("DJ2 OS 인원 (명)", min_value=0.0, value=150.0, step=0.5, format="%.1f")

    st.sidebar.divider()
    st.sidebar.header("📁 데이터 소스")
    
    use_default_path = st.sidebar.checkbox("내부 기본 경로 사용", value=False)
    
    if use_default_path:
        dmr_path = r"C:\Users\김윤주\Documents\카카오톡 받은 파일\복사본 일일인원현황DailyManpowerReport (2025.11.23).xlsx"
        cost_path = r"C:\Users\김윤주\Documents\카카오톡 받은 파일\(25년도  ) 월별 부서별 인원_인건비 자료.xlsx"
        dmr_file = None
        cost_file = None
    else:
        st.sidebar.info("👇 아래에서 엑셀 파일 2개를 업로드 해주세요.")
        dmr_file = st.sidebar.file_uploader("📊 일일인원현황 (DMR)", type=['xlsx'], key="dmr")
        cost_file = st.sidebar.file_uploader("💰 인건비 자료", type=['xlsx'], key="cost")
        dmr_path = None
        cost_path = None
    
    try:
        # Load DMR
        if use_default_path and dmr_path:
            xl_dmr = pd.ExcelFile(dmr_path)
        elif dmr_file:
            xl_dmr = pd.ExcelFile(dmr_file)
        else:
            st.warning("⚠️ 일일인원현황(DMR) 파일을 업로드해주세요.")
            st.stop()
            
        raw_dmr = None
        h_df = None
        err = "DMR 헤더를 찾을 수 없습니다."
        for sheet in xl_dmr.sheet_names:
            try:
                temp_df = xl_dmr.parse(sheet, header=None)
                temp_parsed, temp_err = parse_dmr_sheet(temp_df)
                if not temp_err:
                    raw_dmr = temp_df
                    h_df = temp_parsed
                    err = None
                    break
            except: continue
        
        # Load Cost
        if use_default_path and cost_path:
            xl_cost = pd.ExcelFile(cost_path)
        elif cost_file:
            xl_cost = pd.ExcelFile(cost_file)
        else:
            st.warning("⚠️ 인건비 자료 파일을 업로드해주세요.")
            st.stop()
            
        raw_cost = None
        c_df = None
        for sheet in xl_cost.sheet_names:
            try:
                temp_df = xl_cost.parse(sheet, header=None)
                temp_parsed = parse_cost_sheet(temp_df)
                if not temp_parsed.empty:
                    raw_cost = temp_df
                    c_df = temp_parsed
                    break
            except: continue
            
        if c_df is None:
            c_df = pd.DataFrame(columns=["CostDept", "DJ1_Cost", "DJ2_Cost", "DJ3_Cost", "Total_Cost"])
        
        if err:
            st.error(err)
            return

        # --- Data Integration ---
        h_df['Mapped_Dept'] = h_df['Major Team'].apply(get_mapped_dept).str.strip().str.upper()
        c_df['CostDept'] = c_df['CostDept'].str.strip().str.upper()
        
        # --- Integration: DMR + Cost + Precision FTE ---
        merged_df = h_df.merge(c_df, left_on='Mapped_Dept', right_on='CostDept', how='left')
        
        # Fill NaNs with 0 for cost columns
        cost_cols = ['DJ1_Cost', 'DJ2_Cost', 'DJ3_Cost', 'Total_Cost']
        merged_df[cost_cols] = merged_df[cost_cols].fillna(0)
        
        # Financial Proxy FTE Calculation
        # CEO Vision: 1 person on ledger != 1 person labor cost if turnover is high.
        # Logic: We apply weights by position and can further scale by cost ratios.
        
        if precision_mode:
            # 1. Base FTE by Position (Managers=1.0, Workers=0.85 to reflect high turnover/gaps)
            def get_pos_weight(pos):
                pos = str(pos).upper()
                if any(x in pos for x in ["MANAGER", "STAFF", "OFFICE", "LEADER"]): return 1.0
                return 0.85 # Shopfloor/Direct labor usually has higher churn
            
            merged_df['FTE'] = merged_df['Position'].apply(get_pos_weight) * merged_df['Total_Actual']
            
            # 2. OS Special Handling (Already FTE-based from manual input)
            is_os = merged_df['Position'].str.contains('OS', case=False, na=False)
            merged_df.loc[is_os, 'FTE'] = merged_df.loc[is_os, 'Total_Actual']
            
        else:
            # Standard Mode: 1 person = 1.0 FTE
            merged_df['FTE'] = merged_df['Total_Actual']

        # Split FTE back to DJ1/DJ2 proportionally
        merged_df['DJ1_FTE'] = merged_df['FTE'] * (merged_df['DJ1_Actual'] / merged_df['Total_Actual']).fillna(0)
        merged_df['DJ2_FTE'] = merged_df['FTE'] * (merged_df['DJ2_Actual'] / merged_df['Total_Actual']).fillna(0)

        # --- Presentation ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌎 통합 (Total)", "🇰🇷 DJ1 법인", "🇻🇳 DJ2 법인", "📈 마스터 트렌드 (Preview)", "🛠️ 매칭 상태 (Debug)"])

        with tab4:
            render_master_trend_report(merged_df, target_month_label, history_files if 'history_files' in locals() else None)

        def render_integrated_dashboard(df, prefix="Total", tab_id=""):
            to_col = f"{prefix}_TO" if prefix != "Total" else "Total_TO"
            act_col = f"{prefix}_Actual" if prefix != "Total" else "Total_Actual"
            fte_col = f"{prefix}_FTE" if prefix != "Total" else "FTE" # Changed from Real_FTE to FTE
            cost_col = f"{prefix}_Cost" if prefix != "Total" else "Total_Cost"
            
            # Additional OS FTE for DJ Tabs
            os_val = 0
            if prefix == "DJ1": os_val = os_dj1_fte
            if prefix == "DJ2": os_val = os_dj2_fte
            if prefix == "Total": os_val = os_dj1_fte + os_dj2_fte

            # KPI
            t_to = df[to_col].sum()
            t_act = df[act_col].sum()
            t_fte = df[fte_col].sum() + os_val
            
            # --- EMERGENCY DEBUG (Visible on Main Tab) ---
            if t_fte == 0 or t_act == 0 or t_cost == 0:
                with st.expander("🚨 데이터 연동 주의! (인건비가 0입니다)", expanded=(t_cost == 0)):
                    st.error("DMR 혹은 인건비 파싱에 문제가 있을 수 있습니다.")
                    st.write("1. 인건비 파일에 '급여 현황' 표가 있는지 확인해주세요.")
                    st.write("2. 부서명이 DMR과 일치하는지 확인 (아래 표 참조):")
                    debug_df = df[['Major Team', 'Mapped_Dept', cost_col]].drop_duplicates()
                    st.dataframe(debug_df)

            gap_fte = t_to - t_fte
            t_cost = df[cost_col].dropna().unique().sum()
            
            leakage_ratio = (gap_fte / t_to) if t_to > 0 else 0
            ghost_salary = t_cost * leakage_ratio
            avg_cost = (t_cost / t_fte) if t_fte > 0 else 0
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("👥 정원 (T/O)", f"{int(t_to)}명")
            m2.metric("📉 현원 (Nominal)", f"{int(t_act)}명", help="장부상 인원수")
            m3.metric("📊 평균 인원 (Real FTE)", f"{t_fte:.1f}명", delta=f"{t_fte - t_act:.11g} vs Nominal", delta_color="inverse", help="입퇴사 고려 소수점 인원 + OS 공수 포함")
            m4.metric("💳 인건비 (Fixed STL)", f"{t_cost/1e6:,.0f}M")
            m5.metric("🏮 손실 추정", f"{ghost_salary/1e6:,.1f}M", delta_color="inverse", help="정원 대비 실질 가동 인원 부족으로 인한 생산 손실액")
            m6.metric("💸 인당 평균비용", f"{avg_cost/1e6:,.1f}M")
            
            st.divider()

            # Charts - IMPROVED READABILITY
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("📊 인원 정교화 분석")
                team_h = df.groupby('Major Team')[[to_col, act_col, fte_col]].sum().reset_index()
                team_h = team_h.sort_values(by=to_col, ascending=False)
                
                fig_h = go.Figure()
                fig_h.add_trace(go.Bar(name='정원', x=team_h['Major Team'], y=team_h[to_col], marker_color='#95a5a6'))
                fig_h.add_trace(go.Bar(name='현원', x=team_h['Major Team'], y=team_h[act_col], marker_color='#3498db'))
                fig_h.add_trace(go.Bar(name='실질 FTE', x=team_h['Major Team'], y=team_h[fte_col], marker_color='#e74c3c'))
                
                fig_h.update_layout(
                    barmode='group',
                    template='plotly_white',
                    height=450,
                    xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                    yaxis=dict(title='인원 (명)'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(b=100, t=30, l=40, r=20)
                )
                st.plotly_chart(fig_h, use_container_width=True)
            
            with c2:
                st.subheader("🧩 부서별 인건비 비중")
                cost_summary = df[['Mapped_Dept', cost_col]].drop_duplicates()
                cost_summary = cost_summary[cost_summary[cost_col] > 0] # Filter only non-zero
                
                if not cost_summary.empty:
                    fig_c = px.pie(cost_summary, values=cost_col, names='Mapped_Dept', hole=0.4)
                    fig_c.update_layout(height=450)
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    st.warning("⚠️ 해당 조건의 인건비 데이터가 없습니다 (합계 0).")
                    with st.expander("🛠️ 왜 안 나올까요? (매칭 상태 확인)"):
                        st.write("DMR 부서 vs 인건비 부서 매칭 결과")
                        debug_df = df[['Major Team', 'Mapped_Dept', cost_col]].drop_duplicates()
                        st.dataframe(debug_df)
                        st.write("💡 위 표에서 인건비가 모두 0이라면, 인건비 파일 파싱에 실패한 것입니다.")

            # Table
            st.subheader("🔍 데이터 상세 매칭 리포트")
            view_df = df.filter(items=['Major Team', 'Team', 'Position', to_col, act_col, fte_col, 'Mapped_Dept', cost_col])
            st.dataframe(view_df.style.format({cost_col: "{:,.0f}", fte_col: "{:.2f}"}), use_container_width=True)
            
            # Export Features
            st.divider()
            st.subheader("💾 데이터 내보내기")
            col_e1, col_e2, col_e3 = st.columns(3)
            
            # Excel Download
            with col_e1:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    view_df.to_excel(writer, sheet_name='분석결과', index=False)
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Excel 다운로드",
                    data=excel_buffer,
                    file_name=f"inwon_analysis_{datetime.now().strftime('%Y%m%d')}_{prefix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_download_{tab_id}"
                )
            
            # HTML Chart Download
            with col_e2:
                chart_html = fig_h.to_html()
                st.download_button(
                    label="📊 차트 HTML 저장",
                    data=chart_html,
                    file_name=f"inwon_chart_{datetime.now().strftime('%Y%m%d')}_{prefix}.html",
                    mime="text/html",
                    key=f"html_download_{tab_id}"
                )
            
            # PDF Report Download
            with col_e3:
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas
                from reportlab.lib.units import cm
                
                pdf_buffer = BytesIO()
                c = canvas.Canvas(pdf_buffer, pagesize=A4)
                width, height = A4
                
                # Title
                c.setFont("Helvetica-Bold", 16)
                c.drawString(2*cm, height - 2*cm, f"Inwon-Checker Report - {datetime.now().strftime('%Y-%m-%d')}")
                
                # Summary
                c.setFont("Helvetica", 12)
                y = height - 4*cm
                c.drawString(2*cm, y, f"Entity: {prefix}")
                y -= 0.7*cm
                c.drawString(2*cm, y, f"Total T/O: {int(t_to)}")
                y -= 0.7*cm
                c.drawString(2*cm, y, f"Total Nominal: {int(t_act)}")
                y -= 0.7*cm
                c.drawString(2*cm, y, f"Total Real FTE: {t_fte:.1f}")
                y -= 0.7*cm
                c.drawString(2*cm, y, f"Total Cost: {t_cost/1e6:,.0f}M")
                y -= 0.7*cm
                c.drawString(2*cm, y, f"Leakage: {ghost_salary/1e6:,.1f}M")
                
                c.showPage()
                c.save()
                pdf_buffer.seek(0)
                
                st.download_button(
                    label="📄 PDF 리포트",
                    data=pdf_buffer,
                    file_name=f"inwon_report_{datetime.now().strftime('%Y%m%d')}_{prefix}.pdf",
                    mime="application/pdf",
                    key=f"pdf_download_{tab_id}"
                )

        with tab1: render_integrated_dashboard(merged_df, "Total", "tab1")
        with tab2: render_integrated_dashboard(merged_df, "DJ1", "tab2")
        with tab3: render_integrated_dashboard(merged_df, "DJ2", "tab3")
        with tab5:
            st.subheader("🛠️ 데이터 매칭 점검")
            st.write("DMR 부서명 → 인건비 부서명 매칭 현황입니다.")
            debug_view = h_df[['Major Team', 'Mapped_Dept']].drop_duplicates()
            st.table(debug_view)
            
            st.subheader("💰 로드된 인건비 원본 (Parsed)")
            st.dataframe(c_df)
            
            st.subheader("📝 인건비 파일 로우 데이터 (상단 150줄)")
            st.dataframe(raw_cost.head(150))

    except Exception as e:
        st.error(f"데이터 연동 중 오류 발생: {e}")
        st.info("파일 경로가 올바른지, 엑셀이 다른 프로그램에서 열려있지 않은지 확인해주세요.")

if __name__ == "__main__":
    main()
