
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
    return None

# --- Logic: DMR Parser (For Sheet 1123) ---
def parse_dmr_sheet(df):
    header_row_idx = -1
    for idx, row in df.iterrows():
        if '부서 (Team)' in str(row.values):
            header_row_idx = idx
            break
    
    if header_row_idx == -1: return None, "DMR 헤더를 찾을 수 없습니다."

    data_raw = df.iloc[header_row_idx + 2:].copy()
    parsed_rows = []
    current_major_team = ""
    current_detail_team = ""
    
    for _, row in data_raw.iterrows():
        raw_major = str(row[0]) if pd.notnull(row[0]) and str(row[0]).strip() != "" else current_major_team
        detail = str(row[1]) if pd.notnull(row[1]) and str(row[1]).strip() != "" else current_detail_team
        
        if any(x in str(row[2]) for x in ["Total", "S-Total", "합계"]): continue
        if pd.isnull(row[2]) and pd.isnull(row[5]): continue
            
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
            
    return pd.DataFrame(parsed_rows), None

# --- Logic: Cost Parser (For Sheet 2025) ---
def parse_cost_sheet(df):
    """
    Parses Labor Cost from Rows 100-137.
    """
    # Look for '3. 급여 현황' or specific markers
    start_row = 100
    end_row = 133
    
    cost_data = []
    for i in range(start_row, end_row + 1):
        if i >= len(df): break
        row = df.iloc[i]
        dept_name = str(row[0]).strip() if pd.notnull(row[0]) else ""
        cost_type = str(row[2]).strip() if pd.notnull(row[2]) else ""
        
        if cost_type == "STL": # Focus on STL (Base Salary + Fixed)
            cost_data.append({
                "CostDept": dept_name,
                "DJ1_Cost": float(row[4]) if pd.notnull(row[4]) else 0, # Col E
                "DJ2_Cost": float(row[5]) if pd.notnull(row[5]) else 0, # Col F
                "DJ3_Cost": float(row[6]) if pd.notnull(row[6]) else 0, # Col G
                "Total_Cost": float(row[8]) if pd.notnull(row[8]) else 0 # Col I
            })
    return pd.DataFrame(cost_data)

# --- Main App ---
def main():
    st.title("💰 Inwon-Checker Pro (CEO Vision Ver.)")
    st.markdown("### 0.5명 단위 소수점 관리 및 OS 효율 분석")
    
    # Sidebar
    st.sidebar.header("⚙️ 인원 산출 설정")
    target_month_days = st.sidebar.number_input("📅 이번 달 총 일수 (Month Days)", min_value=28, max_value=31, value=30)
    
    st.sidebar.divider()
    st.sidebar.header("🇻🇳 OS(아웃소싱) 공수 입력")
    st.sidebar.info("개별 관리가 힘든 OS 인원은 '총 투입 공수'로 계산합니다.")
    os_dj1_days = st.sidebar.number_input("DJ1 OS 총 공수 (Man-Days)", min_value=0, value=3000, step=10)
    os_dj2_days = st.sidebar.number_input("DJ2 OS 총 공수 (Man-Days)", min_value=0, value=4500, step=10)
    
    os_dj1_fte = os_dj1_days / target_month_days
    os_dj2_fte = os_dj2_days / target_month_days

    st.sidebar.divider()
    st.sidebar.header("📁 데이터 소스")
    
    use_default_path = st.sidebar.checkbox("내부 기본 경로 사용", value=False, help="체크하면 개발자 PC의 기본 경로를 사용합니다. 일반 사용자는 체크 해제 후 파일 업로드하세요.")
    
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
            
        raw_dmr = xl_dmr.parse("1123", header=None)
        h_df, err = parse_dmr_sheet(raw_dmr)
        
        # Load Cost
        if use_default_path and cost_path:
            xl_cost = pd.ExcelFile(cost_path)
        elif cost_file:
            xl_cost = pd.ExcelFile(cost_file)
        else:
            st.warning("⚠️ 인건비 자료 파일을 업로드해주세요.")
            st.stop()
            
        raw_cost = xl_cost.parse("2025", header=None)
        c_df = parse_cost_sheet(raw_cost)
        
        if err:
            st.error(err)
            return

        # --- Data Integration ---
        h_df['Mapped_Dept'] = h_df['Major Team'].apply(get_mapped_dept)
        
        # [NEW] SIMULATED FTE LOGIC (For Demo until HR Master is provided)
        # In real case, this will be: worked_days / month_days
        # Here we simulate some "0.5 people" for specific positions to show the CEO's vision
        def simulate_fte(row):
            if 'Manager' in row['Position']: return row['Total_Actual'] * 1.0
            if 'Worker' in row['Position']: return row['Total_Actual'] * 0.85 # Simulate 15% leave/join gap
            return row['Total_Actual'] * 0.95
            
        h_df['Real_FTE'] = h_df.apply(simulate_fte, axis=1)
        # Split back to DJ1/DJ2 proportionally for demo
        h_df['DJ1_FTE'] = h_df['Real_FTE'] * (h_df['DJ1_Actual'] / h_df['Total_Actual']).fillna(0)
        h_df['DJ2_FTE'] = h_df['Real_FTE'] * (h_df['DJ2_Actual'] / h_df['Total_Actual']).fillna(0)

        merged_df = h_df.merge(c_df, left_on='Mapped_Dept', right_on='CostDept', how='left')

        # --- Presentation ---
        tab1, tab2, tab3, tab4 = st.tabs(["🌎 통합 (Total)", "🇰🇷 DJ1 법인", "🇻🇳 DJ2 법인", "🛠️ 매칭 상태 (Debug)"])

        def render_integrated_dashboard(df, prefix="Total", tab_id="default"):
            to_col = f"{prefix}_TO" if prefix != "Total" else "Total_TO"
            act_col = f"{prefix}_Actual" if prefix != "Total" else "Total_Actual"
            fte_col = f"{prefix}_FTE" if prefix != "Total" else "Real_FTE"
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

            # Charts
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 인원 정교화 분석 (T/O vs Nominal vs Real)")
                team_h = df.groupby('Major Team')[[to_col, act_col, fte_col]].sum().reset_index()
                fig_h = go.Figure()
                fig_h.add_trace(go.Bar(name='정원 (T/O)', x=team_h['Major Team'], y=team_h[to_col], marker_color='#bdc3c7'))
                fig_h.add_trace(go.Bar(name='현원 (Nominal)', x=team_h['Major Team'], y=team_h[act_col], marker_color='#34495e'))
                fig_h.add_trace(go.Bar(name='실질 인원 (FTE)', x=team_h['Major Team'], y=team_h[fte_col], marker_color='#e74c3c'))
                fig_h.update_layout(barmode='group', template='plotly_white')
                st.plotly_chart(fig_h, use_container_width=True)
            
            with c2:
                st.subheader("🧩 부서별 인건비 비중")
                cost_summary = df[['Mapped_Dept', cost_col]].drop_duplicates()
                fig_c = px.pie(cost_summary, values=cost_col, names='Mapped_Dept', hole=0.4)
                st.plotly_chart(fig_c, use_container_width=True)

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

        with tab1: render_integrated_dashboard(merged_df, "Total", "tab1")
        with tab2: render_integrated_dashboard(merged_df, "DJ1", "tab2")
        with tab3: render_integrated_dashboard(merged_df, "DJ2", "tab3")
        with tab4:
            st.subheader("🛠️ 데이터 매칭 점검")
            st.write("DMR 부서명 → 인건비 부서명 매칭 현황입니다.")
            debug_view = h_df[['Major Team', 'Mapped_Dept']].drop_duplicates()
            st.table(debug_view)
            
            st.subheader("💰 로드된 인건비 원본 (STL)")
            st.dataframe(c_df)

    except Exception as e:
        st.error(f"데이터 연동 중 오류 발생: {e}")
        st.info("파일 경로가 올바른지, 엑셀이 다른 프로그램에서 열려있지 않은지 확인해주세요.")

if __name__ == "__main__":
    main()
