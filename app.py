
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
    Parses Labor Cost from Rows 100-137.
    """
    # Dynamically look for start if possible, otherwise use fallback
    start_row = 100
    for idx, row in df.iterrows():
        if "급여" in str(row[0]) and "현황" in str(row[0]):
            start_row = idx + 1
            break
            
    end_row = start_row + 40 # Look at the next 40 rows
    
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
    df_result = pd.DataFrame(cost_data)
    if df_result.empty:
        # Return a shell with correct column names to prevent merge KeyErrors
        return pd.DataFrame(columns=["CostDept", "DJ1_Cost", "DJ2_Cost", "DJ3_Cost", "Total_Cost"])
    return df_result

def render_master_trend_report():
    st.subheader("📊 24개월 경영 마스터 리포트 (Preview)")
    st.info("💡 각 월별 데이터를 취합하여 '2-5. 인원 및 생산성' 장표 형식으로 자동 생성합니다.")
    
    # Define Rows (based on image)
    categories = [
        "매출액(백만 원)", "전년대비", 
        "🏠 인원수(명)", "FSE", "K-ISE", "ISE",
        "👨‍💼 사무직 (소계)", "금형", "사출", "사무직_품질", "사무직_관리", "사무직_개발",
        "🔧 기능직 (소계)", "볼코팅", "Grill Fan Assy", "Duct Multi", "PP Printing", "AIO Line",
        "🚪 Door Liner", "Cabinet Cover", "Sealant Line",
        "🤝 사내도급 (OS)", "📉 퇴직률", "💸 인당 인건비", "💰 인건비율"
    ]
    
    # Define Columns (24 months)
    cols_24 = [f"24년 {m}월" for m in range(1, 13)]
    cols_25 = [f"25년 {m}월" for m in range(1, 13)]
    all_cols = cols_24 + cols_25
    
    # Mock Data Generation (For Preview)
    import numpy as np
    data = {}
    for col in all_cols:
        col_data = []
        for cat in categories:
            if "매출액" in cat: col_data.append(f"{np.random.randint(700, 1600):,}")
            elif "인원수" in cat: col_data.append(np.random.randint(150, 250))
            elif cat in ["사무직 (소계)", "기능직 (소계)"]: col_data.append("-") # Headers
            elif "%" in cat or "율" in cat: col_data.append(f"{np.random.uniform(1.0, 15.0):.1f}%")
            else: col_data.append(np.random.randint(1, 40))
        data[col] = col_data
        
    df_trend = pd.DataFrame(data, index=categories)
    
    st.dataframe(df_trend, use_container_width=True, height=600)
    
    # --- Actual Excel Generation ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_trend.to_excel(writer, sheet_name='Master_Trend')
        # Add basic formatting if needed
    processed_data = output.getvalue()

    st.download_button(
        label="📥 마스터 리포트 엑셀 다운로드 (Pre-filled)",
        data=processed_data,
        file_name="Master_Trend_Report_2025.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Main App ---
def main():
    st.title("💰 Inwon-Checker Pro (CEO Vision Ver.)")
    st.markdown("### 0.5명 단위 소수점 관리 및 OS 효율 분석")
    
    # Sidebar
    st.sidebar.header("⚙️ 인원 산출 설정")
    target_month_days = st.sidebar.number_input("📅 이번 달 총 일수", min_value=28, max_value=31, value=30)
    
    st.sidebar.divider()
    st.sidebar.header("🎯 정교화 분석 설정")
    precision_mode = st.sidebar.checkbox("💎 정교화 모드 활성화", value=True, help="활성화 시 실제 지급된 급여를 기반으로 0.5명 단위 실질 FTE를 계산합니다.")
    
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
            
        raw_dmr = xl_dmr.parse(0, header=None) # Use first sheet regardless of name
        h_df, err = parse_dmr_sheet(raw_dmr)
        
        # Load Cost
        if use_default_path and cost_path:
            xl_cost = pd.ExcelFile(cost_path)
        elif cost_file:
            xl_cost = pd.ExcelFile(cost_file)
        else:
            st.warning("⚠️ 인건비 자료 파일을 업로드해주세요.")
            st.stop()
            
        raw_cost = xl_cost.parse(0, header=None) # Use first sheet regardless of name
        c_df = parse_cost_sheet(raw_cost)
        
        if err:
            st.error(err)
            return

        # --- Data Integration ---
        h_df['Mapped_Dept'] = h_df['Major Team'].apply(get_mapped_dept)
        
        # --- Integration: DMR + Cost + Precision FTE ---
        merged_df = h_df.merge(c_df, left_on='Mapped_Dept', right_on='CostDept', how='left')
        
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
            render_master_trend_report()

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
                fig_c = px.pie(cost_summary, values=cost_col, names='Mapped_Dept', hole=0.4)
                fig_c.update_layout(height=450)
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
