import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
import io
from datetime import datetime, timedelta
import concurrent.futures

# -----------------------------------------------------------------------------
# 1. 설정 및 유틸리티
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. 설정 및 유틸리티
# -----------------------------------------------------------------------------
DATA_FILE = "sp500_data.pkl"
ANALYST_FILE = "sp500_analyst.pkl"
SPY_TICKER = "SPY"

# Streamlit 페이지 설정
st.set_page_config(page_title="SP500 퀀트 전략 (Alpha Hunter)", layout="wide")

@st.cache_data
def get_sp500_tickers_and_names():
    """위키피디아에서 S&P 500 티커와 회사명을 가져옵니다."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        table = pd.read_html(io.StringIO(response.text))
        df = table[0]
        
        # 티커와 회사명 매핑 딕셔너리 생성
        mapping = dict(zip(df['Symbol'], df['Security']))
        
        # yfinance용 티커 수정 및 매핑 키 수정
        clean_mapping = {}
        for t, n in mapping.items():
            clean_t = t.replace('.', '-')
            clean_mapping[clean_t] = n
            
        tickers = list(clean_mapping.keys())
        return tickers, clean_mapping
    except Exception as e:
        st.error(f"티커 목록 가져오기 실패: {e}")
        return [], {}

def get_analyst_upgrades(tickers):
    """
    각 티커별 'upgrades_downgrades'를 조회하여 상세 정보를 가져옵니다.
    Return: {ticker: {'is_up': bool, 'desc': str}}
    """
    # 1. 기존 데이터 로드
    if os.path.exists(ANALYST_FILE):
        try:
            stored_data = pd.read_pickle(ANALYST_FILE)
            file_time = datetime.fromtimestamp(os.path.getmtime(ANALYST_FILE)).date()
            if file_time == datetime.now().date():
                 return stored_data
            else:
                 pass 
        except:
             pass
    
    # 데이터가 없으면 빈 딕셔너리 (업데이트 버튼으로 수행)
    return {}

# 실제 업데이트를 수행하는 함수 (버튼 연결용)
def update_analyst_data_action(tickers):
    st.info("애널리스트 평가 정보를 업데이트합니다. (약 1~3분 소요)")
    results = {}
    
    progress_bar = st.progress(0, text="Analyst Data Fetching...")
    
    # ThreadPool로 가속
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_single_analyst, t): t for t in tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
            t = future_to_ticker[future]
            try:
                # Returns (ticker, is_up, description_str)
                ticker, is_up, desc = future.result()
                results[ticker] = {'is_up': is_up, 'desc': desc}
            except:
                results[t] = {'is_up': False, 'desc': '-'}
            
            if i % 10 == 0:
                 progress_bar.progress((i + 1) / len(tickers))
                 
    progress_bar.empty()
    
    # 저장
    pd.to_pickle(results, ANALYST_FILE)
    return results

def fetch_single_analyst(ticker):
    try:
        t = yf.Ticker(ticker)
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            ud.index = pd.to_datetime(ud.index)
            # 최근 30일
            recent = ud[ud.index >= (datetime.now() - timedelta(days=30))]
            if not recent.empty:
                # 가장 최근 액션
                latest = recent.iloc[-1]
                action = str(latest.get('Action', '')) # Up, Down, Main, Init...
                from_g = str(latest.get('FromGrade', ''))
                to_g = str(latest.get('ToGrade', ''))
                
                desc = f"{action} ({from_g}->{to_g})"
                
                is_up = False
                if 'Up' in action or 'Init' in action:
                    is_up = True
                    
                return ticker, is_up, desc
    except:
        pass
    return ticker, False, "-"

# -----------------------------------------------------------------------------
# 2. 데이터 관리 (Incremental Update) - 기존 유지
# -----------------------------------------------------------------------------
def load_and_update_data(tickers):
    # (기존 코드와 동일 - 생략 없이 전체 포함해야 replace가 잘 됨, 여기서는 기존 함수 내용을 그대로 써야 함)
    # 하지만 replace_file_content는 부분 교체가 가능하므로, 
    # 여기서는 Analyst 함수부와 Main 부만 수정하면 될 것 같지만, 안전하게 전체 컨텍스트 고려
    # 일단 load_and_update_data 는 수정할 필요 없음.
    start_date = None
    existing_df = pd.DataFrame()
    full_data = pd.DataFrame()

    if os.path.exists(DATA_FILE):
        try:
            if os.path.getsize(DATA_FILE) < 100 * 1024:
                st.warning("⚠️ 기존 데이터 파일이 손상되었거나 너무 작습니다. 삭제 후 다시 받습니다.")
                os.remove(DATA_FILE)
            else:
                existing_df = pd.read_pickle(DATA_FILE)
                if not existing_df.empty:
                    last_date = existing_df.index[-1].date()
                    today = datetime.now().date()
                    if last_date < (today - timedelta(days=1)):
                        st.info(f"🔄 기존 데이터({last_date}) 이후를 업데이트합니다...")
                        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    else:
                        st.success(f"✅ 데이터가 최신입니다 ({last_date}).")
                        return existing_df
        except Exception as e:
            st.warning(f"⚠️ 기존 파일 로드 오류 ({e}). 새로 다운로드합니다.")
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            existing_df = pd.DataFrame()

    if start_date is None:
        if existing_df.empty:
            st.info("⬇️ 전체 데이터를 다운로드합니다 (최근 2년). 50개씩 분할 다운로드 중...")
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    target_tickers = list(set(tickers + [SPY_TICKER]))
    new_data_list = []
    
    if start_date:
        chunk_size = 50
        chunks = [target_tickers[i:i + chunk_size] for i in range(0, len(target_tickers), chunk_size)]
        progress_bar = st.progress(0, text="데이터 다운로드 중...")
        for i, chunk in enumerate(chunks):
            try:
                batch_data = yf.download(chunk, start=start_date, group_by='ticker', threads=True, progress=False, auto_adjust=True)
                if not batch_data.empty:
                    new_data_list.append(batch_data)
            except Exception as e:
                st.error(f"⚠️ 다운로드 중 오류 발생 (Batch {i}): {e}")
            progress_bar.progress((i + 1) / len(chunks), text=f"데이터 다운로드 중... ({i+1}/{len(chunks)})")
        progress_bar.empty()
        
        if new_data_list:
            new_data = pd.concat(new_data_list, axis=1)
            if not existing_df.empty:
                combined = pd.concat([existing_df, new_data])
                combined = combined[~combined.index.duplicated(keep='last')]
                full_data = combined
            else:
                full_data = new_data
            full_data.to_pickle(DATA_FILE)
            st.success("💾 데이터 업데이트 및 저장 완료!")
        else:
            st.warning("새로운 데이터를 가져오지 못했습니다.")
            full_data = existing_df
    else:
        full_data = existing_df

    return full_data

# -----------------------------------------------------------------------------
# 3. 전략 계산 로직
# -----------------------------------------------------------------------------
def calculate_strategies(df, tickers, ticker_names, analyst_data):
    results = []
    
    if SPY_TICKER in df.columns.levels[0]:
        spy = df[SPY_TICKER].copy()
    else:
        spy = pd.DataFrame()

    progress_bar = st.progress(0, text="전략 분석 중...")
    
    for idx, ticker in enumerate(tickers):
        if ticker == SPY_TICKER: continue
        if ticker not in df.columns.levels[0]: continue
        
        data = df[ticker].dropna(how='all') 
        if len(data) < 60: continue 

        try:
            close = data['Close']
            volume = data['Volume']
            
            curr_price = close.iloc[-1]
            prev_price = close.iloc[-2]
            curr_vol = volume.iloc[-1]
            prev_vol = volume.iloc[-2] if len(volume) > 1 else curr_vol
            
            price_chg_pct = ((curr_price - prev_price) / prev_price) * 100
            vol_chg_pct = ((curr_vol - prev_vol) / (prev_vol + 1e-9)) * 100
            
            company_name = ticker_names.get(ticker, ticker)
            
            # --- 전략 1: VCP ---
            std_10 = close.rolling(10).std().iloc[-1]
            std_60 = close.rolling(60).std().iloc[-1]
            vol_ma5 = volume.rolling(5).mean().iloc[-1]
            vol_ma20 = volume.rolling(20).mean().iloc[-1]
            
            vcp_ratio = std_10 / (std_60 + 1e-9)
            is_vol_dry = vol_ma5 < (vol_ma20 * 0.7)
            
            score_vcp = 0
            if vcp_ratio < 0.5: score_vcp += 10 
            if vcp_ratio < 0.7: score_vcp += 5
            if is_vol_dry: score_vcp += 10 
            
            # --- 전략 2: RS ---
            score_rs = 0
            if not spy.empty and len(spy) > 60:
                stock_ret_3m = close.pct_change(60).iloc[-1]
                spy_ret_3m = spy['Close'].pct_change(60).iloc[-1]
                rs_rating = stock_ret_3m - spy_ret_3m
                spy_ret_1m = spy['Close'].pct_change(20).iloc[-1]
                stock_ret_1m = close.pct_change(20).iloc[-1]
                if rs_rating > 0.1: score_rs += 10 
                elif rs_rating > 0: score_rs += 5
                if spy_ret_1m < 0 and stock_ret_1m > -0.02: score_rs += 10
            
            high_52 = close.rolling(250).max().iloc[-1]
            if curr_price >= high_52 * 0.85: score_rs += 10

            # --- 전략 3: Pocket Pivot ---
            last_10_days = data.iloc[-11:-1]
            down_days = last_10_days[last_10_days['Close'] < last_10_days['Open']]
            max_down_vol = down_days['Volume'].max() if not down_days.empty else 0
            ma10 = close.rolling(10).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            score_pocket = 0
            if curr_vol > max_down_vol and (curr_price > ma10 or curr_price > ma50):
                score_pocket = 20

            # --- 전략 4: OBV ---
            score_obv = 0
            price_diff = close.diff()
            obv_dir = np.sign(price_diff).fillna(0)
            obv = (obv_dir * volume).cumsum()
            p_slope = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            obv_slope = (obv.iloc[-1] - obv.iloc[-20]) / (abs(obv.iloc[-20]) + 1e-9)
            
            if -0.05 <= p_slope <= 0.05:
                if obv_slope > 0.15: 
                    score_obv = 30 
                    if obv_slope > 0.1: score_obv = 20
                    elif obv_slope > 0: score_obv = 10
            
            # --- 전략 5: Analyst ---
            score_eps = 0
            analyst_info = analyst_data.get(ticker, {'is_up': False, 'desc': '-'})
            if isinstance(analyst_info, bool): # 호환성
                 analyst_info = {'is_up': analyst_info, 'desc': 'Check Update'}

            is_analyst_up = analyst_info.get('is_up', False)
            analyst_desc = analyst_info.get('desc', '-')
            
            if is_analyst_up:
                score_eps = 20 


            # --- 전략 6: GMMA ---
            score_gmma = 0
            mas = []
            for period in [3,5,8,10,12,15, 30,35,40,45,50,60]:
                mas.append(close.rolling(period).mean().iloc[-1])
            gmma_std = np.std(mas)
            if gmma_std / curr_price < 0.02:
                score_gmma = 20
            
            short_group_avg = np.mean(mas[:6])
            long_group_avg = np.mean(mas[6:])
            if short_group_avg > long_group_avg and (short_group_avg / long_group_avg) < 1.02:
                score_gmma += 10

            # --- 합산 ---
            total_score = score_vcp + score_rs + score_pocket + score_obv + score_eps + score_gmma
            stealth_score = score_obv + (score_eps * 2) 
            
            results.append({
                'Ticker': ticker,
                'Name': company_name,
                'Total Score': total_score,
                'Analyst Score': score_eps,
                'VCP': score_vcp,
                'RS': score_rs,
                'Pocket': score_pocket,
                'OBV': score_obv,
                'GMMA': score_gmma,
                'Price': curr_price,
                'Chg(%)': round(price_chg_pct, 2),
                'Vol Chg(%)': round(vol_chg_pct, 2),
                'Analyst Change': analyst_desc
            })

        except Exception as e:
            continue
        
        if idx % 10 == 0:
            progress_bar.progress((idx+1)/len(tickers), text=f"전략 분석 중... ({idx+1}/{len(tickers)})")
            
    progress_bar.empty()
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 4. Streamlit UI 메인
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="SP500 퀀트 전략 (Alpha Hunter)", layout="wide") # Duplicate 방지 위해 위에서 호출했지만 여기도 둠 (문제없음)
    st.title("🚀 S&P 500 Alpha Hunter (Q1)")
    
    with st.expander("ℹ️ 전략별 점수 산정 기준 보기"):
        st.markdown("""
        **전략1) VCP (변동성 축소)** : 주가가 급등하기 전에는 반드시 변동성이 줄어들며 숨을 고르는 구간이 있다
        - 변동성 축소(0.5 이하 : 10점) & 거래량 말라감(0.7 이하 : 5점)
        - 거래량 감축(5일 거래량이 20일간 거래량의 70%, 10점) 점수화
        - 매수신호는 볼륨 폭발(2배) & 가격 상승(3% 이상)

        **전략2) RS (상대 강도) 다이버전스** : 지수가 하락하거나 횡보할 때 **'혼자 안 떨어지는 종목'**을 찾는 방법. 세력(기관/외국인)의 주가 관리 증거.
        - 시장대비 10% 초과 수익(3개월), 시장하락시 방어 (20일), 신고가 근처(15%) 각 10점 (5% 초과 수익은 5점)

        **전략3) Pocket Pivot (거래량 돌파)** : 박스권 내에서 기관의 매집 흔적을 찾아 미리 진입하는 공격적 전략
        - 오늘 거래량이 지난 10일간의 '최대 하락일 거래량'보다 많으면 20점

        **전략4) OBV 다이버전스 (스텔스 매집 포착)** : 주가는 횡보/하락 중인데 누군가 몰래 매집(OBV 상승)하는 경우
        - 가격은 횡보/하락(-5% ~ +5%)인데 OBV는 상승 15%: 30점, 10% : 20점, 0%이상 10점

        **전략5) Analyst Revisions** : 애널리스트의 추천이 상향된 경우
        - 최근 1개월 내 투자의견 상향 시 20점

        **전략6) GMMA (압축과 확산)** : 단기 이평선과 장기 이평선이 모였다가(압축) 펼쳐지는(확산) 초기 포착
        - 표준편차가 2%이내로 초 압축시 20점
        - 골든크로스 초기 추가 10점
        """)
    
    with st.sidebar:
        st.header("⚙️ 데이터 제어")
        
        if 'tickers_map' not in st.session_state:
            tickers, t_map = get_sp500_tickers_and_names()
            st.session_state['tickers'] = tickers
            st.session_state['tickers_map'] = t_map
        
        tickers = st.session_state.get('tickers', [])
        t_map = st.session_state.get('tickers_map', {})
        
        st.write(f"대상 종목 수: {len(tickers)}개")
        
        if st.button("티커 및 회사명 목록 갱신"):
            tickers, t_map = get_sp500_tickers_and_names()
            st.session_state['tickers'] = tickers
            st.session_state['tickers_map'] = t_map
            st.success("갱신 완료")
            
        st.divider()
        
        st.write("📊 애널리스트 의견 업데이트")
        if st.button("Analyst Data Update (Slow)"):
            analyst_data = update_analyst_data_action(tickers)
            st.session_state['analyst_data'] = analyst_data
            st.success("업데이트 완료")
        
        st.divider()
        
        if st.button("데이터 분석 시작 (Start Job)"):
            with st.spinner("데이터 동기화 및 분석 중..."):
                df_all = load_and_update_data(tickers)
                analyst_data = get_analyst_upgrades(tickers)
                if 'analyst_data' in st.session_state:
                    analyst_data = st.session_state['analyst_data']
                
                if not df_all.empty:
                    res_df = calculate_strategies(df_all, tickers, t_map, analyst_data)
                    if not res_df.empty:
                        res_df = res_df.sort_values(by='Total Score', ascending=False)
                        st.session_state['results'] = res_df
                        st.success("분석 완료!")
                    else:
                        st.warning("조건에 맞는 종목이 없습니다.")
                else:
                    st.error("데이터 로드 실패")

    # 메인 결과 화면
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        tab1, tab2 = st.tabs(["🔥 급등 임박 (Watchlist)", "🐉 스텔스 잠룡 (Hidden Dragon)"])
        
        # 표시할 컬럼 (순서 조정)
        cols_to_show = [
            'Ticker', 'Name', 'Total Score', 
            'Price', 'Chg(%)', 'Vol Chg(%)',
            'VCP', 'RS', 'Pocket', 'OBV', 'GMMA', 'Analyst Score', 'Analyst Change'
        ]
        
        selected_ticker = None
        
        with tab1:
            st.subheader("Top Picks (테이블 행을 클릭하여 차트 확인)")
            top_df = results.head(50)
            
            if not top_df.empty:
                # [Interactivity] on_select 사용
                event = st.dataframe(
                    top_df[cols_to_show].style.background_gradient(subset=['Total Score'], cmap='Reds')
                                      .format({'Price': '{:.2f}', 'Chg(%)': '{:+.2f}', 'Vol Chg(%)': '{:+.2f}'}),
                    use_container_width=True,
                    height=600,
                    on_select="rerun", # 행 선택 시 리런
                    selection_mode="single-row"
                )
                
                if len(event.selection.rows) > 0:
                    selected_idx = event.selection.rows[0]
                    selected_ticker = top_df.iloc[selected_idx]['Ticker']
        
        with tab2:
            st.subheader("매집 징후 포착 (OBV Divergence)")
            # Stealth View에서는 중요 컬럼 위주 (중복 방지 명시적 정의)
            stealth_cols = ['Ticker', 'Name', 'Total Score', 'Analyst Score', 'VCP', 'RS', 'Pocket', 'OBV', 'GMMA', 'Analyst Change']
            
            if 'Stealth Score' in results.columns:
                stealth_df = results.sort_values(by='Stealth Score', ascending=False).head(30)
            else:
                stealth_df = results.head(30)
            
            st.dataframe(
                stealth_df[stealth_cols].style.background_gradient(subset=['OBV'], cmap='Greens'),
                use_container_width=True
            )
            
        # 차트 분석 섹션
        st.divider()
        st.subheader("📊 차트 정밀 분석")
        
        if selected_ticker:
            try:
                full_df = pd.read_pickle(DATA_FILE)
                if selected_ticker in full_df.columns.levels[0]:
                    stock_data = full_df[selected_ticker].dropna().tail(250) 
                    
                    ma20 = stock_data['Close'].rolling(20).mean()
                    std20 = stock_data['Close'].rolling(20).std()
                    upper = ma20 + (std20 * 2)
                    lower = ma20 - (std20 * 2)
                    
                    chart_df = pd.DataFrame({
                        'Close': stock_data['Close'],
                        'Upper BB': upper,
                        'Lower BB': lower,
                        'MA50': stock_data['Close'].rolling(50).mean()
                    })
                    
                    st.caption(f"선택된 종목: **{selected_ticker}**")
                    st.line_chart(chart_df, color=["#FF0000", "#AAAAAA", "#AAAAAA", "#0000FF"])
                    st.bar_chart(stock_data['Volume'])
                    
                    row = results[results['Ticker'] == selected_ticker].iloc[0]
                    st.info(f"**{row['Name']}** ({row['Ticker']})")
                    st.write(f"**종합 점수:** {row['Total Score']}점")
                    st.write(f"**Analyst:** {row['Analyst Score']} ({row['Analyst Change']})")
                    
                    bandwidth = (upper - lower) / ma20
                    st.area_chart(bandwidth)
                    st.caption("볼린저 밴드 폭 (낮을수록 수렴)")
            except Exception as e:
                st.error(f"차트 데이터 로드 실패: {e}")
        else:
            st.info("👆 위 테이블에서 종목을 선택(클릭)하면 상세 차트가 표시됩니다.")

if __name__ == "__main__":
    main()
