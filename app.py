import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Konfiguracija stranice i izgleda aplikacije (zadržan wide layout)
st.set_page_config(page_title="Predviđanje Potrošnje Energije", layout="wide")

# Učitavanje podataka pripremljenih iz Colab okruženja
@st.cache_data
def ucitaj_podatke():
    # KLJUČNI POPRAVAK: Eksplicitno pretvaramo prvi stupac u DatetimeIndex i sortiramo ga
    df = pd.read_csv('streamlit_data.csv', parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

try:
    df = ucitaj_podatke()
except Exception:
    st.error("Konfiguracijska pogreška: Datoteka 'streamlit_data.csv' nije pronađena u radnom direktoriju aplikacije.")
    st.stop()

# Glavni naslov aplikacije u minimalističkom stilu
st.title("Analiza i predviđanje potrošnje električne energije")
st.markdown("Ova interaktivna aplikacija demonstrira primjenu tradicionalnih algoritama strojnog učenja i naprednih arhitektura dubokog učenja na problemu modeliranja vremenskih nizova u elektroenergetskom sustavu.")

# Stvaranje tri funkcionalne kartice za navigaciju
tab1, tab2, tab3 = st.tabs(["Pregled projekta", "Eksploratorna analiza podataka", "Usporedba performansi modela"])

# --- 1. KARTICA: PREGLED PROJEKTA ---
with tab1:
    st.header("Teorijski okvir i struktura vremenskog niza")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Koncept vremenskih nizova
        Vremenski niz predstavlja kronološki uređen slijed opažanja snimljenih u jednakim i uzastopnim vremenskim intervalima. U okviru ovog projekta analiziraju se povijesni podaci o opterećenju elektroenergetske mreže na satnoj razini uzorkovanja.
        
        ### Cilj istraživanja
        Glavi cilj je na temelju povijesnih kalendarskih indikatora (sat, dan u tjednu, mjesec) te meteoroloških uvjeta (temperatura zraka) konstruirati modele sposobne za precizno predviđanje buduće potražnje za električnom energijom izraženom u megavatima (MW).
        """)
    
    with col2:
        st.markdown("### Reprezentativni uzorak strukturiranog vremenskog niza:")
        st.dataframe(df[['Stvarna potrošnja', 'Temperatura', 'Sat', 'Vikend']].head(6))

    st.markdown(
        """
        <div style="
            border: 1px solid rgba(49, 51, 63, 0.2); 
            padding: 15px; 
            border-radius: 4px; 
            margin-top: 20px;
            background-color: transparent;">
            <strong>Napomena o obuhvatu podataka:</strong> Analiza je fokusirana isključivo na potrošnju električne energije unutar mreže. Drugi oblici toplinske energije, poput gradskog centralnog grijanja ili plinske infrastrukture, nisu dio ovog skupa podataka. To objašnjava specifične sezonske razlike u opterećenju, gdje ljetni valovi vrućine uzrokuju znatno brži i oštriji skok potrošnje zbog klimatizacije nego što je to slučaj s potrošnjom struje tijekom zimskih mjeseci.
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- 2. KARTICA: EKSPLORATORNA ANALIZA (EDA) ---
with tab2:
    st.header("Identifikacija sezonskih trendova i ovisnosti značajki")
    
    prikaz_grafikona = st.radio("Odaberite prikaz vizualizacije:", 
                                ["Nelinearni utjecaj temperature (U-krivulja)", 
                                 "Satni profili potrošnje: Radni dan u odnosu na vikend",
                                 "Makro sezonalnost: Mjesečni trendovi kroz godine",
                                 "Distribucija i disperzija potrošnje po danima u tjednu"])
    
    # Globalne postavke za Matplotlib i Seaborn
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. GRAFIKON: U-Krivulja temperature
    if prikaz_grafikona == "Nelinearni utjecaj temperature (U-krivulja)":
        st.subheader("Korelacija temperature zraka i opterećenja mreže")
        
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        df_temp_trend = df.groupby(df['Temperatura'].round()).agg({'Stvarna potrošnja': 'mean'}).reset_index()
        df_temp_filtered = df_temp_trend[df_temp_trend['Temperatura'] >= 0]
        
        sns.scatterplot(data=df_temp_filtered, x='Temperatura', y='Stvarna potrošnja', color='#2b5c5f', s=45, alpha=0.7, ax=ax, label='Izmjereni prosjek')
        
        izracun_polinoma = np.polyfit(df_temp_filtered['Temperatura'], df_temp_filtered['Stvarna potrošnja'], 2)
        polinom = np.poly1d(izracun_polinoma)
        x_trend = np.linspace(df_temp_filtered['Temperatura'].min(), df_temp_filtered['Temperatura'].max(), 100)
        
        ax.plot(x_trend, polinom(x_trend), color='#ae4444', linewidth=2.5, label='Krivulja regresijskog trenda')
        
        ax.set_xlabel("Temperatura (°C)", fontsize=11, color='#333333')
        ax.set_ylabel("Potrošnja energije (MW)", fontsize=11, color='#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        
        g_col1, g_col2, g_col3 = st.columns([1, 2.8, 1])
        with g_col2:
            st.pyplot(fig)
            
        st.markdown("**Interpretacija trenda:** Krivulja poprima oblik parabole s najnižom točkom u zoni ugodnosti (oko 20°C). Progresivan rast temperature prema ljetnim ekstremima uzrokuje strmi uzlazni trend opterećenja mreže, što izravno korelira s masovnim uključivanjem rashladnih sustava.")
        
    # 2. GRAFIKON: Satni profili (Radni dan vs Vikend)
    elif prikaz_grafikona == "Satni profili potrošnje: Radni dan u odnosu na vikend":
        st.subheader("Dnevna dinamika potrošnje pod utjecajem kalendarskih značajki")
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        
        sns.lineplot(data=df, x='Sat', y='Stvarna potrošnja', hue='Vikend', palette=['#3a6073', '#828282'], marker='o', markeredgecolor='white', linewidth=2, ax=ax)
        
        ax.set_xticks(range(0, 24))
        ax.set_xlabel("Sat u danu", fontsize=11, color='#333333')
        ax.set_ylabel("Potrošnja energije (MW)", fontsize=11, color='#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(['Radni dan', 'Vikend'], frameon=False)
        
        g_col1, g_col2, g_col3 = st.columns([1, 2.8, 1])
        with g_col2:
            st.pyplot(fig)
            
        st.markdown("**Interpretacija trenda:** Profil radnog dana reflektira jasnu dvofaznu dinamiku. Izražen pad intenziteta potrošnje između 14:00 i 16:00 sati predstavlja izravnu manifestaciju socioekonomskih specifičnosti i tradicionalne pauze na španjolskom tržištu rada (siesta), tijekom koje dolazi do privremenog gašenja komercijalnih i industrijskih pogona.")

    # 3. GRAFIKON: Makro sezonalnost kroz godine (SADA POTPUNO IDENTIČAN BILJEŽNICI)
    elif prikaz_grafikona == "Makro sezonalnost: Mjesečni trendovi kroz godine":
        st.subheader("Godišnja cikličnost i makro-gibanja potrošnje kroz godine")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        
        df_pomoćni = df.copy()
        df_pomoćni['Godina'] = df_pomoćni.index.year
        df_pomoćni['Mjesec'] = df_pomoćni.index.month
        
        # Grupiranje i računanje prosjeka
        df_monthly = df_pomoćni.groupby(['Godina', 'Mjesec']).agg({'Stvarna potrošnja': 'mean'}).reset_index()
        
        # Koristimo 'viridis' paletu baš kao na slici iz tvoje bilježnice
        sns.lineplot(data=df_monthly, x='Mjesec', y='Stvarna potrošnja', hue='Godina', palette='viridis', marker='s', linewidth=2, ax=ax)
        
        ax.set_xlabel('Mjesec u godini', fontsize=11)
        ax.set_ylabel('Prosječna potrošnja energije (MW)', fontsize=11)
        ax.set_xticks(range(1, 13))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(title='Godina', frameon=False)
        
        g_col1, g_col2, g_col3 = st.columns([1, 2.8, 1])
        with g_col2:
            st.pyplot(fig)
            
        st.markdown("**Interpretacija trenda:** Grafikon zorno prikazuje ponovljive godišnje valove kroz četverogodišnje razdoblje. Potrošnja redovito doseže lokalne vrhunce usred zime (siječanj/veljača) te ekstremne sezonske maksimume u srpnju zbog ljetnih toplinskih valova i hlađenja, dok su proljetni i jesenski mjeseci razdoblja stabilizacije i minimalnog opterećenja.")

    # 4. GRAFIKON: Box plot distribucija po danima
    elif prikaz_grafikona == "Distribucija i disperzija potrošnje po danima u tjednu":
        st.subheader("Statistička disperzija i prisutnost ekstrema po danima")
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        
        df_pomoćni = df.copy()
        dani_imena = {0: 'Pon', 1: 'Uto', 2: 'Sri', 3: 'Čet', 4: 'Pet', 5: 'Sub', 6: 'Ned'}
        df_pomoćni['Dan_u_tjednu'] = df_pomoćni.index.dayofweek.map(dani_imena)
        
        sns.boxplot(data=df_pomoćni, x='Dan_u_tjednu', y='Stvarna potrošnja', palette='pastel', 
                    order=['Pon', 'Uto', 'Sri', 'Čet', 'Pet', 'Sub', 'Ned'], ax=ax)
        
        ax.set_xlabel('Dan u tjednu', fontsize=11)
        ax.set_ylabel('Potrošnja energije (MW)', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        g_col1, g_col2, g_col3 = st.columns([1, 2.8, 1])
        with g_col2:
            st.pyplot(fig)
            
        st.markdown("**Interpretacija trenda:** Kutijasti dijagram (Box plot) precizno vizualizira pad medijana i cjelokupnog kvartilnog raspona potrošnje tijekom subote, a posebno nedjelje, kada se industrijski pogoni gase. Točkice izvan 'brkova' (outlieri) predstavljaju specifične dane anomalija ili ekstremnih klimatskih uvjeta unutar pojedinih dana.")

# --- 3. KARTICA: USPOREDBA S PREDIKCIJAMA ---
with tab3:
    st.header("Vizualizacija povijesnih podataka i validacija predikcija")
    st.markdown("Definirajte željeni vremenski prozor u satima kako biste detaljno analizirali ponašanje modela na testnom skupu podataka.")
    
    broj_sati = st.slider("Vremenski prozor prikaza (u satima):", min_value=24, max_value=24*14, value=24*4, step=24)
    
    odabir_modela = st.selectbox("Selektirajte arhitekturu modela za prikaz:", 
                                 ["Usporedni prikaz oba modela", "Random Forest Regressor", "LSTM neuronska mreža"])
    
    uzorak_prikaza = df.head(broj_sati)
    
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(uzorak_prikaza.index, uzorak_prikaza['Stvarna potrošnja'], label='Stvarna potrošnja (Actual)', color='#1d3557', linewidth=2.5)
    
    if odabir_modela == "Random Forest Regressor" or odabir_modela == "Usporedni prikaz oba modela":
        ax.plot(uzorak_prikaza.index, uzorak_prikaza['Random Forest predikcija'], label='Predviđanje Random Foresta', color='#457b9d', linewidth=1.8, linestyle='--')
        
    if odabir_modela == "LSTM neuronska mreža" or odabir_modela == "Usporedni prikaz oba modela":
        ax.plot(uzorak_prikaza.index, uzorak_prikaza['LSTM predikcija'], label='Predviđanje LSTM-a', color='#e63946', linewidth=1.8, linestyle='-.')
        
    ax.set_title(f"Kronološki prikaz predviđenih vrijednosti naspram stvarnog opterećenja (prozor od {broj_sati} sati)", fontsize=11, pad=15)
    ax.set_xlabel("Datum i sat", fontsize=11)
    ax.set_ylabel("Potrošnja energije (MW)", fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=False, loc='upper right')
    plt.xticks(rotation=15)
    
    p_col1, p_col2, p_col3 = st.columns([0.8, 4.4, 0.8])
    with p_col2:
        st.pyplot(fig)
    
    # --- INTERAKTIVNE KARTICE S METRIKAMA ---
    st.markdown("### 🏆 Performanse modela na testnom skupu podataka")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="R² Score (Preciznost LSTM modela)", 
            value="64.27%", 
            delta="+6.42% vs Random Forest",
            help="Koeficijent determinacije označava koliki postotak varijance podataka naš model uspješno objašnjava."
        )
    with col2:
        st.metric(
            label="MAE (Prosječna greška)", 
            value="1,943.45 MW", 
            delta="-141.54 MW vs RF", 
            delta_color="inverse",
            help="Mean Absolute Error prikazuje prosječno odstupanje modela od stvarne vrijednosti u megavatima."
        )
    with col3:
        st.metric(
            label="RMSE (Kazna za ekstreme)", 
            value="2,708.04 MW", 
            delta="-233.55 MW vs RF", 
            delta_color="inverse",
            help="Root Mean Squared Error strože kažnjava veće promašaje, dajući uvid u stabilnost tijekom vršnih opterećenja."
        )
        
    st.markdown(" ")
    st.markdown("### 📋 Detaljna tablica usporedbe metrika")
    
    df_streamlit_usporedba = pd.DataFrame({
        'Metrika evaluacije': ['MAE (Srednja apsolutna pogreška)', 'RMSE (Korijen srednje kvadratne pogreške)', 'R² Score (Koeficijent determinacije)'],
        'Random Forest Regressor (Baseline)': ["2,084.99 MW", "2,941.59 MW", "57.85%"],
        'LSTM neuronska mreža (30 epoha - Deep Learning)': ["1,943.45 MW", "2,708.04 MW", "64.27%"]
    })
    
    st.dataframe(df_streamlit_usporedba, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Zaključak komparativne analize:** LSTM neuronska mreža (optimizirana na 30 epoha) postiže superiornu razinu ukupne robusnosti s koeficijentom determinacije od **$64.27\%$** te značajno nižom vrijednošću MAE i RMSE metrika u odnosu na bazični Random Forest model. 
    
    Za razliku od Random Foresta koji svaku satnu točku promatra izolirano i unosi lokalni šum u predikcije, rekurentna priroda LSTM mreže uspješno koristi look-back prozor od prethodna 24 sata. To modelu omogućuje dinamičko prepoznavanje vremenske inercije sustava, eliminaciju satnih oscilacija i znatno preciznije praćenje nelinearnih vršnih opterećenja, što potvrđuje drastičan pad RMSE pogreške.
    """)