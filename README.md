# Kratkoročno predviđanje potrošnje električne energije

Interaktivna Streamlit web aplikacija namijenjena vizualizaciji, analizi i predviđanju opterećenja elektroenergetskog sustava. Korisnicima omogućuje pregled povijesnih podataka, interaktivnu eksplorativnu analizu ključnih trendova te dinamičku usporedbu i validaciju predikcija naprednog LSTM modela dubokog učenja i bazičnog Random Forest modela na testnom skupu podataka.

## Kako pokrenuti aplikaciju?

Za lokalno pokretanje aplikacije na računalu potrebno je izvršiti sljedeće korake u terminalu:

### Klonirajte repozitorij
```bash
git clone https://github.com/eivelic/streamlit-energy-forecasting.git
```
### Otvorite kloniranu mapu streamlit-energy-forecasting unutar VS Code-a, otvorite novi terminal (engl. New Terminal) i pokrenite sljedeće naredbe:
```
# Instalacija svih potrebnih biblioteka
pip install streamlit tensorflow scikit-learn pandas numpy matplotlib seaborn

# Pokretanje Streamlit aplikacije
streamlit run app.py
```
