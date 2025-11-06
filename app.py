# ============================================
# 🏀 NBA AI ANALYZER - Base Inicial de Projeto
# ============================================

# Este código é um exemplo prático e didático.
# Ele coleta estatísticas da NBA, organiza em um DataFrame
# e faz uma análise simples usando IA para identificar jogadores
# em destaque nos últimos jogos.

# ⚙️ Etapas:
# 1. Importar bibliotecas
# 2. Buscar dados de jogadores da NBA
# 3. Organizar e analisar os dados
# 4. Exibir resultados e tendências
# ============================================

# 1️⃣ Importando bibliotecas
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# 2️⃣ Coletando dados públicos da NBA
# (usamos uma API aberta do site balldontlie.io)
url = "https://www.balldontlie.io/api/v1/stats?seasons[]=2025&per_page=50"
response = requests.get(url)
data = response.json()

# Convertendo os dados em tabela
stats = []
for item in data["data"]:
    player = item["player"]
    team = item["team"]
    stats.append({
        "Jogador": f"{player['first_name']} {player['last_name']}",
        "Time": team["full_name"],
        "Pontos": item["pts"],
        "Assistências": item["ast"],
        "Rebotes": item["reb"],
        "Minutos": item["min"]
    })

df = pd.DataFrame(stats)

# 3️⃣ Normalizando e aplicando uma análise simples de cluster (IA)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Pontos", "Assistências", "Rebotes"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Grupo de Desempenho"] = kmeans.fit_predict(scaled)

# 4️⃣ Exibindo os resultados
print("📊 Análise de Desempenho (IA aplicada a estatísticas da NBA)")
print("-" * 60)
print(df.sort_values(by="Pontos", ascending=False).head(10))
print("-" * 60)
print("✅ Agrupamento de desempenho concluído com sucesso!")

# 5️⃣ (Opcional) Salvar resultado como CSV
df.to_csv("nba_ai_analise.csv", index=False)
print("📁 Arquivo salvo: nba_ai_analise.csv")

# ============================================
# 🔮 Próximos passos:
# - Integrar odds e previsões (usando APIs de apostas)
# - Criar interface com Streamlit ou FastAPI
# - Automatizar análise diária no Google Colab ou Replit
# ============================================
