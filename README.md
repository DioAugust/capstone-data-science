<p align="center">
  <img src="data/logo.png" alt="Logo do Projeto" width="200">
</p>

# Jala Scholar - Dashboard de Busca de Documentação Técnica

## 📖 Descrição do Projeto

O **Jala Scholar** é um dashboard interativo desenvolvido em Python utilizando o framework Streamlit. Ele permite realizar buscas semânticas em uma coleção de documentos técnicos, visualizar os resultados de forma interativa e explorar clusters de documentos com base em embeddings semânticos. O objetivo principal é facilitar a navegação e a análise de grandes volumes de documentação técnica.

## 🚀 Funcionalidades

- **Busca Semântica**: Utilize consultas em linguagem natural para encontrar documentos relevantes com base em similaridade semântica.
- **Visualização de Documentos**: Exiba o conteúdo dos documentos encontrados
- **Visualização de Clusters**: Explore agrupamentos de documentos utilizando técnicas de redução de dimensionalidade e clustering.

## 🛠️ Algoritmos e Modelos Utilizados

1. **Embeddings Semânticos**:

   - Os documentos são representados como vetores em um espaço de alta dimensionalidade utilizando embeddings gerados por modelos de linguagem pré-treinados, como BERT ou Sentence Transformers.

2. **Busca Semântica**:

   - A similaridade entre a consulta e os documentos é calculada utilizando a métrica de similaridade de cosseno.

3. **Redução de Dimensionalidade**:

   - Técnicas como t-SNE, UMAP e PCA são utilizadas para projetar os embeddings em um espaço de 2D ou 3D para visualização.

4. **Clustering**:
   - Algoritmos como K-Means são aplicados para agrupar documentos com base em seus embeddings semânticos.

## 📚 Bibliotecas Utilizadas

- **Streamlit**: Para criação do dashboard interativo.
- **Sentence Transformers**: Para geração de embeddings semânticos.
- **Scikit-learn**: Para clustering e redução de dimensionalidade.
- **Matplotlib/Seaborn**: Para visualização de dados.
- **Pandas**: Para manipulação de dados tabulares.

## 📁 Estrutura do Projeto

```bash
capstone-main/
├── app/
│   ├── dashboard.py  # Código principal do dashboard
├── data/
│   ├── processed/    # Documentos processados
│   ├── raw/          # Documentos brutos
├── src/
│   ├── search/       # Algoritmos de busca semântica
│   ├── visualization/ # Funções de visualização
├── requirements.txt  # Dependências do projeto
├── README.md         # Este arquivo
```

## ▶️ Como Iniciar o Projeto

1. **Pré-requisitos**:

   - Python 3.10
   - Pip para gerenciamento de pacotes

2. **Instalação**:

   - Clone o repositório:

     ```bash
     git clone <URL_DO_REPOSITORIO>
     cd capstone-main
     ```

   - Instale as dependências:

     ```bash
     pip install -r requirements.txt
     ```

3. **Preparação dos Dados**:

   - O programa baixará automaticamente os arquivos do GitHub caso você não tenha eles. Fique tranquilo!
   - Basta iniciar o arquivo main.py

4. **Executando o Dashboard**:

   - Certifique-se de estar na raíz do projeto.
   - Inicie o arquivo main.py com o comando:

     ```bash
     python main.py
     ```

   - Acesse o dashboard no navegador em `http://localhost:8501`.
   - **PS:** Caso você não tenha os arquivos baixados e pré-processados, a execução inicial pode demorar alguns vários minutos.

## 👨‍💻 Autor

- Diogines Barbosa
