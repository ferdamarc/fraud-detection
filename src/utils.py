"""
Funções auxiliares reutilizáveis entre os notebooks do projeto.
"""

import os
import pandas as pd


DATA_DIR = os.environ.get(
    "FRAUD_DATA_DIR",
    "/home/fernando-daniel-marcelino/data/ieee-cis-fraud-detection"
)


def load_data(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os arquivos de treino (transação + identidade) do disco.

    Retorna
    -------
    df_transaction : DataFrame com as transações (590k linhas, ~394 colunas)
    df_identity    : DataFrame com os dados de identidade (141k linhas, ~41 colunas)
    """
    transaction_path = os.path.join(data_dir, "train_transaction.csv")
    identity_path = os.path.join(data_dir, "train_identity.csv")

    print(f"Carregando transações de: {transaction_path}")
    df_transaction = pd.read_csv(transaction_path)
    print(f"  -> {df_transaction.shape[0]:,} linhas, {df_transaction.shape[1]} colunas")

    print(f"Carregando identidade de: {identity_path}")
    df_identity = pd.read_csv(identity_path)
    print(f"  -> {df_identity.shape[0]:,} linhas, {df_identity.shape[1]} colunas")

    return df_transaction, df_identity


def merge_tables(df_transaction: pd.DataFrame, df_identity: pd.DataFrame) -> pd.DataFrame:
    """
    Faz o LEFT JOIN entre transaction e identity pelo campo 'TransactionID'.

    LEFT JOIN porque queremos manter TODAS as transações — nem todas
    têm dados de identidade associados. Os valores ausentes após o merge
    serão tratados no pré-processamento.
    """
    df = df_transaction.merge(df_identity, on="TransactionID", how="left")
    print(f"Dataset após merge: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    return df


def missing_summary(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Retorna um DataFrame com a contagem e percentual de valores ausentes por coluna.

    Parâmetros
    ----------
    threshold : float
        Se > 0, retorna apenas colunas com missing acima desse percentual (0.0 a 1.0).
    """
    missing_count = df.isnull().sum()
    missing_pct = missing_count / len(df)
    summary = (
        pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
    )
    if threshold > 0:
        summary = summary[summary["missing_pct"] >= threshold]
    summary["missing_pct"] = summary["missing_pct"].map("{:.1%}".format)
    return summary
