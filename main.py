import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Konfiguracja Pandas
pd.set_option('display.max_columns', None)

# --- Funkcje do wczytywania i dzielenia danych ---
def load_data(file_path):
    """Wczytuje dane z pliku CSV i wykonuje wstępne czyszczenie."""
    df = pd.read_csv(file_path, sep=',', header=0)
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')  # Dodano errors='ignore'
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df

def split_data(df, test_size=0.2, random_state=None):
    """Dzieli DataFrame na zbiór treningowy i testowy."""
    if random_state is not None:
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

    test_count = int(len(df_shuffled) * test_size)
    test_df = df_shuffled[:test_count]
    train_df = df_shuffled[test_count:]
    return train_df, test_df

# --- Definicja wag ---
wages = {
    "age": 0.019818,
    "sex": 0.220780,
    "chest_pain_type": 0.539849,
    "resting_bps": 0.000053,
    "cholesterol": 0.000503,
    "fasting_blood_sugar": 0.245997,
    "resting_ecg": 0.105090,
    "max_heart_rate": 0.006436,
    "exercise_angina": 0.505639,
    "oldpeak": 0.170031,
    "ST_slope": 1.728198
}

# --- Funkcje do tworzenia zbiorów miękkich ---
def dataFieldToSoftSet_wages(df):
    softSet = np.zeros(df.shape, dtype=float)  # Używamy float, ponieważ będziemy mnożyć przez wagi
    i = 0
    for col in df.columns:
        if col == 'age':
            softSet[:, i] = np.where(df[col] <= 46, 1, 0)
            softSet[:, i] *= wages["age"]
        elif col == 'sex':
            softSet[:, i] = df[col].values
            softSet[:, i] *= wages["sex"]
        elif col == 'chest pain type':
            softSet[:, i] = np.where(df[col] <= 2, 1, 0)
            softSet[:, i] *= wages["chest_pain_type"]
        elif col == 'resting bps':
            softSet[:, i] = np.where(df[col] <= 120, 1, 0)
            softSet[:, i] *= wages["resting_bps"]
        elif col == 'cholesterol':
            softSet[:, i] = np.where(df[col] <= 208, 1, 0)
            softSet[:, i] *= wages["cholesterol"]
        elif col == 'fasting blood sugar':
            softSet[:, i] = df[col].values
            softSet[:, i] *= wages["fasting_blood_sugar"]
        elif col == 'resting ecg':
            softSet[:, i] = df[col].values
            softSet[:, i] *= wages["resting_ecg"]
        elif col == 'max heart rate':
            softSet[:, i] = np.where(df[col] <= 130, 1, 0)
            softSet[:, i] *= wages["max_heart_rate"]
        elif col == 'exercise angina':
            softSet[:, i] = df[col].values
            softSet[:, i] *= wages["exercise_angina"]
        elif col == 'oldpeak':
            softSet[:, i] = np.where(df[col] < 0, 1, 0)
            softSet[:, i] *= wages["oldpeak"]
        elif col == 'ST slope':
            softSet[:, i] = np.where(df[col] <= 1, 1, 0)
            softSet[:, i] *= wages["ST_slope"]
        i += 1
    return softSet

def dataFieldToSoftSet(df):
    """Tworzy zbiór miękki bez uwzględnienia wag."""
    soft_set = np.zeros(df.shape, dtype=float)
    for i, col in enumerate(df.columns):
        if col == 'age':
            soft_set[:, i] = np.where(df[col] <= 46, 1, 0)
        elif col == 'sex':
            soft_set[:, i] = df[col].values
        elif col == 'chest pain type':
            soft_set[:, i] = np.where(df[col] <= 2, 1, 0)
        elif col == 'resting bps':
            soft_set[:, i] = np.where(df[col] <= 120, 1, 0)
        elif col == 'cholesterol':
            soft_set[:, i] = np.where(df[col] <= 208, 1, 0)
        elif col == 'fasting blood sugar':
            soft_set[:, i] = df[col].values
        elif col == 'resting ecg':
            soft_set[:, i] = df[col].values
        elif col == 'max heart rate':
            soft_set[:, i] = np.where(df[col] <= 130, 1, 0)
        elif col == 'exercise angina':
            soft_set[:, i] = df[col].values
        elif col == 'oldpeak':
            soft_set[:, i] = np.where(df[col] < 0, 1, 0)
        elif col == 'ST slope':
            soft_set[:, i] = np.where(df[col] <= 1, 1, 0)
    return soft_set

def sum_soft_set_rows(soft_set):
    """Sumuje wartości w wierszach zbioru miękkiego."""
    return np.sum(soft_set, axis=1)

# --- Funkcje do analizy i diagnozy ---
def analyze_training_data(train_df, sum_column, target_column='target'):
    """Analizuje dane treningowe, aby określić zakres kwartyli dla pozytywnej klasy."""
    positive_cases = train_df[train_df[target_column] == 1][sum_column]
    return positive_cases.quantile([0.25, 0.75])

def diagnose_test_data(test_df, ranges, sum_column):
    """Diagnozuje dane testowe na podstawie zakresu kwartyli."""
    lower_bound, upper_bound = ranges[0.25], ranges[0.75]
    test_df['diagnoza'] = np.where((test_df[sum_column] >= lower_bound) & (test_df[sum_column] <= upper_bound), 1, 0)
    return test_df

# --- Funkcje do ewaluacji modelu ---
def evaluate_model(y_true, y_pred):
    """Ewaluuje model klasyfikacji."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    confusion = np.array([[tn, fp], [fn, tp]])
    return accuracy, recall, precision, f1, confusion

# --- Funkcje klasyfikatora Naiwnego Bayesa (bez sklearn) ---
def fit_gaussian_nb(X, y):
    classes = np.unique(y)
    means = {}
    stds = {}
    priors = {}

    for c in classes:
        X_c = X[y == c]
        means[c] = X_c.mean(axis=0)
        stds[c] = np.where(X_c.std(axis=0) == 0, 1e-6, X_c.std(axis=0))
        priors[c] = X_c.shape[0] / X.shape[0]

    return classes, means, stds, priors

def gaussian_prob(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def predict_gaussian_nb(X, classes, means, stds, priors):
    predictions = []
    for x in X.values:
        probs = {}
        for c in classes:
            prob = np.prod(gaussian_prob(x, means[c], stds[c])) * priors[c]
            probs[c] = prob
        predictions.append(max(probs, key=probs.get))
    return np.array(predictions)

def plot_confusion_matrix(cm, labels=["0", "1"], title="Macierz Pomylek"):
    plt.figure(figsize=(6, 5))

    # Normalizacja do wartości procentowych (opcjonalnie)
    cm_percent = cm / cm.sum() * 100
    cm_annot = np.array([[f"{val}\n({perc:.1f}%)" for val, perc in zip(row_vals, row_perc)]
                         for row_vals, row_perc in zip(cm, cm_percent)])

    ax = sns.heatmap(cm_percent, annot=cm_annot, fmt="", cmap="YlOrRd_r",
                     xticklabels=labels, yticklabels=labels, cbar=True, linewidths=0.5, linecolor='gray')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Przewidywane", fontsize=12)
    plt.ylabel("Rzeczywiste", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def get_k_fold_splits(df, k=5, random_state=None):
    """Zwraca listę k par (train_df, test_df) dla k-fold cross-validation."""
    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        df = df.sample(frac=1).reset_index(drop=True)

    fold_size = len(df) // k
    folds = []

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i != k - 1 else len(df)
        test_df = df.iloc[test_start:test_end]
        train_df = pd.concat([df.iloc[:test_start], df.iloc[test_end:]])
        folds.append((train_df.reset_index(drop=True), test_df.reset_index(drop=True)))

    return folds

# --- Funkcja do K-Fold dla trzech modeli ---
def k_fold_cross_validation(df, k=5):
    folds = get_k_fold_splits(df.copy(), k=k, random_state=42)
    metrics_soft = []
    metrics_soft_w = []
    metrics_nb = []

    features = ['age', 'sex', 'chest pain type', 'resting bps', 'cholesterol',
                'fasting blood sugar', 'resting ecg', 'max heart rate',
                'exercise angina', 'oldpeak', 'ST slope']

    for fold_idx, (train_df, test_df) in enumerate(folds, 1):
        # --- Zbiór miękki bez wag ---
        soft_train = dataFieldToSoftSet(train_df.drop('target', axis=1))
        soft_test = dataFieldToSoftSet(test_df.drop('target', axis=1))

        train_df['suma_soft'] = sum_soft_set_rows(soft_train)
        test_df['suma_soft'] = sum_soft_set_rows(soft_test)

        ranges_soft = analyze_training_data(train_df, 'suma_soft')
        test_soft_diag = diagnose_test_data(test_df.copy(), ranges_soft, 'suma_soft')
        m_soft = evaluate_model(test_soft_diag['target'].values, test_soft_diag['diagnoza'].values)
        metrics_soft.append(m_soft[:4])  # (accuracy, recall, precision, f1)

        # --- Zbiór miękki z wagami ---
        soft_w_train = dataFieldToSoftSet_wages(train_df.drop('target', axis=1))
        soft_w_test = dataFieldToSoftSet_wages(test_df.drop('target', axis=1))

        train_df['suma_soft_w'] = sum_soft_set_rows(soft_w_train)
        test_df['suma_soft_w'] = sum_soft_set_rows(soft_w_test)

        ranges_soft_w = analyze_training_data(train_df, 'suma_soft_w')
        test_soft_w_diag = diagnose_test_data(test_df.copy(), ranges_soft_w, 'suma_soft_w')
        m_soft_w = evaluate_model(test_soft_w_diag['target'].values, test_soft_w_diag['diagnoza'].values)
        metrics_soft_w.append(m_soft_w[:4])

        # --- Naiwny Bayes ---
        classes_nb, means_nb, stds_nb, priors_nb = fit_gaussian_nb(train_df[features], train_df['target'])
        pred_nb = predict_gaussian_nb(test_df[features], classes_nb, means_nb, stds_nb, priors_nb)
        m_nb = evaluate_model(test_df['target'], pred_nb)
        metrics_nb.append(m_nb[:4])


    def avg_metrics(metrics_list):
        metrics_array = np.array(metrics_list)
        return metrics_array.mean(axis=0)

    print(f"\n=== ŚREDNIE WYNIKI -kFOLD dla k = {k} === ")
    print("\nZbiór Miękki Bez Wag:")
    acc, rec, prec, f1 = avg_metrics(metrics_soft)
    print(f"Accuracy: {acc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, F1-score: {f1:.4f}")

    plot_confusion_matrix(m_soft_w[4], labels=["0", "1"],
                          title=f"SoftSet z Wagami – Fold {fold_idx}")  # MACIERZ POMYlEK

    print("\nZbiór Miękki z Wagami:")
    acc, rec, prec, f1 = avg_metrics(metrics_soft_w)
    print(f"Accuracy: {acc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, F1-score: {f1:.4f}")
    plot_confusion_matrix(m_soft[4], labels=["0", "1"], title=f"SoftSet Bez Wag – Fold {fold_idx}")  # MACIERZ POMYlEK

    print("\nNaive Bayes:")
    acc, rec, prec, f1 = avg_metrics(metrics_nb)
    print(f"Accuracy: {acc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, F1-score: {f1:.4f}")
    plot_confusion_matrix(m_nb[4], labels=["0", "1"], title=f"Naive Bayes – Fold {fold_idx}")  # MACIERZ POMYlEK

# --- Glówna część kodu ---
if __name__ == "__main__":
    file_path = 'Dataset Heart Disease.csv'
    df = load_data(file_path)

    for i in range(5,16,5):
        k_fold_cross_validation(df.copy(), k=i)


# --- WYKRESY DO ANALIZY BAZY DANYCH ---#
def plot_heart_disease_analysis(ax, data: pd.DataFrame, age_col: str, disease_col: str):
    # Definiowanie przedzialów wiekowych
    age_bins = [ 25, 35, 45, 55, 65, 100]
    age_labels = ["25-35", "35-45", "45-55", "55-65", "65+"]
    data['AgeGroup'] = pd.cut(data[age_col], bins=age_bins, labels=age_labels, right=False)

    # Zliczanie osób chorych i zdrowych w każdej grupie wiekowej
    grouped = data.groupby(['AgeGroup', disease_col], observed=False).size().unstack(fill_value=0)

    grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Tworzenie wykresu slupkowego
    grouped.plot(kind='bar', colormap='coolwarm', ax=ax)

    ax.set_xlabel("Grupa wiekowa")
    ax.set_ylabel("Liczba osób", fontsize=10)  # Zmniejszenie czcionki
    ax.set_title("Porównanie liczby i procentu chorych i zdrowych w grupach wiekowych", fontsize=10)
    ax.legend(["Zdrowi", "Chorzy"], loc="upper left")

    # Dodanie wartości liczbowych i procentowych nad slupkami
    for i in range(len(grouped)):
        for j, value in enumerate(grouped.iloc[i]):
            percent_value = grouped_percent.iloc[i, j]  # Pobranie wartości procentowej
            ax.text(i, value + 1, f"{value} ({percent_value:.1f}%)", ha='center', fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.7)

def plot_gender_heart_disease(ax, data: pd.DataFrame, gender_col: str, disease_col: str):
    # Obliczanie procentu osób chorych w każdej grupie plci
    gender_counts = data.groupby(gender_col, observed=False)[disease_col].mean() * 100
    gender_labels = {0: "Kobiety", 1: "Mężczyźni"}

    # Tworzenie wykresu
    sns.barplot(ax=ax, x=[gender_labels[g] for g in gender_counts.index], y=gender_counts.values,
                hue=[gender_labels[g] for g in gender_counts.index], palette='coolwarm', legend=False)
    ax.set_xlabel("Pleć")
    ax.set_ylabel("Procent osób z chorobą serca")
    ax.set_title("Porównanie liczby chorych kobiet i mężczyzn", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(gender_counts.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

def plot_exercise_angina_heart_disease(ax, data: pd.DataFrame, angina_col: str, disease_col: str):
    # Zliczanie osób z dusznicą bolesną podzielonych na chorych i zdrowych
    angina_counts = data.groupby([angina_col, disease_col]).size().unstack(fill_value=0)
    # Tworzenie wykresu
    angina_counts.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
    ax.set_xlabel("Wystepowanie duszności (0 = Nie wystepują, 1 = Występują)")
    ax.set_ylabel("Liczba osób")
    ax.set_title("Liczba osób z dusznościami a choroba serca", fontsize=10)
    ax.legend(["Zdrowi", "Chorzy"])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Tworzenie wykresów jednocześnie
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

df_columns = df.columns
if 'age' in df_columns and 'target' in df_columns:
    plot_heart_disease_analysis(axes[0], df, 'age', 'target')
else:
    axes[0].set_title("Brak danych dla wieku i choroby serca")

if 'sex' in df_columns and 'target' in df_columns:
    plot_gender_heart_disease(axes[1], df, 'sex', 'target')
else:
    axes[1].set_title("Brak danych dla plci i choroby serca")

if 'exercise angina' in df_columns and 'target' in df_columns:
    plot_exercise_angina_heart_disease(axes[2], df, 'exercise angina', 'target')
else:
    axes[2].set_title("Brak danych dla dusznicy wysilkowej i choroby serca")
plt.tight_layout()
plt.show()