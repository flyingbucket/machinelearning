import csv
from collections import defaultdict
import math

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class DiscreteNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 平滑参数
        self.class_prior = {}  # P(y)
        self.feature_cond_prob = {}  # P(x_i=v | y)
        self.label_encoders = {}  # 文本特征映射表
        self.classes = []
        self.feature_values = {}  # 每个特征可能出现的取值

    # 文本离散特征编码器
    def fit_label_encoders(self, X):
        # X: list of samples, each is a list of string features
        n_features = len(X[0])
        for i in range(n_features):
            self.label_encoders[i] = {}
            self.feature_values[i] = set()

        for x in X:
            for i, v in enumerate(x):
                if v not in self.label_encoders[i]:
                    self.label_encoders[i][v] = len(self.label_encoders[i])
                self.feature_values[i].add(v)
        for i in range(n_features):
            self.label_encoders[i]["__UNK__"] = -1

    def encode(self, X):
        X_enc = []
        for x in X:
            row = []
            for i, v in enumerate(x):
                row.append(self.label_encoders[i][v])
            X_enc.append(row)
        return X_enc

    def fit(self, X, y):
        # 拟合文本编码器
        self.fit_label_encoders(X)
        X = self.encode(X)

        self.classes = list(set(y))
        n_samples = len(X)
        n_features = len(X[0])

        # 类别计数
        class_count = defaultdict(int)
        # 每个类别、每个特征维度、每个可能取值的计数
        feature_count = defaultdict(
            lambda: [defaultdict(int) for _ in range(n_features)]
        )

        # 统计频数
        for x_row, label in zip(X, y):
            class_count[label] += 1
            for i, v in enumerate(x_row):
                feature_count[label][i][v] += 1

        # 计算先验 P(y)
        for c in self.classes:
            self.class_prior[c] = (class_count[c] + self.alpha) / (
                n_samples + self.alpha * len(self.classes)
            )

        # 计算条件概率 P(x_i=v | y)
        self.feature_cond_prob = defaultdict(list)
        for c in self.classes:
            for i in range(n_features):
                cond = {}
                total_count = sum(feature_count[c][i].values())
                V = len(self.feature_values[i])
                for v_enc in feature_count[c][i]:
                    cond[v_enc] = (feature_count[c][i][v_enc] + self.alpha) / (
                        total_count + self.alpha * V
                    )
                # 对未出现的取值做平滑
                for txt_val, v_enc in self.label_encoders[i].items():
                    if v_enc not in cond:
                        cond[v_enc] = self.alpha / (total_count + self.alpha * V)
                self.feature_cond_prob[c].append(cond)

    # 预测单条
    def predict_one(self, x):
        # 文本转编码
        x = [
            self.label_encoders[i].get(v, -1)  # 未见过的特征值 → -1
            for i, v in enumerate(x)
        ]
        best_class = None
        best_logp = -1e18

        for c in self.classes:
            logp = math.log(self.class_prior[c])
            for i, v_enc in enumerate(x):
                logp += math.log(self.feature_cond_prob[c][i][v_enc])
            if logp > best_logp:
                best_logp = logp
                best_class = c

        return best_class

    # 批量预测
    def predict(self, X):
        return [self.predict_one(x) for x in X]


# 数据加载
def load_data(path):
    X = []
    y = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 选择部分离散特征
            # 你可按需要修改
            feature_row = [
                row["DayOfWeek"],
                row["PdDistrict"],
                row["Descript"],
            ]
            X.append(feature_row)
            y.append(row["Category"])
    return X, y


# 评估
def evaluate(y_true, y_pred):
    print("====== Evaluation ======")

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # precision, recall, f1 for each class (macro means均值)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1-score (macro):  {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


if __name__ == "__main__":
    X_train, y_train = load_data("./EX4/data/train.csv")
    X_test, y_test = load_data("./EX4/data/test.csv")

    nb = DiscreteNB(alpha=1.0)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    evaluate(y_test, y_pred)
