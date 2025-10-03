import sys
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import jieba
import tensorflow as tf
from flask import Flask, render_template_string, request, jsonify

# 确保中文正常显示
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

models = tf.keras.models
layers = tf.keras.layers
callbacks = tf.keras.callbacks

# 全局变量
max_words = 8000
max_len = 100
min_word_count = 3
le = LabelEncoder()
vocab = {}
model = None

# 初始化模型路径
base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "model"
os.makedirs(model_dir, exist_ok=True)
model_path = model_dir / "LSTM.keras"

# 初始化 Flask 应用
app = Flask(__name__)

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>情感分析系统</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .container { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 150px; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; }
        button { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 4px; font-size: 16px; }
        button:hover { background-color: #45a049; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; font-size: 18px; font-weight: bold; text-align: center; }
        .positive { background-color: #dff0d8; color: #3c763d; }
        .negative { background-color: #f2dede; color: #a94442; }
        .unknown { background-color: #f5f5f5; color: #777; }
        .footer { margin-top: 20px; text-align: center; color: #777; font-size: 14px; }
    </style>
</head>
<body>
    <h1>情感分析系统</h1>
    <div class="container">
        <form id="analysisForm">
            <textarea id="textInput" placeholder="请输入要分析的文本..."></textarea>
            <button type="submit">分析情感</button>
        </form>
        <div id="result" class="result"></div>
    </div>
    <div class="footer">基于LSTM的中文情感分析系统</div>

    <script>
        document.getElementById('analysisForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const text = document.getElementById('textInput').value.trim();
            const resultDiv = document.getElementById('result');

            if (!text) {
                resultDiv.textContent = '请输入文本';
                resultDiv.className = 'result unknown';
                return;
            }

            fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                resultDiv.textContent = '分析结果: ' + data.result;
                if (data.result === 'active') resultDiv.className = 'result positive';
                else if (data.result === 'negative') resultDiv.className = 'result negative';
                else resultDiv.className = 'result unknown';
            })
            .catch(error => {
                resultDiv.textContent = '分析出错: ' + error;
                resultDiv.className = 'result unknown';
            });
        });
    </script>
</body>
</html>
"""


# 数据处理函数
def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if (uchar >= '\u4e00' and uchar <= '\u9fa5'):
        return True
    return False


def reserve_chinese_and_important_chars(content):
    """保留中文、数字和重要标点"""
    content_str = ''
    for i in content:
        if is_chinese(i) or i.isdigit() or i in ['!', '?', '，', '。', ',', '.']:
            content_str += i
    return content_str


def getStopWords():
    """加载停用词表"""
    try:
        stopwords_path = Path(__file__).resolve().parent / 'stopwords.txt'
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print("警告：未找到停用词文件，使用空停用词表")
        return []


def dataParse(text, stop_words):
    label, content = text.split('    ####    ')
    content = reserve_chinese_and_important_chars(content)
    words = jieba.lcut(content)
    words = [i for i in words if i not in stop_words]
    return words, int(label)


# 手动构建词汇表
def build_vocab(data, max_words=8000, min_count=3):
    """构建词汇表，过滤低频词和停用词"""
    stop_words = getStopWords()
    word_count = {}
    for text in data:
        for word in text:
            if word not in stop_words:  # 过滤停用词
                word_count[word] = word_count.get(word, 0) + 1

    # 过滤低频词
    word_count = {word: count for word, count in word_count.items() if count >= min_count}

    # 按词频排序并限制词汇表大小
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    vocab = {}
    # 0 保留给 padding, 1 保留给 unknown words
    for idx, (word, _) in enumerate(sorted_words[:max_words - 2]):
        vocab[word] = idx + 2

    # 添加 padding 和 unknown 标记
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    return vocab


# 手动文本序列化
def texts_to_sequences(texts, vocab, max_len):
    sequences = []
    for text in texts:
        seq = []
        for word in text:
            seq.append(vocab.get(word, vocab['<UNK>']))  # 未登录词使用 <UNK>
        # 截断或填充到固定长度
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq.extend([vocab['<PAD>']] * (max_len - len(seq)))
        sequences.append(seq)
    return np.array(sequences)


def getData(file=None):
    if file is None:
        file_path = Path(__file__).resolve().parent / 'data' / 'data.txt'
    else:
        file_path = Path(file)

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    stop_words = getStopWords()
    all_words = []
    all_labels = []

    for index, row in df.iterrows():
        try:
            content, label = row['review'], row['label']
            content = reserve_chinese_and_important_chars(str(content))
            words = jieba.lcut(content)
            words = [i for i in words if i not in stop_words]
            if len(words) > 0:
                all_words.append(words)
                all_labels.append(label)
        except Exception as e:
            print(f"解析错误: {e}")
            continue

    return all_words, all_labels


def train_model():
    global le, vocab, model, model_path

    # 定义模型路径
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "LSTM.keras"

    # 读取数据集（使用相对路径）
    data_path = base_dir / "data" / "waimai_10k.csv"
    data, label = getData(str(data_path))
    print(f"总样本数: {len(data)}")

    # 检查类别分布
    df = pd.DataFrame({'label': label})
    print("类别分布:")
    print(df['label'].value_counts())

    # 划分数据集
    X_train, X_t, train_y, v_y = train_test_split(data, label, test_size=0.3, random_state=42)
    X_val, X_test, val_y, test_y = train_test_split(X_t, v_y, test_size=0.5, random_state=42)

    # 编码标签
    train_y_encoded = le.fit_transform(train_y)
    val_y_encoded = le.transform(val_y)
    test_y_encoded = le.transform(test_y)

    print("标签映射:", dict(zip(le.classes_, le.transform(le.classes_))))

    # 只用训练集构建词汇表
    vocab = build_vocab(X_train, max_words, min_word_count)
    print(f"词汇表大小: {len(vocab)}")

    # 序列化文本
    train_seq_mat = texts_to_sequences(X_train, vocab, max_len)
    val_seq_mat = texts_to_sequences(X_val, vocab, max_len)
    test_seq_mat = texts_to_sequences(X_test, vocab, max_len)

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(train_y_encoded),
                                         y=train_y_encoded)
    class_weight_dict = dict(zip(np.unique(train_y_encoded), class_weights))
    print("类别权重:", class_weight_dict)

    # 定义模型
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(len(vocab), 128, input_length=max_len)(inputs)
    x = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 回调函数
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.CSVLogger('training_log.csv')  # 保存训练日志
    ]

    # 模型训练
    model.fit(train_seq_mat, train_y_encoded, batch_size=32, epochs=50,  # 训练轮数为50轮
              validation_data=(val_seq_mat, val_y_encoded),
              class_weight=class_weight_dict,
              callbacks=cb)

    # 保存模型
    save_path_str = str(model_path)
    print(f"准备保存模型到: {save_path_str}")
    model.save(save_path_str)
    print(f"模型已保存到: {save_path_str}")

    # 保存词汇表和标签编码器
    with open(model_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"词汇表已保存到: {model_dir / 'vocab.pkl'}")

    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"标签编码器已保存到: {model_dir / 'label_encoder.pkl'}")

    # 评估模型
    test_pre = model.predict(test_seq_mat)
    pred = (test_pre > 0.5).astype(int).flatten()
    real = test_y_encoded

    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='weighted')
    recall = recall_score(real, pred, average='weighted')
    f1 = f1_score(real, pred, average='weighted')

    print("混淆矩阵:")
    print(cv_conf)
    print(f"测试集: acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")


def load_existing_model():
    global model, model_path, vocab, le
    if model is None and model_path and model_path.exists():
        print(f"加载已有模型: {model_path}")
        try:
            # 加载模型
            model = models.load_model(str(model_path))

            # 加载词汇表
            vocab_path = model_path.parent / "vocab.pkl"
            if vocab_path.exists():
                with open(vocab_path, "rb") as f:
                    vocab = pickle.load(f)
                print(f"词汇表已加载，大小: {len(vocab)}")
            else:
                print("警告：未找到词汇表文件")
                return False

            # 加载标签编码器
            le_path = model_path.parent / "label_encoder.pkl"
            if le_path.exists():
                with open(le_path, "rb") as f:
                    le = pickle.load(f)
                print("标签编码器已加载")
            else:
                print("警告：未找到标签编码器文件")
                return False

            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    return model is not None


def dataParse_(content, stop_words):
    content = reserve_chinese_and_important_chars(content)
    words = jieba.lcut(content)
    words = [i for i in words if i not in stop_words]
    return words


def getData_one(text):
    stop_words = getStopWords()
    all_words = []
    content = dataParse_(text, stop_words)
    all_words.append(content)
    return all_words


def predict_(text_o):
    global model, le, vocab, max_len

    if model is None:
        if not load_existing_model():
            print("没有找到模型，正在训练新模型...")
            train_model()
            load_existing_model()

    data_cut = getData_one(text_o)
    t_seq = texts_to_sequences(data_cut, vocab, max_len)

    t_pre = model.predict(t_seq)
    prediction = int((t_pre > 0.5).astype(int).flatten()[0])

    # 映射到原始标签
    try:
        pred_label = le.inverse_transform([prediction])[0]
        # 转换为情感标签
        if pred_label == 0:
            return "negative"
        else:
            return "active"
    except Exception as e:
        print(f"标签映射错误: {e}")
        return "unknown"


# Flask 路由
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'result': '请输入文本喵'})

    try:
        result = predict_(text)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'result': f'分析出错喵: {str(e)}'})


def main():
    # 初始化模型路径
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    os.makedirs(model_dir, exist_ok=True)
    global model_path
    model_path = model_dir / "LSTM.keras"

    # 检查是否已有模型，没有则训练
    if not model_path.exists():
        print("没有找到已有模型，正在训练新模型喵...")
        train_model()
    else:
        print("发现已有模型，将在需要时加载喵")

    print("\n情感分析系统已启动喵！")
    print("请在浏览器中访问 http://127.0.0.1:5000 来使用系统喵！")

    # 启动 Flask 应用
    app.run(debug=True)


if __name__ == "__main__":
    main()