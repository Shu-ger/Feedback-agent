import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import faiss
import pickle


CSV_PATH = 'feedback.csv' # שם הקובץ
EMB_MODEL = 'all-MiniLM-L6-v2' # מהירות טובה ואיכות סבירה
EMB_FILE = 'embeddings.npy'
DF_FILE = 'df_processed.pkl'
FAISS_INDEX_FILE = 'faiss.index'
TOPIC_MODEL_FILE = 'bertopic_model.pkl'


# --------------------
# 1. טענת CSV
# --------------------
print('Loading CSV...')
df = pd.read_csv(CSV_PATH, encoding='utf-8')
df.columns = [c.strip() for c in df.columns]
# התאמות למבנה מצופה
if 'Text' not in df.columns:
# ננסה למצוא עמודה שמכילה טקסט
    possible = [c for c in df.columns if 'text' in c.lower()]
    if possible:
        df = df.rename(columns={possible[0]: 'Text'})
    else:
        raise Exception('No Text column found in CSV')


if 'Level' not in df.columns:
    possible = [c for c in df.columns if 'level' in c.lower() or 'score' in c.lower()]
    if possible:
        df = df.rename(columns={possible[0]: 'Level'})
    else:
        df['Level'] = None

# --------------------
# 2. קדם-עיבוד טקסט עברי
# --------------------
HEB_STOPWORDS = set([
'של', 'ה', 'ו', 'על', 'עם', 'את', 'לה', 'כי', 'או', 'גם', 'אם', 'לא', 'יש', 'מה', 'זה',
])
# שמירה על מקור
original_count = len(df)
df = df[['Text','Level']].dropna(subset=['Text']).reset_index(drop=True)
print(f'Loaded {len(df)} rows (dropped {original_count-len(df)} empty).')

def normalize_hebrew(text):
    text = str(text)
    text = text.lower()
# הסרת ניקוד
    text = re.sub(r'[\u0591-\u05C7]', '', text)
# הסרת סימני פיסוק מיוחדים (משאירה אותיות ומספרים ומרחבים)
    text = re.sub(r'[^\w\s\u05D0-\u05EA]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
# הסרת stopwords פשוטה
    tokens = [t for t in text.split() if t not in HEB_STOPWORDS]
    return ' '.join(tokens)


print('Normalizing texts...')
df['Text_norm'] = df['Text'].apply(normalize_hebrew)


# --------------------
# 3. embeddings
# --------------------
print('Loading embedding model...')
model = SentenceTransformer(EMB_MODEL)
texts = df['Text_norm'].tolist()
print('Encoding embeddings... (this may take a while)')
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


# שמירה
np.save(EMB_FILE, embeddings)
with open(DF_FILE, 'wb') as f:
    pickle.dump(df, f)
print('Saved embeddings and processed dataframe.')


# --------------------
# 4. Topic modeling (BERTopic)
# --------------------
print('Fitting BERTopic (this may take time)...')
topic_model = BERTopic(language='multilingual')
topics, probs = topic_model.fit_transform(df['Text_norm'].tolist(), embeddings)


df['topic'] = topics


with open(TOPIC_MODEL_FILE, 'wb') as f:
    pickle.dump(topic_model, f)
print('Saved topic model. Topics info:')
print(topic_model.get_topic_info().head(10))


# --------------------
# 5. FAISS index (נרמל ונבנה)
# --------------------
print('Building FAISS index...')
# נרמל
from numpy.linalg import norm
emb_norm = embeddings.copy()
for i in range(emb_norm.shape[0]):
    n = norm(emb_norm[i])
    if n>0:
        emb_norm[i] = emb_norm[i]/n


d = emb_norm.shape[1]
index = faiss.IndexFlatIP(d) # inner product on normalized vecs ~ cosine
index.add(emb_norm)
faiss.write_index(index, FAISS_INDEX_FILE)
print('FAISS index saved.')


print('MVP pipeline finished successfully.')