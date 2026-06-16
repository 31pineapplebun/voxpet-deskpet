"""
RAG 建库脚本 v3 - 多人版 (读 friends.py 花名册, 给每个好友建独立向量库)
=====================================================================
和 v2 的区别:
  - 不再只建一个人, 而是遍历 friends.py 里所有好友
  - 每个人单独一个向量库: friend_db/<英文key>/
  - 昵称/别名从花名册的 rag_names 读, 不用在这里手填

用法:
  python build_rag.py                  给花名册里所有人建库
  python build_rag.py buddy          只给某一个人建库
  python build_rag.py --list           只列出所有 CSV 里的人名 (发现别名用)
"""
import sys
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from pathlib import Path
from collections import Counter

from friends import FRIENDS, get_friend

# ====== 配置 ======
CSV_DIR     = r"D:\微信聊天记录\texts"   # 微信导出的 CSV 都在这 (5个: 私聊们+群聊)
DB_ROOT     = "./friend_db"          # 每个人的库在 friend_db/<key>/
COLLECTION  = "friend_chat"          # 每个库内的 collection 名 (统一, 因为库已按人分开)
WINDOW      = 3
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def scan_csv_files():
    csv_files = sorted(Path(CSV_DIR).glob("*.csv"))
    if not csv_files:
        print(f"❌ {CSV_DIR} 里没有 CSV 文件")
        sys.exit(1)
    return csv_files


def list_all_talkers(csv_files):
    """列出所有 CSV 里的发言者 (发现别名用)"""
    all_talkers = Counter()
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
            df = df[df["type_name"] == "text"]
            for talker, count in df["talker"].value_counts().items():
                all_talkers[talker] += count
        except Exception as e:
            print(f"  WARN {csv_file.name} read fail: {e}")
    print("所有 CSV 里出现的发言者(按发言条数排序):")
    print(f"  {'条数':>8}   昵称")
    print(f"  {'-'*8}   {'-'*30}")
    for talker, count in all_talkers.most_common():
        print(f"  {count:>8}   {talker}")
    print("\n-> 把目标好友的昵称填到 friends.py 里对应人的 rag_names")


def build_one(key, csv_files, embedder):
    """给单个好友建库"""
    f = get_friend(key)
    target_names = f["rag_names"]
    display = f["display_name"]
    db_path = str(Path(DB_ROOT) / key)

    print(f"\n{'='*55}")
    print(f"  建库: {display} (key={key})  昵称={target_names}")
    print(f"{'='*55}")

    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, embedding_function=embedder)

    docs, metas, ids = [], [], []
    id_counter = 0
    hit_sessions = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
        except Exception as e:
            print(f"  WARN {csv_file.name} read fail: {e}")
            continue

        df = df[df["type_name"] == "text"].copy()
        df["msg"] = df["msg"].astype(str).str.strip()
        df = df[df["msg"].str.len() > 0]
        df = df.sort_values("CreateTime").reset_index(drop=True)

        target_count = df["talker"].isin(target_names).sum()
        if target_count == 0:
            continue
        hit_sessions += 1

        records = df[["talker", "msg"]].values.tolist()
        for i, (talker, msg) in enumerate(records):
            if talker not in target_names:
                continue
            ctx_lines = []
            for j in range(max(0, i - WINDOW), i):
                t, m = records[j]
                ctx_lines.append(f"{t}: {m}")
            ctx = "\n".join(ctx_lines)
            full_doc = (ctx + "\n" if ctx else "") + f"{talker}: {msg}"
            docs.append(full_doc)
            metas.append({"reply": msg, "source": csv_file.stem})
            ids.append(f"msg_{id_counter}")
            id_counter += 1

    if not docs:
        print(f"  FAIL 没找到 {target_names} 的任何发言, 跳过")
        print(f"       检查 friends.py 里 {key} 的 rag_names 是否和 CSV 里昵称一致")
        print(f"       (python build_rag.py --list 查看实际昵称)")
        return False

    BATCH = 100
    for k in range(0, len(docs), BATCH):
        collection.add(
            documents=docs[k:k+BATCH],
            metadatas=metas[k:k+BATCH],
            ids=ids[k:k+BATCH],
        )
    print(f"  OK  {display}: 入库 {len(docs)} 条 (来自 {hit_sessions} 个会话) -> {db_path}")
    return True


def main():
    args = sys.argv[1:]

    csv_files = scan_csv_files()
    print(f"找到 {len(csv_files)} 个 CSV 文件")

    if "--list" in args:
        list_all_talkers(csv_files)
        return

    print("加载 embedding 模型 ...")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if args:
        keys = [a for a in args if a in FRIENDS]
        if not keys:
            print(f"FAIL 参数 {args} 不在花名册里。花名册有: {list(FRIENDS.keys())}")
            sys.exit(1)
    else:
        keys = list(FRIENDS.keys())

    print(f"准备给这些人建库: {keys}")

    ok, fail = [], []
    for key in keys:
        if build_one(key, csv_files, embedder):
            ok.append(key)
        else:
            fail.append(key)

    print(f"\n{'='*55}")
    print(f"全部完成: 成功 {len(ok)} 人 {ok}")
    if fail:
        print(f"         失败 {len(fail)} 人 {fail} (多半是 rag_names 对不上)")
    print(f"{'='*55}")

    if ok:
        test_key = ok[0]
        db_path = str(Path(DB_ROOT) / test_key)
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(COLLECTION, embedding_function=embedder)
        print(f"\n检索自测 ({get_friend(test_key)['display_name']}):")
        for q in ["在干嘛呢", "吃饭了吗"]:
            print(f"\n[Q] {q}")
            res = collection.query(query_texts=[q], n_results=2, include=["documents", "metadatas"])
            for doc in res["documents"][0]:
                last = doc.split("\n")[-1]
                print(f"   {last}")


if __name__ == "__main__":
    main()