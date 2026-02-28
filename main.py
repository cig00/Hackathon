# ============================================================
#  PRODUCT COMBINATION OPTIMIZER  –  FastAPI Service
#  Datasets : rep_s_00191_SMRY_labelled.csv  (DS-191)
#             REP_S_00502_labelled.csv        (DS-502)
#  Goal     : Identify optimal product combinations based on
#             customer purchasing patterns
# ============================================================

# ── 0. Dependencies ──────────────────────────────────────────

import os, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# ML / Stats
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing     import TransactionEncoder
from sklearn.preprocessing     import LabelEncoder, MinMaxScaler
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (mean_squared_error,
                                       mean_absolute_error)
from sklearn.cluster           import KMeans
from sklearn.metrics           import silhouette_score
from sklearn.decomposition     import PCA

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR   = Path("Data Preparation")
OUTPUT_DIR = Path("Model Output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Conut AI – Chief of Operations Agent",
    description="AI-driven operational intelligence for Conut sweets & beverages.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State (loaded once at startup) ────────────────────
STATE = {}


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 ─ DATA INGESTION & CLEANING                     ║
# ╚══════════════════════════════════════════════════════════════╝

def load_ds191(path: Path) -> pd.DataFrame:
    """
    Loads the labelled DS-191 file.
    Handles: repeated headers, metadata rows, comma-embedded numbers,
             empty barcodes, malformed totals.
    """
    raw = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    raw.columns = [c.strip() for c in raw.columns]

    header_pattern = re.compile(
        r"Branch|Division|Group|Total by|Description|30-Jan|"
        r"REP_S|Copyright|www\.|Years:|Page",
        re.IGNORECASE
    )
    mask = raw["Description"].apply(
        lambda x: bool(re.search(header_pattern, str(x)))
    )
    raw = raw[~mask].copy()
    raw = raw[raw["Description"].notna() & (raw["Description"].str.strip() != "")]

    def clean_num(s):
        s = str(s).replace('"', '').replace("'", "").strip()
        s = re.sub(r",(?=\d{3})", "", s)
        try:
            return float(s)
        except ValueError:
            return np.nan

    raw["Qty"]          = raw["Qty"].apply(clean_num)
    raw["Total_Amount"] = raw["Total_Amount"].apply(clean_num)
    raw = raw.dropna(subset=["Qty", "Total_Amount"])
    raw = raw[raw["Qty"] > 0].copy()

    for col in ["Branch", "Division", "Group", "Description"]:
        raw[col] = (raw[col]
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.upper())

    raw["Unit_Price_Ratio"] = raw["Total_Amount"] / raw["Qty"]
    print(f"  DS-191 loaded  : {len(raw):,} rows  "
          f"| unique items : {raw['Description'].nunique()}")
    return raw.reset_index(drop=True)


def load_ds502(path: Path) -> pd.DataFrame:
    """
    Loads the labelled DS-502 file.
    Handles: ITEM vs TOTAL rows, repeated headers, signed quantities,
             anonymised customer names.
    """
    raw = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[raw["Row_Type"] == "ITEM"].copy()

    def clean_num(s):
        s = str(s).replace('"', '').replace("'", "").strip()
        s = re.sub(r",(?=\d{3})", "", s)
        try:
            return float(s)
        except ValueError:
            return np.nan

    raw["Qty"]   = raw["Qty"].apply(clean_num)
    raw["Price"] = raw["Price"].apply(clean_num)
    raw = raw[raw["Qty"] > 0].copy()
    raw = raw[raw["Price"] >= 0].copy()
    raw = raw.dropna(subset=["Qty", "Price"])

    for col in ["Branch", "Customer_Name", "Description"]:
        raw[col] = (raw[col]
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.upper())

    option_pattern = re.compile(
        r"NO |PRESSED|REGULAR\.|FULL FAT|ICED$|HOT$|DECAF|"
        r"TAKE AWAY|SEND CUTLERY|DONT SEND|LACTOSE|SKIMMED|"
        r"WHIPPED CREAM\.\.\.|NO WHIPPED|NO CINNAMON|NO MARSH|"
        r"NUTELLA SPREAD|LOTUS SPREAD|WHITE CHOC.*SPREAD|PISTACHIO SPREAD|"
        r"SWITCH TO ICE CREAM|PISTACHIO TOPPING",
        re.IGNORECASE
    )
    raw = raw[
        ~((raw["Price"] == 0) &
          raw["Description"].str.contains(option_pattern, regex=True))
    ].copy()

    raw["Revenue_Ratio"] = raw["Price"] / (raw["Price"].max() + 1e-9)
    print(f"  DS-502 loaded  : {len(raw):,} rows  "
          f"| unique items : {raw['Description'].nunique()}  "
          f"| customers    : {raw['Customer_Name'].nunique()}")
    return raw.reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 2 ─ FEATURE ENGINEERING                           ║
# ╚══════════════════════════════════════════════════════════════╝

def build_features(ds191: pd.DataFrame, ds502: pd.DataFrame):
    print("\n[2] Feature Engineering …")

    # DS-191 item-level features
    item_stats_191 = (
        ds191.groupby(["Division", "Group", "Description"])
        .agg(
            Total_Qty      = ("Qty",              "sum"),
            Total_Revenue  = ("Total_Amount",     "sum"),
            Avg_Unit_Price = ("Unit_Price_Ratio", "mean"),
            Branch_Count   = ("Branch",           "nunique"),
        )
        .reset_index()
    )
    item_stats_191["Popularity_Score"] = (
        item_stats_191["Total_Qty"].rank(pct=True)
    )
    item_stats_191["Revenue_Ratio"] = (
        item_stats_191["Total_Revenue"] /
        item_stats_191["Total_Revenue"].sum()
    )
    item_stats_191.to_csv(OUTPUT_DIR / "item_stats_191.csv", index=False)

    # DS-502 basket construction
    baskets_502 = (
        ds502.groupby(["Branch", "Customer_Name"])["Description"]
        .apply(list)
        .reset_index()
        .rename(columns={"Description": "Basket"})
    )
    baskets_502 = baskets_502[baskets_502["Basket"].apply(len) > 1]

    # DS-191 branch-level baskets
    baskets_191 = (
        ds191.groupby(["Branch", "Division"])["Description"]
        .apply(list)
        .reset_index()
        .rename(columns={"Description": "Basket"})
    )
    baskets_191 = baskets_191[baskets_191["Basket"].apply(len) > 1]

    # Combined basket pool
    all_baskets = pd.concat(
        [baskets_191[["Basket"]], baskets_502[["Basket"]]],
        ignore_index=True
    )
    print(f"  Total baskets for mining : {len(all_baskets):,}")
    return item_stats_191, all_baskets


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 3 ─ ASSOCIATION RULE MINING (FP-GROWTH)           ║
# ╚══════════════════════════════════════════════════════════════╝

def mine_rules(all_baskets: pd.DataFrame):
    print("\n[3] Association Rule Mining (FP-Growth) …")

    te = TransactionEncoder()
    te_array  = te.fit_transform(all_baskets["Basket"].tolist())
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    MIN_SUPPORT    = 0.05
    MIN_CONFIDENCE = 0.30
    MIN_LIFT       = 1.2

    frequent_itemsets = fpgrowth(
        basket_df, min_support=MIN_SUPPORT, use_colnames=True
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    print(f"  Frequent itemsets found  : {len(frequent_itemsets):,}")

    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=MIN_LIFT
    )
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: " + ".join(sorted(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: " + ".join(sorted(x))
    )
    rules["Combo_Score"] = (
        rules["lift"] * rules["confidence"] * rules["support"]
    )
    rules = rules.sort_values("Combo_Score", ascending=False)
    rules.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)

    print(f"  Rules extracted          : {len(rules):,}")
    return rules, frequent_itemsets


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 4 ─ NEURAL COLLABORATIVE FILTERING (NCF)          ║
# ╚══════════════════════════════════════════════════════════════╝

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users  = torch.LongTensor(df["user_id"].values)
        self.items  = torch.LongTensor(df["item_id"].values)
        self.scores = torch.FloatTensor(df["Score"].values)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.scores[idx]


class NCFModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, hidden=[64, 32]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        layers = []
        in_dim = emb_dim * 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp    = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)
        return self.sigmoid(self.mlp(x)).squeeze()


def build_ncf(ds502: pd.DataFrame):
    print("\n[4] Neural Collaborative Filtering …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interactions = (
        ds502.groupby(["Customer_Name", "Description"])
        .agg(
            Interaction_Score = ("Revenue_Ratio", "sum"),
            Frequency         = ("Qty",           "sum")
        )
        .reset_index()
    )
    interactions["Score"] = np.log1p(
        interactions["Interaction_Score"] * interactions["Frequency"]
    )
    scaler = MinMaxScaler()
    interactions["Score"] = scaler.fit_transform(interactions[["Score"]])

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    interactions["user_id"] = user_enc.fit_transform(interactions["Customer_Name"])
    interactions["item_id"] = item_enc.fit_transform(interactions["Description"])

    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()
    print(f"  Users : {n_users}  |  Items : {n_items}")

    train_df, val_df = train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(
        InteractionDataset(train_df), batch_size=512, shuffle=True
    )
    val_loader = DataLoader(
        InteractionDataset(val_df), batch_size=512
    )

    EPOCHS   = 20
    EMB_DIM  = 32
    LR       = 1e-3

    model     = NCFModel(n_users, n_items, emb_dim=EMB_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        for u, i, s in train_loader:
            u, i, s = u.to(device), i.to(device), s.to(device)
            optimizer.zero_grad()
            pred = model(u, i)
            loss = criterion(pred, s)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        train_losses.append(t_loss / len(train_loader))

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for u, i, s in val_loader:
                u, i, s = u.to(device), i.to(device), s.to(device)
                pred   = model(u, i)
                v_loss += criterion(pred, s).item()
        val_losses.append(v_loss / len(val_loader))
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"Train MSE: {train_losses[-1]:.4f}  "
                  f"Val MSE: {val_losses[-1]:.4f}")

    # Save model
    torch.save(model.state_dict(), OUTPUT_DIR / "ncf_model.pt")

    # Item embeddings → PCA
    model.eval()
    with torch.no_grad():
        item_embeddings = (
            model.item_emb.weight.cpu().numpy()
        )
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(item_embeddings)
    item_names = item_enc.classes_

    return (model, user_enc, item_enc, interactions,
            n_items, device, item_embeddings,
            coords, item_names, train_losses, val_losses,
            val_loader)


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 5 ─ ITEM SIMILARITY (COSINE)                      ║
# ╚══════════════════════════════════════════════════════════════╝

def build_similarity(item_embeddings, item_names):
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(item_embeddings)
    sim_df     = pd.DataFrame(
        sim_matrix, index=item_names, columns=item_names
    )
    return sim_df


def top_k_similar(item_name, sim_df, k=5):
    if item_name not in sim_df.index:
        return pd.Series(dtype=float)
    return (
        sim_df[item_name]
        .drop(index=item_name)
        .nlargest(k)
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 6 ─ CUSTOMER RECOMMENDATIONS                      ║
# ╚══════════════════════════════════════════════════════════════╝

def recommend_combinations(
    customer_name: str,
    model, user_enc, item_enc,
    interactions, n_items, device, sim_df,
    top_n: int = 5
) -> pd.DataFrame:
    try:
        uid = user_enc.transform([customer_name.upper()])[0]
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customer_name}' not found."
        )

    bought = set(
        interactions.loc[
            interactions["user_id"] == uid, "item_id"
        ].tolist()
    )
    not_bought = [i for i in range(n_items) if i not in bought]

    model.eval()
    with torch.no_grad():
        u_tensor = torch.LongTensor([uid] * len(not_bought)).to(device)
        i_tensor = torch.LongTensor(not_bought).to(device)
        scores   = model(u_tensor, i_tensor).cpu().numpy()

    top_idx   = np.argsort(scores)[::-1][:top_n]
    top_items = [item_enc.inverse_transform([not_bought[i]])[0]
                 for i in top_idx]
    top_scores = [scores[i] for i in top_idx]

    recs = []
    for item, score in zip(top_items, top_scores):
        similar = top_k_similar(item, sim_df, k=2)
        pairs   = " | ".join(similar.index.tolist()) if not similar.empty else "—"
        recs.append({
            "Recommended_Item": item,
            "Predicted_Score":  round(float(score), 4),
            "Pairs_Well_With":  pairs
        })
    return pd.DataFrame(recs)


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 7 ─ COMBO REPORT BUILDER                          ║
# ╚══════════════════════════════════════════════════════════════╝

def build_combo_report(
    rules_df: pd.DataFrame,
    item_stats: pd.DataFrame,
    top_n: int = 30
) -> pd.DataFrame:
    pop_lookup = dict(zip(item_stats["Description"],
                          item_stats["Popularity_Score"]))
    rev_lookup = dict(zip(item_stats["Description"],
                          item_stats["Revenue_Ratio"]))

    rows = []
    for _, r in rules_df.head(top_n).iterrows():
        ant_items = list(r["antecedents"])
        con_items = list(r["consequents"])

        ant_pop = np.mean([pop_lookup.get(i, 0) for i in ant_items])
        con_pop = np.mean([pop_lookup.get(i, 0) for i in con_items])
        con_rev = np.mean([rev_lookup.get(i, 0) for i in con_items])

        opportunity = r["lift"] * r["confidence"] * (con_rev + 1e-6)

        rows.append({
            "Trigger_Items":      r["antecedents_str"],
            "Recommended_Items":  r["consequents_str"],
            "Support":            round(r["support"],    4),
            "Confidence":         round(r["confidence"], 4),
            "Lift":               round(r["lift"],       4),
            "Combo_Score":        round(r["Combo_Score"], 6),
            "Trigger_Popularity": round(ant_pop,          4),
            "Reco_Popularity":    round(con_pop,          4),
            "Reco_Revenue_Ratio": round(con_rev,          6),
            "Opportunity_Score":  round(opportunity,      6),
        })

    report = pd.DataFrame(rows).sort_values(
        "Opportunity_Score", ascending=False
    ).reset_index(drop=True)
    report.index += 1
    report.index.name = "Rank"
    return report


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 8 ─ CLUSTER ANALYSIS                              ║
# ╚══════════════════════════════════════════════════════════════╝

def run_clustering(item_embeddings, item_names, coords):
    print("\n[9] Item Cluster Analysis …")

    inertia, sil_scores, K_range = [], [], range(2, 12)
    for k in K_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(item_embeddings)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(item_embeddings, labels))

    best_k = K_range.start + int(np.argmax(sil_scores))
    print(f"  Optimal K : {best_k}")

    km_final      = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(item_embeddings)

    cluster_df = pd.DataFrame({
        "Item":    item_names,
        "Cluster": cluster_labels,
        "PCA_X":   coords[:, 0],
        "PCA_Y":   coords[:, 1]
    })
    cluster_df.to_csv(OUTPUT_DIR / "item_clusters.csv", index=False)
    return cluster_df, best_k


# ╔══════════════════════════════════════════════════════════════╗
# ║  STARTUP  –  Load data & train models once                  ║
# ╚══════════════════════════════════════════════════════════════╝

@app.on_event("startup")
def startup():
    print("=" * 65)
    print("  CONUT AI  –  Pipeline Starting …")
    print("=" * 65)

    ds191 = load_ds191(DATA_DIR / "rep_s_00191_SMRY_labelled.csv")
    ds502 = load_ds502(DATA_DIR / "REP_S_00502_labelled.csv")

    item_stats_191, all_baskets = build_features(ds191, ds502)
    rules, frequent_itemsets    = mine_rules(all_baskets)

    (model, user_enc, item_enc, interactions,
     n_items, device, item_embeddings,
     coords, item_names,
     train_losses, val_losses,
     val_loader) = build_ncf(ds502)

    sim_df     = build_similarity(item_embeddings, item_names)
    combo_rep  = build_combo_report(rules, item_stats_191, top_n=50)
    combo_rep.to_csv(OUTPUT_DIR / "optimal_combinations_report.csv")

    cluster_df, best_k = run_clustering(item_embeddings, item_names, coords)

    # Store everything globally
    STATE.update({
        "ds191":           ds191,
        "ds502":           ds502,
        "item_stats_191":  item_stats_191,
        "all_baskets":     all_baskets,
        "rules":           rules,
        "frequent_itemsets": frequent_itemsets,
        "model":           model,
        "user_enc":        user_enc,
        "item_enc":        item_enc,
        "interactions":    interactions,
        "n_items":         n_items,
        "device":          device,
        "item_embeddings": item_embeddings,
        "coords":          coords,
        "item_names":      item_names,
        "sim_df":          sim_df,
        "combo_report":    combo_rep,
        "cluster_df":      cluster_df,
        "best_k":          best_k,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    })
    print("\n  Pipeline Complete ✓  –  API Ready\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  API ROUTES                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Request / Response schemas ────────────────────────────────

class RecommendRequest(BaseModel):
    customer_name: str
    top_n: Optional[int] = 5

class ComboQueryRequest(BaseModel):
    top_n: Optional[int] = 15

class ForecastRequest(BaseModel):
    branch: Optional[str] = None
    top_n: Optional[int]  = 10

class StaffingRequest(BaseModel):
    branch: str
    expected_orders: Optional[int] = None

class ExpansionRequest(BaseModel):
    top_n: Optional[int] = 5

class StrategyRequest(BaseModel):
    category: Optional[str] = "coffee"   # "coffee" | "milkshake"
    top_n:    Optional[int]  = 10


# ── Health Check ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Conut AI – Chief of Operations Agent",
        "status":  "online",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    ready = len(STATE) > 0
    return {"status": "ready" if ready else "loading", "models_loaded": ready}


# ── 1. Product Combination Recommendations ────────────────────

@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Recommend products for a specific customer using NCF.
    """
    recs = recommend_combinations(
        customer_name = req.customer_name,
        model         = STATE["model"],
        user_enc      = STATE["user_enc"],
        item_enc      = STATE["item_enc"],
        interactions  = STATE["interactions"],
        n_items       = STATE["n_items"],
        device        = STATE["device"],
        sim_df        = STATE["sim_df"],
        top_n         = req.top_n
    )
    return {
        "customer":        req.customer_name.upper(),
        "recommendations": recs.to_dict(orient="records")
    }


# ── 2. Top Product Combos (Association Rules) ─────────────────

@app.post("/combos")
def combos(req: ComboQueryRequest):
    """
    Return top product combinations by Opportunity Score.
    """
    report = STATE["combo_report"].head(req.top_n)
    return {
        "top_combos": report.reset_index().to_dict(orient="records")
    }

@app.get("/combos/top")
def combos_top():
    """Quick GET for top 10 combos."""
    report = STATE["combo_report"].head(10)
    return {
        "top_combos": report.reset_index().to_dict(orient="records")
    }


# ── 3. Demand Forecasting by Branch ──────────────────────────

@app.post("/forecast")
def forecast(req: ForecastRequest):
    """
    Forecast demand per branch using historical sales patterns.
    Returns top items by revenue ratio and popularity.
    """
    ds191 = STATE["ds191"]
    ds502 = STATE["ds502"]

    # Branch-level aggregation from DS-191
    branch_demand = (
        ds191.groupby("Branch")
        .agg(
            Total_Qty     = ("Qty",          "sum"),
            Total_Revenue = ("Total_Amount", "sum"),
            Unique_Items  = ("Description",  "nunique"),
        )
        .reset_index()
    )
    branch_demand["Demand_Index"] = (
        branch_demand["Total_Qty"].rank(pct=True)
    )
    branch_demand = branch_demand.sort_values(
        "Demand_Index", ascending=False
    )

    if req.branch:
        branch_demand = branch_demand[
            branch_demand["Branch"].str.upper() == req.branch.upper()
        ]

    return {
        "branch_demand_forecast": branch_demand.head(req.top_n).to_dict(
            orient="records"
        )
    }


# ── 4. Staffing Recommendations ───────────────────────────────

@app.post("/staffing")
def staffing(req: StaffingRequest):
    """
    Estimate staffing needs for a branch based on order volume
    and item complexity derived from historical data.
    """
    ds502  = STATE["ds502"]
    branch = req.branch.upper()

    branch_data = ds502[ds502["Branch"] == branch]
    if branch_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Branch '{req.branch}' not found in DS-502."
        )

    total_orders   = branch_data["Customer_Name"].nunique()
    avg_basket_size = (
        branch_data.groupby("Customer_Name")["Description"]
        .count()
        .mean()
    )
    unique_items   = branch_data["Description"].nunique()

    # Complexity factor: more unique items → more skilled staff needed
    complexity = round(unique_items / 10, 2)

    # Orders to staff
    expected = req.expected_orders or int(total_orders)
    base_staff   = max(2, int(expected / 20))
    complex_staff = max(1, int(complexity))
    total_staff   = base_staff + complex_staff

    # Peak hour estimation (proxy: top 3 items volume)
    top_items = (
        branch_data.groupby("Description")["Qty"]
        .sum()
        .nlargest(5)
        .reset_index()
        .rename(columns={"Qty": "Total_Qty"})
    )

    return {
        "branch":               branch,
        "historical_orders":    int(total_orders),
        "avg_basket_size":      round(float(avg_basket_size), 2),
        "unique_items_offered": int(unique_items),
        "complexity_factor":    complexity,
        "expected_orders":      expected,
        "recommended_staff":    total_staff,
        "breakdown": {
            "base_staff":      base_staff,
            "complexity_staff": complex_staff
        },
        "top_volume_items": top_items.to_dict(orient="records")
    }


# ── 5. Branch Expansion Feasibility ───────────────────────────

@app.post("/expansion")
def expansion(req: ExpansionRequest):
    """
    Rank branches by performance metrics to identify
    candidates for new branch openings or resource scaling.
    """
    ds191 = STATE["ds191"]
    ds502 = STATE["ds502"]

    # Aggregate DS-191 branch metrics
    branch_191 = (
        ds191.groupby("Branch")
        .agg(
            Total_Qty_191      = ("Qty",              "sum"),
            Total_Revenue_191  = ("Total_Amount",     "sum"),
            Unique_Items_191   = ("Description",      "nunique"),
            Avg_Unit_Price_191 = ("Unit_Price_Ratio", "mean"),
        )
        .reset_index()
    )

    # Aggregate DS-502 branch metrics
    branch_502 = (
        ds502.groupby("Branch")
        .agg(
            Total_Revenue_502  = ("Price",         "sum"),
            Unique_Customers   = ("Customer_Name", "nunique"),
            Total_Orders       = ("Customer_Name", "count"),
            Avg_Order_Value    = ("Price",         "mean"),
        )
        .reset_index()
    )

    merged = pd.merge(
        branch_191, branch_502, on="Branch", how="outer"
    ).fillna(0)

    # Expansion Score: normalised composite
    scaler = MinMaxScaler()
    score_cols = [
        "Total_Revenue_191", "Total_Revenue_502",
        "Unique_Customers",  "Total_Qty_191"
    ]
    merged["Expansion_Score"] = scaler.fit_transform(
        merged[score_cols]
    ).mean(axis=1)

    merged["Expansion_Score"]  = merged["Expansion_Score"].round(4)
    merged["Avg_Order_Value"]   = merged["Avg_Order_Value"].round(2)
    merged["Avg_Unit_Price_191"]= merged["Avg_Unit_Price_191"].round(2)

    result = merged.sort_values(
        "Expansion_Score", ascending=False
    ).head(req.top_n).reset_index(drop=True)

    result.index     += 1
    result.index.name = "Rank"

    return {
        "expansion_candidates": result.reset_index().to_dict(
            orient="records"
        )
    }


# ── 6. Coffee & Milkshake Growth Strategy ─────────────────────

@app.post("/strategy")
def strategy(req: StrategyRequest):
    """
    Analyse growth opportunities for Coffee or Milkshake
    product lines across branches.
    """
    ds191    = STATE["ds191"]
    ds502    = STATE["ds502"]
    category = req.category.lower()

    # Category keyword mapping
    keywords = {
        "coffee":    ["COFFEE", "ESPRESSO", "LATTE", "CAPPUCCINO",
                      "AMERICANO", "MOCHA", "FLAT WHITE", "MACCHIATO"],
        "milkshake": ["MILKSHAKE", "SHAKE", "SMOOTHIE", "FRAPPE",
                      "FRAPPUCCINO", "BLENDED"]
    }

    if category not in keywords:
        raise HTTPException(
            status_code=400,
            detail="Category must be 'coffee' or 'milkshake'."
        )

    kw_pattern = "|".join(keywords[category])

    # Filter DS-191 for category items
    cat_191 = ds191[
        ds191["Description"].str.contains(kw_pattern, case=False, regex=True)
    ].copy()

    # Filter DS-502 for category items
    cat_502 = ds502[
        ds502["Description"].str.contains(kw_pattern, case=False, regex=True)
    ].copy()

    if cat_191.empty and cat_502.empty:
        return {
            "category": category,
            "message":  "No items found for this category.",
            "insights": []
        }

    # DS-191 insights
    branch_perf_191 = (
        cat_191.groupby("Branch")
        .agg(
            Category_Qty     = ("Qty",          "sum"),
            Category_Revenue = ("Total_Amount", "sum"),
            Unique_Cat_Items = ("Description",  "nunique"),
        )
        .reset_index()
    )

    # DS-502 insights
    branch_perf_502 = (
        cat_502.groupby("Branch")
        .agg(
            Cat_Orders    = ("Customer_Name", "count"),
            Cat_Customers = ("Customer_Name", "nunique"),
            Cat_Revenue   = ("Price",         "sum"),
        )
        .reset_index()
    )

    merged_cat = pd.merge(
        branch_perf_191, branch_perf_502,
        on="Branch", how="outer"
    ).fillna(0)

    scaler = MinMaxScaler()
    merged_cat["Growth_Score"] = scaler.fit_transform(
        merged_cat[["Category_Revenue", "Cat_Revenue",
                     "Cat_Customers",    "Category_Qty"]]
    ).mean(axis=1).round(4)

    merged_cat = merged_cat.sort_values(
        "Growth_Score", ascending=False
    ).head(req.top_n).reset_index(drop=True)

    # Top items in category
    top_items_191 = (
        cat_191.groupby("Description")
        .agg(
            Total_Qty     = ("Qty",          "sum"),
            Total_Revenue = ("Total_Amount", "sum"),
        )
        .reset_index()
        .sort_values("Total_Revenue", ascending=False)
        .head(10)
    )

    top_items_502 = (
        cat_502.groupby("Description")
        .agg(
            Total_Orders  = ("Customer_Name", "count"),
            Total_Revenue = ("Price",         "sum"),
        )
        .reset_index()
        .sort_values("Total_Revenue", ascending=False)
        .head(10)
    )

    return {
        "category":           category,
        "branch_growth":      merged_cat.to_dict(orient="records"),
        "top_items_ds191":    top_items_191.to_dict(orient="records"),
        "top_items_ds502":    top_items_502.to_dict(orient="records"),
        "total_cat_qty_191":  int(cat_191["Qty"].sum()),
        "total_cat_rev_191":  round(float(cat_191["Total_Amount"].sum()), 2),
        "total_cat_rev_502":  round(float(cat_502["Price"].sum()),        2),
    }


# ── 7. Item Similarity Lookup ─────────────────────────────────

class SimilarityRequest(BaseModel):
    item_name: str
    top_k: Optional[int] = 5

@app.post("/similar")
def similar_items(req: SimilarityRequest):
    """
    Return items most similar to the queried item
    using NCF embedding cosine similarity.
    """
    sim_df  = STATE["sim_df"]
    item_up = req.item_name.upper()

    if item_up not in sim_df.index:
        # Fuzzy fallback: partial match
        matches = [i for i in sim_df.index if item_up in i]
        if not matches:
            raise HTTPException(
                status_code=404,
                detail=f"Item '{req.item_name}' not found."
            )
        item_up = matches[0]

    similars = top_k_similar(item_up, sim_df, k=req.top_k)
    return {
        "query_item": item_up,
        "similar_items": [
            {"item": item, "similarity_score": round(float(score), 4)}
            for item, score in similars.items()
        ]
    }


# ── 8. Cluster Overview ───────────────────────────────────────

@app.get("/clusters")
def clusters():
    """
    Return item cluster assignments from KMeans on NCF embeddings.
    """
    cluster_df = STATE["cluster_df"]
    best_k     = STATE["best_k"]

    summary = (
        cluster_df.groupby("Cluster")["Item"]
        .apply(list)
        .reset_index()
        .rename(columns={"Item": "Items"})
    )
    summary["Item_Count"] = summary["Items"].apply(len)

    return {
        "optimal_clusters": int(best_k),
        "clusters":         summary.to_dict(orient="records")
    }


# ── 9. Model Performance ──────────────────────────────────────

@app.get("/model/performance")
def model_performance():
    """
    Return NCF training and validation loss history.
    """
    train_losses = STATE["train_losses"]
    val_losses   = STATE["val_losses"]

    return {
        "epochs":       len(train_losses),
        "final_train_mse": round(train_losses[-1], 6),
        "final_val_mse":   round(val_losses[-1],   6),
        "loss_history": [
            {
                "epoch":      i + 1,
                "train_loss": round(t, 6),
                "val_loss":   round(v, 6)
            }
            for i, (t, v) in enumerate(zip(train_losses, val_losses))
        ]
    }


# ── 10. Association Rules Explorer ───────────────────────────

class RulesRequest(BaseModel):
    min_lift:       Optional[float] = 1.2
    min_confidence: Optional[float] = 0.30
    top_n:          Optional[int]   = 20

@app.post("/rules")
def get_rules(req: RulesRequest):
    """
    Filter and return association rules by lift and confidence thresholds.
    """
    rules = STATE["rules"]
    filtered = rules[
        (rules["lift"]       >= req.min_lift) &
        (rules["confidence"] >= req.min_confidence)
    ].head(req.top_n)

    return {
        "total_rules_matching": len(filtered),
        "rules": [
            {
                "trigger":    row["antecedents_str"],
                "recommends": row["consequents_str"],
                "support":    round(row["support"],    4),
                "confidence": round(row["confidence"], 4),
                "lift":       round(row["lift"],       4),
                "combo_score":round(row["Combo_Score"],6),
            }
            for _, row in filtered.iterrows()
        ]
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = int(os.environ.get("PORT", 8000)),
        reload  = False
    )
