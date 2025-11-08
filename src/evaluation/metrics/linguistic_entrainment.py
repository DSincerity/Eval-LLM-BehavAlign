"""Linguistic Entrainment Gap (LEG) analysis using nCLiD (normalized Conversational Linguistic Distance).

This module implements the analysis based on Nasir et al. (2019):
"Modeling Interpersonal Linguistic Coordination in Conversations using Word Mover's Distance"
"""
import numpy as np
import pandas as pd
import bisect
import warnings
import os
import json
import re
from typing import Dict, Any, Tuple, List, Optional
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import download
from tqdm import tqdm

# Ensure stopwords are downloaded
try:
    stopwords.words('english')
except LookupError:
    download('stopwords', quiet=True)

stop_words = stopwords.words('english')


class CLiD:
    """Conversational Linguistic Distance as proposed by Nasir et al (2019).

    Calculates linguistic coordination between speakers using Word Mover's Distance.
    """

    def __init__(self, word2vec):
        """Initialize CLiD calculator.

        Args:
            word2vec: Pre-trained word2vec model (e.g., 'word2vec-google-news-300')
        """
        self.word2vec = word2vec
        self.tokenizer = RegexpTokenizer(r'\w+')

    def _content_preprocess(self, content: str) -> List[str]:
        """Preprocess text content by tokenizing and lowercasing.

        Args:
            content: Raw text string

        Returns:
            List of preprocessed tokens
        """
        content = self.tokenizer.tokenize(content)
        content = [w.lower() for w in content]
        return content

    def cal_dist(
        self,
        clid_dir: str,
        k: int,
        dlg_df: pd.DataFrame,
        therapistName: str = 'buyer',
        patientName: str = 'seller'
    ) -> Tuple[float, float]:
        """Calculate CLiD between two speakers.

        Args:
            clid_dir: Direction ('t2p' or 'p2t') - which speaker coordinates to which
            k: Number of following utterances to consider
            dlg_df: Dialogue dataframe with 'speaker' and 'content' columns
            therapistName: Name of first speaker (default: 'buyer')
            patientName: Name of second speaker (default: 'seller')

        Returns:
            Tuple of (uclid, nclid) - unnormalized and normalized CLiD values
        """
        self.k = k

        # Set anchor and coordinator based on direction
        if clid_dir == 't2p':
            # patientName coordinating to therapistName
            self.anchor = therapistName
            self.crd = patientName
        else:  # p2t
            self.anchor = patientName
            self.crd = therapistName

        # Calculate uCLiD
        uclid = self._cal_uCLiD(dlg_df)
        # Calculate nCLiD
        nclid = self._cal_nCLiD(dlg_df, uclid)

        return uclid, nclid

    def _cal_uCLiD(self, dlg: pd.DataFrame) -> float:
        """Calculate unnormalized CLiD.

        Args:
            dlg: Dialogue dataframe

        Returns:
            Unnormalized CLiD value
        """
        anchor_rows = dlg[dlg['speaker'] == self.anchor]
        crd_row_index = dlg[dlg['speaker'] == self.crd].index.tolist()
        N = 0
        d_sum = 0

        for row_index, anchor_row in anchor_rows.iterrows():
            # Find k following coordinator rows
            pos = bisect.bisect_right(crd_row_index, row_index)
            crd_rows = crd_row_index[pos:pos + self.k]

            if len(crd_rows) == 0:
                continue

            # Calculate minimum distance to k following rows
            di = float('inf')

            for crd_row in crd_rows:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    wmd_now = self.word2vec.wmdistance(
                        self._content_preprocess(anchor_row['content']),
                        self._content_preprocess(dlg.loc[crd_row]['content'])
                    )

                if wmd_now < di:
                    di = wmd_now

            if di == float('inf'):  # Skip illegal anchors
                continue

            d_sum += di
            N += 1

        # Final uCLiD
        if N != 0:
            uclid = d_sum * (1.0 / N)
        else:
            uclid = float('inf')

        return uclid

    def _cal_nCLiD(self, dlg: pd.DataFrame, uclid: float) -> float:
        """Calculate normalized CLiD.

        Args:
            dlg: Dialogue dataframe
            uclid: Unnormalized CLiD value

        Returns:
            Normalized CLiD value
        """
        anchor_rows = dlg[dlg['speaker'] == self.anchor]
        crd_rows = dlg[dlg['speaker'] == self.crd]

        # Calculate alpha_a (within anchor speaker)
        alpha_a = 0
        N_a = 0
        len_a = anchor_rows.shape[0]

        for i in range(len_a):
            for j in range(i + 1, len_a):
                Ai = anchor_rows.iloc[i]['content']
                Aj = anchor_rows.iloc[j]['content']

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    now_alpha = self.word2vec.wmdistance(
                        self._content_preprocess(Ai),
                        self._content_preprocess(Aj)
                    )

                if now_alpha == float('inf'):
                    continue
                N_a += 1
                alpha_a += now_alpha

        if N_a != 0:
            alpha_a = alpha_a / N_a
        else:
            alpha_a = float('inf')

        # Calculate alpha_b (within coordinator speaker)
        alpha_b = 0
        N_b = 0
        len_b = crd_rows.shape[0]

        for i in range(len_b):
            for j in range(i + 1, len_b):
                Bi = crd_rows.iloc[i]['content']
                Bj = crd_rows.iloc[j]['content']

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    now_alpha = self.word2vec.wmdistance(
                        self._content_preprocess(Bi),
                        self._content_preprocess(Bj)
                    )

                if now_alpha == float('inf'):
                    continue
                N_b += 1
                alpha_b += now_alpha

        if N_b != 0:
            alpha_b = alpha_b / N_b
        else:
            alpha_b = float('inf')

        # Calculate alpha_ab (between anchor and coordinator)
        alpha_ab = 0
        N_ab = 0
        crd_row_index = dlg[dlg['speaker'] == self.crd].index.tolist()

        for dlg_i, anchor_row in anchor_rows.iterrows():
            start_j = bisect.bisect_right(crd_row_index, dlg_i)

            if start_j != len(crd_row_index):
                for crd_j in range(start_j, len(crd_row_index)):
                    dlg_j = crd_row_index[crd_j]
                    Ai = anchor_rows.loc[dlg_i]['content']
                    Bj = crd_rows.loc[dlg_j]['content']

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        now_alpha = self.word2vec.wmdistance(
                            self._content_preprocess(Ai),
                            self._content_preprocess(Bj)
                        )

                    if now_alpha == float('inf'):
                        continue
                    N_ab += 1
                    alpha_ab += now_alpha

        if N_ab != 0:
            alpha_ab = alpha_ab / N_ab
        else:
            alpha_ab = float('inf')

        # Calculate final nCLiD
        alpha = alpha_a + alpha_b + alpha_ab

        if alpha != float('inf') and alpha != 0:
            nclid = uclid * (1.0 / alpha)
        else:
            nclid = float('inf')

        return nclid


def remove_inf(arr: np.ndarray) -> np.ndarray:
    """Remove infinite values from array.

    Args:
        arr: Input array

    Returns:
        Array with infinite values removed
    """
    filter_arr = [element != float('inf') for element in arr]
    return arr[filter_arr]


def preprocess_kodis_conversation_from_json(
    kodis_data: Dict,
    kodis_filename: str
) -> pd.DataFrame:
    """Preprocess KODIS conversation from JSON format.

    Args:
        kodis_data: KODIS conversation data (list of turns)
        kodis_filename: Filename to extract KODIS ID from

    Returns:
        Formatted dialogue dataframe
    """
    turns = []

    # Extract KODIS ID from filename
    match = re.match(r"irp_kodis_(\d+).json", kodis_filename)
    kodis_id = match.group(1) if match else kodis_filename

    for turn in kodis_data:
        # Skip termination messages
        if any(x in turn.get("sentence", "") for x in [
            "SUBMISSION", "ACCEPT-DEAL", "REJECT", "WALK-AWAY",
            'Submitted agreement', 'Accept Deal', 'Reject Deal', 'I Walk Away'
        ]):
            continue

        # Map Speaker1 -> buyer, Speaker2 -> seller
        speaker = "buyer" if turn.get("speaker") == "Speaker1" else "seller"
        content = turn.get("sentence", "")

        if content and len(content.strip()) > 0:
            turns.append({
                "speaker": speaker,
                "content": content,
                "kodis-id": kodis_id
            })

    return pd.DataFrame(turns)


def preprocess_model_conversation(conversation: List[Dict[str, str]], conv_id: int, model_name: str) -> pd.DataFrame:
    """Preprocess model conversation into standard format.

    Args:
        conversation: List of turns with 'role' and 'content' keys
        conv_id: Conversation ID
        model_name: Model name for ID column

    Returns:
        Formatted dialogue dataframe
    """
    turns = []
    for turn in conversation:
        # Skip termination messages
        if any(x in turn["content"] for x in [
            "SUBMISSION", "ACCEPT-DEAL", "REJECT", "WALK-AWAY",
            'Submitted agreement', 'Accept Deal', 'Reject Deal', 'I Walk Away'
        ]):
            continue

        speaker = "buyer" if turn["role"] == "Agent2" else "seller"
        turns.append({
            "speaker": speaker,
            "content": turn["content"],
            f"{model_name}-id": conv_id
        })

    return pd.DataFrame(turns)


def calculate_le_values(
    conversations: List[pd.DataFrame],
    word2vec_model,
    id_column: str,
    k: int = 3,
    desc: str = "Calculating LE"
) -> pd.DataFrame:
    """Calculate Linguistic Entrainment (LE) values for a set of conversations.

    Args:
        conversations: List of dialogue dataframes
        word2vec_model: Pre-trained word2vec model
        id_column: Name of ID column in dataframes
        k: Number of following utterances to consider (default: 3)
        desc: Description for progress bar

    Returns:
        DataFrame with LE values and metadata
    """
    model = CLiD(word2vec=word2vec_model)
    seller_to_buyer = []
    buyer_to_seller = []

    for dlg_df in tqdm(conversations, desc=desc, leave=False):
        if len(dlg_df) == 0:
            continue

        # Calculate both directions
        uclid_s2b, nclid_s2b = model.cal_dist(
            clid_dir='t2p',
            k=k,
            dlg_df=dlg_df,
            therapistName='buyer',
            patientName='seller'
        )

        uclid_b2s, nclid_b2s = model.cal_dist(
            clid_dir='p2t',
            k=k,
            dlg_df=dlg_df,
            therapistName='buyer',
            patientName='seller'
        )

        # Store results
        conv_id = dlg_df[id_column].iloc[0]

        seller_to_buyer.append({
            id_column: conv_id,
            'clid_dir': 't2p',
            'uclid': uclid_s2b,
            'nclid': nclid_s2b
        })

        buyer_to_seller.append({
            id_column: conv_id,
            'clid_dir': 'p2t',
            'uclid': uclid_b2s,
            'nclid': nclid_b2s
        })

    # Merge both directions
    bs_df = pd.DataFrame(buyer_to_seller)
    sb_df = pd.DataFrame(seller_to_buyer)
    merged_df = bs_df.merge(sb_df, on=id_column, how="inner")

    # Average both directions for final LE value
    merged_df["both_dir_avg"] = (merged_df["nclid_x"] + merged_df["nclid_y"]) / 2

    # Remove infinite values
    values = merged_df["both_dir_avg"].to_numpy()
    values = remove_inf(values)

    return values


def analyze_linguistic_entrainment(
    kodis_emo_path: str,
    model_data_paths: Dict[str, str],
    output_dir: str = 'data/linguistic_entrainment',
    k: int = 3
) -> Dict[str, Any]:
    """Main function to analyze Linguistic Entrainment Gap (LEG).

    Args:
        kodis_emo_path: Path to KODIS JSON file (data/KODIS-merged_combined_dialogues_irp_emo.json)
        model_data_paths: Dictionary mapping model names to their data JSON paths
        output_dir: Directory to save intermediate LE values
        k: Number of following utterances to consider (default: 3)

    Returns:
        Analysis results including LEG scores for each model
    """
    print("\n" + "=" * 60)
    print("Linguistic Entrainment Gap (LEG) Analysis")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load word2vec model
    print("\nLoading word2vec model (this may take a moment)...")
    try:
        import gensim.downloader as api
        word2vec = api.load('word2vec-google-news-300')
        print("  Word2vec model loaded successfully")
    except Exception as e:
        print(f"  Error loading word2vec model: {e}")
        return {}

    results = {
        'kodis': {},
        'models': {},
        'statistical_tests': {}
    }

    # Process KODIS data
    kodis_le_path = os.path.join(output_dir, 'LE_values_KODIS.csv')

    if os.path.exists(kodis_le_path):
        print(f"\n  Loading cached KODIS LE values from: {kodis_le_path}")
        kodis_le_df = pd.read_csv(kodis_le_path)
        kodis_le_values = kodis_le_df.iloc[:, 0].to_numpy()
        print(f"    Loaded {len(kodis_le_values)} LE values")
    else:
        print(f"\n  Computing KODIS LE values...")
        print(f"  Loading KODIS data from: {kodis_emo_path}")

        # Load KODIS conversations from JSON
        with open(kodis_emo_path, 'r') as f:
            kodis_data = json.load(f)

        # Preprocess conversations
        kodis_conversations = []
        for kodis_filename, conversation in tqdm(
            kodis_data.items(),
            desc="  Preprocessing KODIS",
            leave=False
        ):
            conv_df = preprocess_kodis_conversation_from_json(conversation, kodis_filename)
            if len(conv_df) > 0:
                kodis_conversations.append(conv_df)

        print(f"    Preprocessed {len(kodis_conversations)} conversations")

        # Calculate LE values
        kodis_le_values = calculate_le_values(
            kodis_conversations,
            word2vec,
            id_column='kodis-id',
            k=k,
            desc="  Computing KODIS LE"
        )

        # Save to cache
        pd.DataFrame({f"LE Values - KODIS": kodis_le_values}).to_csv(kodis_le_path, index=False)
        print(f"    Saved LE values to: {kodis_le_path}")
        print(f"    Computed {len(kodis_le_values)} LE values")

    # Store KODIS results
    results['kodis'] = {
        'mean': float(np.mean(kodis_le_values)),
        'std': float(np.std(kodis_le_values)),
        'count': len(kodis_le_values)
    }
    print(f"  KODIS Mean LE: {results['kodis']['mean']:.4f}")

    # Process each model
    for model_name, model_path in model_data_paths.items():
        print(f"\n  Processing {model_name}...")

        # Check for cached LE values
        model_le_path = os.path.join(output_dir, f'LE_values_{model_name}.csv')

        if os.path.exists(model_le_path):
            print(f"    Loading cached LE values from: {model_le_path}")
            model_le_df = pd.read_csv(model_le_path)
            model_le_values = model_le_df.iloc[:, 0].to_numpy()
            print(f"    Loaded {len(model_le_values)} LE values")
        else:
            print(f"    Computing LE values from: {model_path}")

            # Load model conversations
            with open(model_path, 'r') as f:
                model_data = json.load(f)

            # Preprocess conversations
            model_conversations = []
            for conv_id, conv in enumerate(tqdm(
                model_data['conversation'],
                desc=f"    Preprocessing {model_name}",
                leave=False
            )):
                conv_df = preprocess_model_conversation(conv, conv_id, model_name)
                if len(conv_df) > 0:
                    model_conversations.append(conv_df)

            print(f"      Preprocessed {len(model_conversations)} conversations")

            # Calculate LE values
            model_le_values = calculate_le_values(
                model_conversations,
                word2vec,
                id_column=f'{model_name}-id',
                k=k,
                desc=f"    Computing {model_name} LE"
            )

            # Save to cache
            pd.DataFrame({f"LE Values - {model_name}": model_le_values}).to_csv(model_le_path, index=False)
            print(f"      Saved LE values to: {model_le_path}")
            print(f"      Computed {len(model_le_values)} LE values")

        # Calculate LEG (Linguistic Entrainment Gap)
        model_mean = float(np.mean(model_le_values))
        kodis_mean = results['kodis']['mean']
        leg = abs(kodis_mean - model_mean)

        # Store model results
        results['models'][model_name] = {
            'mean': model_mean,
            'std': float(np.std(model_le_values)),
            'count': len(model_le_values),
            'leg': leg
        }

        print(f"    {model_name} Mean LE: {model_mean:.4f}")
        print(f"    LEG: {leg:.4f}")

        # Statistical test
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(model_le_values, kodis_le_values, equal_var=False)
        results['statistical_tests'][model_name] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val)
        }
        print(f"    T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")

    print("\n✓ Linguistic entrainment gap analysis complete!")

    return results
