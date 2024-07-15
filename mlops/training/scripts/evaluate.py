from difflib import SequenceMatcher

import pandas as pd
import seaborn as sns
import spacy
from matplotlib import pyplot as plt

nlp = spacy.load("en_core_web_lg")


def predict_cats(
    text: str,
) -> list[str]:
    """Predicts textCat labels.

    Works for binary, multiclass, and multilabel

    Args:
        text (str): text

    Returns:
        list[str]: a list of predicted labels.
    """
    doc = nlp(text)

    # A weak classifier's scores, might be just above the minimum score
    # e.g. {label1: 0.2, label2: 0.2, label3: 0.2, label1: 0.4}
    minimum_score = round(
        len(doc.cats.keys()) / 100,
        2,
    )

    labels = [k for k, v in doc.cats.items() if round(v, 2) > minimum_score]

    return labels


def predict_ners(
    text: str,
    ents_target: list[str],
) -> dict:
    """Predict ners.

    Args:
        text (str): text
        ents_target (list[str]): list of target entities

    Returns:
        dict: found entities with start and end char
    """
    doc = nlp(text)

    entities = {ent: list() for ent in ents_target}

    for ent in doc.ents:
        if ent.label_.lower() in ents_target:
            entities[ent.label_].append(
                {
                    ent.text: {
                        "start_char": ent.start_char + 1,
                        "end_char": ent.end_char,
                    }
                }
            )

    return entities


def calculate_cats_accuracy(
    labels_gt: dict,
    labels_predicted: list,
) -> float:
    """Calculates accuracy.

    Args:
        labels_gt (dict): ground truth labels as {'LABEL1': 0.0, 'LABEL2': 1.0}
        labels_predicted (list): predicted labels

    Returns:
        float: accuracy score of each example [0, 1]
    """
    # Create a list of GT labels that are equal to 1.0
    labels_gt = [k for k, v in labels_gt.items() if v == 1.0]

    # Find the intersection between GT and predicted
    intersection = set(labels_gt).intersection(labels_predicted)

    # How many labels are predicted correctly?
    accuracy = len(intersection) / len(labels_gt)

    return accuracy


def calculate_intersection_percentage(
    gt: list[str],
    found: list[str],
) -> float:
    """Calculates the intersection percentage between 2 lists.

    Args:
        gt (list[str]): ground truth list of text
        found (list[str]): extracted list of text

    Returns:
        float: percentage of overlap. Returns 100% if both lists are "None"
    """
    if len(found) > 0:
        intersection = set(gt).intersection(set(found))

        percentage = round(
            (len(intersection) / len(gt)) * 100,
            0,
        )

    else:
        percentage = 0.0

    return percentage


def plot_intersection_percentage(
    eval: list[float],
) -> None:
    """Plots calculated intersection percentage in 10 bins.

    Args:
        eval (list[float]): validation scores
    """
    sns.histplot(
        data=eval,
        bins=10,
    )

    plt.xlabel("Accuracy")
    plt.ylabel("Number of Products")
    plt.xticks(list(range(0, 110, 10)))
    plt.legend()

    print(eval.value_counts(normalize=True))


def check_symmetric_difference(
    gt: list[str],
    found: list[str],
) -> list[str]:
    """Find symmetric difference between 2 lists.

    Args:
        gt (list[str]): ground truth list of text
        found (list[str]): extracted list of text

    Returns:
        list[str]: Symmetric difference of ground truth and found
    """
    symmetric_difference = set(gt).symmetric_difference(
        set(found),
    )

    # No symmetric difference (i.e., identical)
    if symmetric_difference:
        return True

    # There is a symmetric difference (i.e., not identical)
    else:
        return False


def get_symmetric_diff(
    gt: pd.Series,
    found: pd.Series,
) -> pd.DataFrame:
    """Gets symmetric difference between two series.

    Args:
        gt (pd.Series): ground truth text
        found (pd.Series): found text

    Returns:
        pd.DataFrame: dataframe
    """
    df = pd.concat(
        [gt, found],
        axis="columns",
    )

    # Check if the 2 columns has symmetric difference (True/False)
    # The function is applied on a Series not a DataFrame
    # Therefore, take first row iloc[0], and second row iloc[1]
    df["is_symmetric_diff"] = df.apply(
        lambda s: check_symmetric_difference(
            s_gt=s.iloc[0],
            s_found=s.iloc[1],
        ),
        axis=1,
    )

    # Only keep rows with symmetric diff
    df_symmetric_diff = df.query("`is_symmetric_diff` == True")

    # Only keep first 2 columns
    return df_symmetric_diff.iloc[:, 0:2]


def calculate_similarity_ratio(
    gt: str,
    found: str,
) -> float:
    """Calculates similarity ratio of strings.

    Args:
        gt (str): ground truth text
        found (str): found text

    Returns:
        float: similarity score
    """
    score = SequenceMatcher(gt, found)
    ratio = score * 100

    return ratio
