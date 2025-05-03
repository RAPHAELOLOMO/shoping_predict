import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Months mapped to integer indices
MONTHS = {
    "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3,
    "May": 4, "June": 5, "Jul": 6, "Aug": 7,
    "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
}


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and return a tuple (evidence, labels).

    evidence should be a list of lists. Each list should contain the following data types:
        - int, float, int, float, int, float, float, float, float, float,
          int, int, int, int, int, int, int

    labels should be a list of integers, where each integer is:
        - 1 if Revenue is TRUE
        - 0 otherwise
    """
    evidence = []
    labels = []

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                MONTHS[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given evidence and labels, return a fitted k=1 KNN model.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given actual labels and predicted labels, return a tuple (sensitivity, specificity).

    sensitivity: True Positive Rate (correctly identified revenue sessions / all actual revenue sessions)
    specificity: True Negative Rate (correctly identified no-revenue sessions / all actual no-revenue sessions)
    """
    true_positives = sum(1 for actual, pred in zip(labels, predictions) if actual == pred == 1)
    true_negatives = sum(1 for actual, pred in zip(labels, predictions) if actual == pred == 0)

    total_positives = sum(1 for label in labels if label == 1)
    total_negatives = sum(1 for label in labels if label == 0)

    sensitivity = true_positives / total_positives if total_positives else 0
    specificity = true_negatives / total_negatives if total_negatives else 0

    return sensitivity, specificity


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data
    evidence, labels = load_data(sys.argv[1])

    # Split into train and test (60% train, 40% test)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=0.4
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate predictions
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    correct = sum(1 for actual, pred in zip(y_test, predictions) if actual == pred)
    incorrect = len(y_test) - correct
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


if __name__ == "__main__":
    main()
