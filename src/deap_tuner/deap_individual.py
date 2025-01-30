import numpy as np
from sklearn.metrics import f1_score

# Define DEAP toolbox callbacks
def create_individual_cb():
    classifier_types = ['svm', 'dt', 'logistic']
    return [
        np.random.choice(classifier_types),  # Classifier type
        np.random.uniform(0, 1),  # False negative shame (missed-frauds)
        np.random.uniform(0, 1),  # False positive shame (false-alarms)
        np.random.uniform(0, 1),  # True negative pride (allowed non-frauds)
        np.random.uniform(0, 1),  # True positive pride (prevented frauds)
    ]

def mutate_individual_cb(individual):
    # Randomly mutate the traits of an individual
    trait_idx = np.random.randint(1, len(individual))
    individual[trait_idx] = np.random.uniform(0, 1)
    return individual,

def crossover_individual_cb(ind1, ind2):
    # Single point crossover: exchange parts of the individuals
    cxpoint = np.random.randint(1, len(ind1))
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2


def _transform_data(train_data, test_data, classifier_type):
    if classifier_type == 'svm':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
    elif classifier_type == 'dt':
        # Decision trees generally don't need scaling
        pass
    elif classifier_type == 'logistic':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    return train_data, test_data


def evaluate_individual_cb(individual, input_dict):

    # Retrieve input data
    train_data = input_dict['train_data']
    train_labels = input_dict['train_labels']
    test_data = input_dict['test_data']
    true_labels = input_dict['true_labels']

    # Retrieve the individual's traits
    classifier_type = individual[0]
    false_neg_shame, false_pos_shame, true_neg_pride, true_pos_pride = individual[1:]
    threshold = get_threshold(false_neg_shame, false_pos_shame, true_neg_pride, true_pos_pride)
    
    # Transform data based on the classifier type
    train_data, test_data = _transform_data(train_data, test_data, classifier_type)

    # Create and train classifier
    classifier = Classifier(classifier_type)
    classifier.fit(train_data, train_labels)

    # Predict on test data
    predictions = classifier.predict(test_data, threshold)

    # Calculate fitness
    fitness = f1_score(true_labels, predictions)

    # Return as the individual's score
    return fitness,


class Classifier:
    def __init__(self, classifier_type: str):
        self.classifier = self._create_classifier(classifier_type)
    
    def _create_classifier(self, classifier_type: str):
        if classifier_type == 'svm':
            from sklearn.svm import SVC
            return SVC()
        elif classifier_type == 'dt':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier()
        elif classifier_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression()
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def fit(self, train_data, train_labels):
        self.classifier.fit(train_data, train_labels)

    def predict(self, test_data, threshold: float):
        raw_predictions = self.classifier.predict(test_data)
        return (raw_predictions >= threshold).astype(int)
