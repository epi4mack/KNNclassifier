import numpy as np
from collections import Counter

class KNNclassifier:
    def __init__(self, k):
        self.k = k

    @staticmethod
    def get_distance(a, b):
        return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

    def fit(self, data):
        temp_data = tuple(list(data.loc[i]) for i in range(len(data)))

        self.data = {
            'Кофе': [],
            'Чай': []
        }
        
        for point in temp_data:
            preference = 'Чай' if point[3] else 'Кофе'
            point.pop(3)
            self.data[preference].append(point)

    def predict(self, new_point):
        distances = []

        for preference in self.data:
            for point in self.data[preference]:
                distance = KNNclassifier.get_distance(point, new_point)
                distances.append((distance, preference))

        preferences = [distance[1] for distance in sorted(distances)[:self.k]]
        result = Counter(preferences).most_common(1)[0][0]
        return result
