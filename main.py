import pandas as pd
from sklearn.preprocessing import LabelEncoder
from KNNclassifier import KNNclassifier

if __name__ == '__main__':
    train_dataset = pd.read_excel('Dataset.xlsx')

    train_dataset = train_dataset.drop(columns=[
        'Как часто вы берете инициативу в свои руки? / Баллы',
        'Как часто вы пропускаете завтраки? / Баллы',
        'Какая культура ближе / Баллы',
        'Выпиваете алкоголь / Баллы',
        'Любимое время года? / Баллы',
        'Что пьют родители / Баллы',
        'Какие напитки любите / Баллы',
        'Формат работы / Баллы',
        'Набрано баллов',
        'Всего баллов',
        'Результат теста'
        ])

    for column in list(train_dataset.columns):
        if column not in ('Возраст', 'Сколько спите ночью в среднем', 'Время подъема'):
            le = LabelEncoder()
            train_dataset[column] = le.fit_transform(train_dataset[column])

    new_data = pd.read_excel('test_data.xlsx')

    for column in list(new_data.columns):
            if column not in ('Возраст', 'Сколько спите ночью в среднем', 'Время подъема'):
                le = LabelEncoder()
                new_data[column] = le.fit_transform(new_data[column])

    size = len(new_data)

    knn_classifier = KNNclassifier()
    knn_classifier.fit(train_dataset)

    results = {}

    for k in range(1, 38, 2):
        correct = 0
        for i in range(size):
            test_point = list(new_data.loc[i])
            actual_preference = 'Чай' if test_point.pop(3) else 'Кофе'
            prediction = knn_classifier.predict(test_point, k)

            if prediction == actual_preference:
                correct += 1

        results[k] = round(correct/size, 2)
        
    for k in dict(sorted(results.items(), key=lambda item: item[1], reverse=True)):
        print(f'Точность при k={k} равна {results[k]}')
    