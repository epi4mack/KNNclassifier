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

    k = 2
    knn_classifier = KNNclassifier(k)
    knn_classifier.fit(train_dataset)

    # Тестирование на тренировочных данных

    size = len(train_dataset)
    correct = 0
    for i in range(size):
        test_point = list(train_dataset.loc[i])
        actual_preference = 'Чай' if test_point.pop(3) else 'Кофе'
        prediction = knn_classifier.predict(test_point)

        if prediction == actual_preference:
            correct += 1
            marker = '+'
        else:
            marker = '-'

        print(f'Для точки {i+1} предсказано предпочтение: {prediction} ({marker});')

    print(f'\nДанная модель имеет точность в {correct / size * 100:.2f}% при k={k}. Тестирование произведено на тренировочных данных.\n')

    # Тестирование на новых данных

    new_data = pd.read_excel('test_data.xlsx')

    for column in list(new_data.columns):
        if column not in ('Возраст', 'Сколько спите ночью в среднем', 'Время подъема'):
            le = LabelEncoder()
            new_data[column] = le.fit_transform(new_data[column])

    size = len(new_data)
    correct = 0
    for i in range(size):
        test_point = list(new_data.loc[i])
        actual_preference = 'Чай' if test_point.pop(3) else 'Кофе'
        prediction = knn_classifier.predict(test_point)

        if prediction == actual_preference:
            correct += 1
            marker = '+'
        else:
            marker = '-'

        print(f'Для точки {i+1} предсказано предпочтение: {prediction} ({marker});')

    print(f'\nДанная модель имеет точность в {correct / size * 100:.2f}% при k={k}. Тестирование произведено на новых данных.')
